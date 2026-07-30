use anyhow::Result;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};

pub const DEFAULT_MODEL: &str = "lightonai/LateOn-Code-edge";

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "model_int8.onnx",
    "tokenizer.json",
    "config_sentence_transformers.json",
    "onnx_config.json",
];

/// Optional files (non-quantized model and model variants without this file)
const OPTIONAL_FILES: &[&str] = &["config.json", "model.onnx"];

/// A Hugging Face model ID must be used as an ID, never interpreted as a path.
///
/// `hf-hub` normalizes `/` to `--` in its cache directory names. IDs containing
/// `--` are deliberately rejected because that normalization is not injective
/// (for example, `owner/a--b` and `owner/a/b` would otherwise share a cache
/// directory).
fn is_safe_model_id(model_id: &str) -> bool {
    if model_id.is_empty()
        || model_id.contains("--")
        || model_id.contains('\\')
        || model_id.contains('\0')
    {
        return false;
    }

    let components: Vec<&str> = model_id.split('/').collect();
    if components.len() > 2
        || components
            .iter()
            .any(|component| component.is_empty() || *component == "." || *component == "..")
    {
        return false;
    }

    !Path::new(model_id).is_absolute()
}

fn is_real_directory(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| {
            let file_type = metadata.file_type();
            file_type.is_dir() && !file_type.is_symlink()
        })
        .unwrap_or(false)
}

fn is_real_regular_file(path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    let file_type = metadata.file_type();
    file_type.is_file() && !file_type.is_symlink() && File::open(path).is_ok()
}

fn is_exact_child(path: &Path, parent: &Path, name: &str) -> bool {
    path.parent() == Some(parent) && path.file_name() == Some(name.as_ref())
}

fn is_safe_revision(revision: &str) -> bool {
    if revision.is_empty()
        || revision.contains('/')
        || revision.contains('\\')
        || revision.contains('\0')
    {
        return false;
    }

    let mut components = Path::new(revision).components();
    matches!(components.next(), Some(Component::Normal(component)) if component == revision)
        && components.next().is_none()
}

/// Read the current ref without canonicalizing it or following a ref symlink.
fn read_current_revision(repo_path: &Path) -> Option<String> {
    let refs_path = repo_path.join("refs");
    if !is_real_directory(&refs_path) {
        return None;
    }

    let ref_path = refs_path.join("main");
    let metadata = fs::symlink_metadata(&ref_path).ok()?;
    let file_type = metadata.file_type();
    if !file_type.is_file() || file_type.is_symlink() {
        return None;
    }

    let revision = fs::read_to_string(ref_path).ok()?.trim().to_owned();
    is_safe_revision(&revision).then_some(revision)
}

/// Validate a required snapshot file.
///
/// A required file may be a regular file in the snapshot or a normal HF cache
/// pointer to a regular blob in this repository's `blobs` directory. Nothing
/// else is accepted.
fn validate_required_file(path: &Path, repo_path: &Path) -> bool {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return false;
    };
    let file_type = metadata.file_type();

    if file_type.is_file() && !file_type.is_symlink() {
        return File::open(path).is_ok();
    }
    if !file_type.is_symlink() {
        return false;
    }

    let blobs_path = repo_path.join("blobs");
    if !is_real_directory(&blobs_path) {
        return false;
    }
    let Ok(blobs_canonical) = blobs_path.canonicalize() else {
        return false;
    };
    if !is_exact_child(&blobs_canonical, repo_path, "blobs") {
        return false;
    }

    let Ok(resolved) = path.canonicalize() else {
        return false;
    };
    if resolved.parent() != Some(blobs_canonical.as_path()) {
        return false;
    }

    is_real_regular_file(&resolved)
}

/// Return a complete snapshot from the exact cache repository/ref represented
/// by `model_id`, or `None` so the caller can use the normal authenticated API.
///
/// The cache API returns file paths rather than a snapshot path. Validate the
/// raw current ref and every required path before returning the snapshot.
fn cached_snapshot(model_id: &str, cache: &Cache) -> Option<PathBuf> {
    if !is_safe_model_id(model_id) {
        return None;
    }

    // Canonicalize the cache root independently. The repository must be the
    // exact direct child for this model, not a symlink or an escaped path.
    let cache_root = cache.path().canonicalize().ok()?;
    let folder_name = Repo::model(model_id.to_string()).folder_name();
    let repo_path = cache_root.join(&folder_name);
    if !is_real_directory(&repo_path) {
        return None;
    }

    // Check the raw ref identity before canonicalizing any selected snapshot.
    let revision = read_current_revision(&repo_path)?;
    let repo_path = repo_path.canonicalize().ok()?;
    if !is_exact_child(&repo_path, &cache_root, &folder_name) {
        return None;
    }

    let snapshots_path = repo_path.join("snapshots");
    if !is_real_directory(&snapshots_path) {
        return None;
    }
    let snapshots_path = snapshots_path.canonicalize().ok()?;
    if !is_exact_child(&snapshots_path, &repo_path, "snapshots") {
        return None;
    }

    let snapshot_path = snapshots_path.join(&revision);
    if !is_real_directory(&snapshot_path) {
        return None;
    }
    let snapshot_path = snapshot_path.canonicalize().ok()?;
    if !is_exact_child(&snapshot_path, &snapshots_path, &revision) {
        return None;
    }

    // Use a cache rooted at the canonical root so CacheRepo paths can be
    // compared byte-for-byte with the raw current snapshot path above.
    let repo = Cache::new(cache_root).model(model_id.to_string());
    for file in REQUIRED_FILES {
        let expected_path = snapshot_path.join(file);
        let path = repo.get(file)?;
        if path != expected_path || !validate_required_file(&path, &repo_path) {
            return None;
        }
    }

    Some(snapshot_path)
}

fn cached_model_snapshot(model_id: &str) -> Option<PathBuf> {
    cached_snapshot(model_id, &Cache::from_env())
}

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
/// The `quiet` parameter is kept for API compatibility but no longer used
/// (output is now handled in IndexBuilder::ensure_model_created after ONNX runtime init).
pub fn ensure_model(model_id: Option<&str>, _quiet: bool) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    // Check if it's a local path. Keep this before model-ID validation so an
    // explicit local directory continues to work even when its spelling is not
    // a valid remote ID.
    let local_path = PathBuf::from(model_id);
    if local_path.exists() && local_path.is_dir() {
        return Ok(local_path);
    }

    if !is_safe_model_id(model_id) {
        return Err(anyhow::anyhow!(
            "Invalid Hugging Face model ID `{model_id}`; expected one or two safe path components"
        ));
    }

    // A complete, exact Hugging Face snapshot is already available in the
    // local cache. Do not use an arbitrary model directory or a guessed repo.
    if let Some(snapshot) = cached_model_snapshot(model_id) {
        return Ok(snapshot);
    }

    // Download from HuggingFace. ApiRepo is cache-first, so validate the
    // resulting cache again instead of trusting any returned path parent.

    // Build API with token from environment variables or token file
    // Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > token file ($HF_HOME/token or ~/.cache/huggingface/token)
    let mut builder = ApiBuilder::from_env();
    let token_from_env = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|t| t.trim_matches('"').trim_matches('\'').to_string());
    if token_from_env.is_some() {
        builder = builder.with_token(token_from_env);
    }
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());

    // Download all required files (cached if already present).
    for file in REQUIRED_FILES {
        repo.get(file)?;
    }

    // Try to download optional files (non-quantized model and config) - ignore errors.
    for file in OPTIONAL_FILES {
        let _ = repo.get(file);
    }

    cached_model_snapshot(model_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Hugging Face cache validation failed for model `{model_id}` after download"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn set(name: &'static str, value: &Path) -> Self {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            Self { name, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }

    fn make_layout(temp: &TempDir, model_id: &str) -> (Cache, PathBuf, PathBuf) {
        let cache_path = temp.path().join("hub");
        let repo_path = cache_path.join(Repo::model(model_id.to_string()).folder_name());
        let snapshot = repo_path.join("snapshots").join("revision-1");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repo_path.join("refs")).unwrap();
        fs::write(repo_path.join("refs/main"), "revision-1").unwrap();
        (Cache::new(cache_path), repo_path, snapshot)
    }

    fn make_cache(temp: &TempDir, model_id: &str, files: &[&str]) -> Cache {
        let (cache, _repo_path, snapshot) = make_layout(temp, model_id);
        for file in files {
            fs::write(snapshot.join(file), b"cached model file").unwrap();
        }
        cache
    }

    fn expected_snapshot(cache: &Cache, model_id: &str) -> PathBuf {
        cache
            .path()
            .canonicalize()
            .unwrap()
            .join(Repo::model(model_id.to_string()).folder_name())
            .join("snapshots/revision-1")
    }

    #[test]
    fn ensure_model_returns_exact_current_canonical_snapshot() {
        let _guard = env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let model_id = "test-owner/complete-model";
        let cache = make_cache(&temp, model_id, REQUIRED_FILES);
        let _hf_home = EnvGuard::set("HF_HOME", temp.path());

        let expected = expected_snapshot(&cache, model_id);
        assert_eq!(ensure_model(Some(model_id), true).unwrap(), expected);
    }

    #[test]
    fn cache_accepts_optional_config_absence_and_rejects_missing_required_file() {
        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/requested";
        let cache = make_cache(&temp, model_id, REQUIRED_FILES);

        let snapshot = cached_snapshot(model_id, &cache).expect("complete cache");
        assert!(!snapshot.join("config.json").exists());

        fs::remove_file(snapshot.join(REQUIRED_FILES[0])).unwrap();
        assert!(cached_snapshot(model_id, &cache).is_none());
    }

    #[test]
    fn cache_requires_exact_model_id() {
        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/requested";
        let cache = make_cache(&temp, model_id, REQUIRED_FILES);

        assert!(cached_snapshot(model_id, &cache).is_some());
        assert!(cached_snapshot("owner/other", &cache).is_none());
    }

    #[test]
    fn is_safe_model_id_covers_all_id_shapes() {
        let cases = [
            ("", false),
            ("/absolute", false),
            ("owner\\model", false),
            ("owner/", false),
            ("owner//model", false),
            (".", false),
            ("..", false),
            ("owner/.", false),
            ("owner/..", false),
            ("owner/a/b", false),
            ("owner/a--b", false),
            ("model", true),
            ("owner/model", true),
        ];

        for (model_id, expected) in cases {
            assert_eq!(is_safe_model_id(model_id), expected, "{model_id:?}");
        }
    }

    #[test]
    fn malformed_remote_id_returns_explicit_error() {
        let error = ensure_model(Some("owner/a--b"), true).unwrap_err();
        assert!(error.to_string().contains("Invalid Hugging Face model ID"));
    }

    #[test]
    fn explicit_local_directory_is_checked_before_remote_id_validation() {
        let temp = tempfile::tempdir().unwrap();
        let local_path = temp.path().to_str().unwrap();
        assert_eq!(ensure_model(Some(local_path), true).unwrap(), temp.path());
    }

    #[cfg(unix)]
    #[test]
    fn cache_accepts_realistic_hf_blob_symlinks() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/blob-model";
        let (cache, repo_path, snapshot) = make_layout(&temp, model_id);
        fs::create_dir_all(repo_path.join("blobs")).unwrap();

        for file in REQUIRED_FILES {
            let blob_name = format!("{file}-blob");
            fs::write(repo_path.join("blobs").join(&blob_name), b"blob model file").unwrap();
            symlink(
                Path::new("../../blobs").join(&blob_name),
                snapshot.join(file),
            )
            .unwrap();
        }

        assert_eq!(
            cached_snapshot(model_id, &cache),
            Some(snapshot.canonicalize().unwrap())
        );
    }

    #[cfg(unix)]
    #[test]
    fn cache_rejects_repo_directory_symlink_escape() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/repo-link";
        let (cache, repo_path, _snapshot) = make_layout(&temp, model_id);
        let outside = temp.path().join("outside-repo");
        fs::create_dir_all(&outside).unwrap();
        fs::remove_dir_all(&repo_path).unwrap();
        symlink(&outside, &repo_path).unwrap();

        assert!(cached_snapshot(model_id, &cache).is_none());
    }

    #[test]
    fn cache_rejects_raw_ref_traversal() {
        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/ref-traversal";
        let (cache, repo_path, _snapshot) = make_layout(&temp, model_id);
        fs::write(repo_path.join("refs/main"), "../escape").unwrap();

        assert!(cached_snapshot(model_id, &cache).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn cache_rejects_symlinked_snapshot() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/snapshot-link";
        let (cache, _repo_path, snapshot) = make_layout(&temp, model_id);
        let outside = temp.path().join("outside-snapshot");
        fs::create_dir_all(&outside).unwrap();
        for file in REQUIRED_FILES {
            fs::write(outside.join(file), b"outside model file").unwrap();
        }
        fs::remove_dir_all(&snapshot).unwrap();
        symlink(&outside, &snapshot).unwrap();

        assert!(cached_snapshot(model_id, &cache).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn cache_rejects_required_file_symlink_outside_exact_repo() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let model_id = "owner/requested";
        let cache = make_cache(&temp, model_id, REQUIRED_FILES);
        let snapshot = cached_snapshot(model_id, &cache).unwrap();
        let outside = temp.path().join("outside");
        fs::write(&outside, b"not the model").unwrap();
        let file = snapshot.join(REQUIRED_FILES[0]);
        fs::remove_file(&file).unwrap();
        symlink(&outside, &file).unwrap();

        assert!(cached_snapshot(model_id, &cache).is_none());
    }
}
