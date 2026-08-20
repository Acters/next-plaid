//! Read-only index loading must never prepare or repair an index on disk.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use next_plaid::index::MmapIndex;
use next_plaid::IndexConfig;
use rand::rngs::StdRng;
use serde_json::json;
use tempfile::TempDir;

#[derive(Debug, PartialEq, Eq)]
struct FileSnapshot {
    bytes: Vec<u8>,
    modified: SystemTime,
}

fn snapshot_files(root: &Path) -> BTreeMap<PathBuf, FileSnapshot> {
    fn visit(root: &Path, current: &Path, files: &mut BTreeMap<PathBuf, FileSnapshot>) {
        for entry in fs::read_dir(current).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                let metadata = fs::metadata(&path).unwrap();
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    FileSnapshot {
                        bytes: fs::read(&path).unwrap(),
                        modified: metadata.modified().unwrap(),
                    },
                );
            }
        }
    }

    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

fn prepared_index() -> TempDir {
    prepared_index_with_batch(64)
}

fn prepared_index_with_batch(batch_size: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let documents: Vec<Array2<f32>> = (0..12)
        .map(|_| Array2::random_using((3, 8), StandardNormal, &mut rng))
        .collect();
    let config = IndexConfig {
        nbits: 2,
        batch_size,
        seed: Some(42),
        force_cpu: true,
        ..Default::default()
    };

    MmapIndex::create_with_kmeans(&documents, dir.path().to_str().unwrap(), &config).unwrap();
    dir
}

fn assert_read_only_failure_without_writes(dir: &TempDir) {
    let before = snapshot_files(dir.path());
    let error = match MmapIndex::load_read_only(dir.path().to_str().unwrap()) {
        Ok(_) => panic!("read-only load unexpectedly accepted an unprepared index"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("Read-only index load rejected"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("normal one-shot search") && message.contains("colgrep init"),
        "error must explain how to prepare the index: {message}"
    );
    assert_eq!(before, snapshot_files(dir.path()));
}

#[test]
fn prepared_index_loads_read_only_without_changing_files() {
    let dir = prepared_index();
    let before = snapshot_files(dir.path());

    let index = MmapIndex::load_read_only(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(index.num_documents(), 12);
    assert_eq!(index.num_embeddings(), 36);
    drop(index);

    assert_eq!(before, snapshot_files(dir.path()));
}

#[test]
fn missing_inverse_norm_sidecar_falls_back_without_writes() {
    let dir = prepared_index();
    fs::remove_file(dir.path().join("merged_inv_norms.npy")).unwrap();
    fs::remove_file(dir.path().join("merged_inv_norms.manifest.json")).unwrap();
    let before = snapshot_files(dir.path());

    let index = MmapIndex::load_read_only(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(index.num_documents(), 12);
    drop(index);

    assert_eq!(before, snapshot_files(dir.path()));
}

#[test]
fn stale_inverse_norm_manifest_falls_back_without_writes() {
    let dir = prepared_index();
    let manifest_path = dir.path().join("merged_inv_norms.manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_reader(fs::File::open(&manifest_path).unwrap()).unwrap();
    manifest["metadata_mtime"] = json!(0.0);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let before = snapshot_files(dir.path());

    let index = MmapIndex::load_read_only(dir.path().to_str().unwrap()).unwrap();
    assert_eq!(index.num_embeddings(), 36);
    drop(index);

    assert_eq!(before, snapshot_files(dir.path()));
}

#[test]
fn compatibility_conversion_is_rejected_without_writes() {
    let dir = prepared_index();
    let metadata_path = dir.path().join("metadata.json");
    let mut metadata: serde_json::Value =
        serde_json::from_reader(fs::File::open(&metadata_path).unwrap()).unwrap();
    metadata["next_plaid_compatible"] = json!(false);
    fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

    assert_read_only_failure_without_writes(&dir);
}

#[test]
fn missing_merged_files_are_rejected_without_writes() {
    for filename in ["merged_codes.npy", "merged_residuals.npy"] {
        let dir = prepared_index();
        fs::remove_file(dir.path().join(filename)).unwrap();
        assert_read_only_failure_without_writes(&dir);
    }
}

#[test]
fn stale_manifests_are_rejected_without_writes() {
    for filename in [
        "merged_codes.manifest.json",
        "merged_residuals.manifest.json",
    ] {
        let dir = prepared_index();
        let manifest_path = dir.path().join(filename);
        let mut manifest: serde_json::Value =
            serde_json::from_reader(fs::File::open(&manifest_path).unwrap()).unwrap();
        manifest["metadata_mtime"] = json!(0.0);
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert_read_only_failure_without_writes(&dir);
    }
}

#[test]
fn invalid_merged_files_are_rejected_without_writes() {
    for filename in ["merged_codes.npy", "merged_residuals.npy"] {
        let dir = prepared_index();
        let path = dir.path().join(filename);
        let mut bytes = fs::read(&path).unwrap();
        bytes.pop();
        fs::write(path, bytes).unwrap();
        assert_read_only_failure_without_writes(&dir);
    }
}

#[test]
fn malformed_document_lengths_are_rejected_without_writes() {
    let dir = prepared_index_with_batch(8);
    let mut doclens: Vec<i64> =
        serde_json::from_reader(fs::File::open(dir.path().join("doclens.0.json")).unwrap())
            .unwrap();
    doclens[0] = -1;
    fs::write(
        dir.path().join("doclens.0.json"),
        serde_json::to_vec(&doclens).unwrap(),
    )
    .unwrap();

    let error = match MmapIndex::load_read_only(dir.path().to_str().unwrap()) {
        Ok(_) => panic!("read-only load unexpectedly accepted malformed doclens"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("negative"),
        "unexpected error: {error}"
    );
}

#[test]
fn mismatched_document_count_and_embedding_total_are_rejected_without_writes() {
    for (metadata_key, value, expected) in [
        ("num_documents", 13, "documents"),
        ("num_embeddings", 37, "embeddings"),
    ] {
        let dir = prepared_index_with_batch(8);
        let metadata_path = dir.path().join("metadata.json");
        let mut metadata: serde_json::Value =
            serde_json::from_reader(fs::File::open(&metadata_path).unwrap()).unwrap();
        metadata[metadata_key] = json!(value);
        fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

        let error = match MmapIndex::load_read_only(dir.path().to_str().unwrap()) {
            Ok(_) => panic!("read-only load unexpectedly accepted inconsistent metadata"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {metadata_key}: {error}"
        );
    }
}

#[test]
fn merged_row_counts_must_match_validated_doclens_without_writes() {
    let dir = prepared_index_with_batch(8);
    let manifest_path = dir.path().join("merged_codes.manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_reader(fs::File::open(&manifest_path).unwrap()).unwrap();
    manifest["total_rows"] = json!(manifest["total_rows"].as_u64().unwrap() + 1);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    assert_read_only_failure_without_writes(&dir);
}
