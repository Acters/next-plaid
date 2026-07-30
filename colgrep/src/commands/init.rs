use std::path::PathBuf;

use anyhow::Result;

use crate::commands::search::{resolve_model, resolve_pool_factor};
use colgrep::{ensure_model, find_parent_index, index_exists, Config, IndexBuilder};

pub struct InitOptions<'a> {
    pub cli_model: Option<&'a str>,
    pub no_pool: bool,
    pub pool_factor: Option<usize>,
    pub auto_confirm: bool,
    pub batch_size: Option<usize>,
    pub encode_batch_size: Option<usize>,
    pub index_chunk_size: Option<usize>,
    pub codec_gpu_memory_mb: Option<u64>,
    pub static_batch: bool,
}

const BYTES_PER_MIB: u64 = 1024 * 1024;

fn codec_gpu_memory_budget_bytes(codec_gpu_memory_mb: Option<u64>) -> Result<Option<usize>> {
    let Some(mib) = codec_gpu_memory_mb else {
        return Ok(None);
    };
    if mib == 0 {
        anyhow::bail!("--codec-gpu-memory-mb must be greater than zero MiB");
    }
    let bytes = mib.checked_mul(BYTES_PER_MIB).ok_or_else(|| {
        anyhow::anyhow!(
            "--codec-gpu-memory-mb={} is too large: MiB-to-byte conversion overflowed",
            mib
        )
    })?;
    let bytes = usize::try_from(bytes).map_err(|_| {
        anyhow::anyhow!(
            "--codec-gpu-memory-mb={} is too large: {} bytes cannot be represented on this platform",
            mib,
            bytes
        )
    })?;
    Ok(Some(bytes))
}

fn resolve_index_runtime_overrides(
    config: &Config,
    cli_batch_size: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    (
        config.configured_parallel_sessions(),
        cli_batch_size
            .map(|batch_size| batch_size.max(1))
            .or_else(|| config.configured_batch_size()),
    )
}

pub fn cmd_init(path: &PathBuf, options: InitOptions<'_>) -> Result<()> {
    let path = std::fs::canonicalize(path)
        .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))?;

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    let config = Config::load().unwrap_or_default();
    let model = resolve_model(&config, options.cli_model);
    let pool_factor = resolve_pool_factor(&config, options.pool_factor, options.no_pool);
    let codec_gpu_memory_budget_bytes = codec_gpu_memory_budget_bytes(options.codec_gpu_memory_mb)?;

    let quantized = !config.use_fp32();
    let (parallel_sessions, batch_size) =
        resolve_index_runtime_overrides(&config, options.batch_size);

    // Check if path is inside an already-indexed parent project
    let parent_info = find_parent_index(&path, &model)?;
    let effective_root = match &parent_info {
        Some(info) => info.project_path.clone(),
        None => path.clone(),
    };

    // Check if index already exists for the effective root
    let has_existing_index = index_exists(&effective_root, &model);

    // Ensure model is downloaded
    let model_path = ensure_model(Some(&model), has_existing_index)?;

    let mut builder = IndexBuilder::with_options(
        &effective_root,
        &model,
        &model_path,
        quantized,
        pool_factor,
        parallel_sessions,
        batch_size,
    )?;
    builder.set_auto_confirm(options.auto_confirm);
    builder.set_dynamic_batch(!options.static_batch);
    if let Some(codec_gpu_memory_budget_bytes) = codec_gpu_memory_budget_bytes {
        builder.set_codec_gpu_memory_budget_bytes(codec_gpu_memory_budget_bytes)?;
    }
    if let Some(encode_batch_size) = options.encode_batch_size {
        builder.set_encode_batch_size(encode_batch_size.max(1));
    }
    if let Some(index_chunk_size) = options.index_chunk_size {
        builder.set_index_chunk_size(index_chunk_size.max(1));
    }
    let stats = builder.index(None, false)?;

    let changes = stats.added + stats.changed + stats.deleted;
    if changes > 0 {
        if let Some(ref info) = parent_info {
            eprintln!(
                "Indexed {} (subdir: {}) (added: {}, changed: {}, deleted: {}, unchanged: {})",
                info.project_path.display(),
                info.relative_subdir.display(),
                stats.added,
                stats.changed,
                stats.deleted,
                stats.unchanged,
            );
        } else {
            eprintln!(
                "Indexed {} (added: {}, changed: {}, deleted: {}, unchanged: {})",
                effective_root.display(),
                stats.added,
                stats.changed,
                stats.deleted,
                stats.unchanged,
            );
        }
    } else {
        eprintln!(
            "Index is up to date for {} ({} files)",
            effective_root.display(),
            stats.unchanged
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_resolve_index_runtime_overrides_preserves_explicit_values() {
        let config = Config {
            parallel_sessions: Some(3),
            batch_size: Some(7),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(9));

        assert_eq!(parallel_sessions, Some(3));
        assert_eq!(batch_size, Some(9));
    }

    #[test]
    fn test_resolve_index_runtime_overrides_defers_auto_defaults() {
        let config = Config::default();

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, None);

        assert_eq!(parallel_sessions, None);
        assert_eq!(batch_size, None);
    }

    #[test]
    fn test_resolve_index_runtime_overrides_normalizes_values() {
        let config = Config {
            parallel_sessions: Some(0),
            batch_size: Some(0),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(0));

        assert_eq!(parallel_sessions, Some(1));
        assert_eq!(batch_size, Some(1));
    }

    #[test]
    fn test_codec_gpu_memory_budget_bytes_defaults_to_none() {
        assert_eq!(codec_gpu_memory_budget_bytes(None).unwrap(), None);
    }

    #[test]
    fn test_codec_gpu_memory_budget_bytes_converts_checked_mib() {
        assert_eq!(
            codec_gpu_memory_budget_bytes(Some(256)).unwrap(),
            Some(256 * 1024 * 1024)
        );
        let error = codec_gpu_memory_budget_bytes(Some(u64::MAX))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conversion overflowed"));
    }

    #[test]
    fn test_codec_gpu_memory_cli_max_u64_is_rejected_by_checked_conversion() {
        let cli = crate::cli::Cli::try_parse_from([
            "colgrep",
            "init",
            "--codec-gpu-memory-mb",
            &u64::MAX.to_string(),
        ])
        .unwrap();
        let Some(crate::cli::Commands::Init {
            codec_gpu_memory_mb,
            ..
        }) = cli.command
        else {
            panic!("expected init command");
        };
        assert_eq!(codec_gpu_memory_mb, Some(u64::MAX));
        assert!(codec_gpu_memory_budget_bytes(codec_gpu_memory_mb).is_err());
    }

    #[test]
    fn test_codec_gpu_memory_budget_bytes_rejects_zero() {
        let error = codec_gpu_memory_budget_bytes(Some(0))
            .unwrap_err()
            .to_string();
        assert!(error.contains("greater than zero"));
    }
}
