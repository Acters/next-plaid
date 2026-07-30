//! End-to-end tests for asymmetric binary quantization: build a binary index,
//! confirm the on-disk document store shrinks, and confirm the asymmetric binary
//! MaxSim scoring path still retrieves the right documents.

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use next_plaid::index::MmapIndex;
use next_plaid::{binary, IndexConfig, SearchParameters};
use rand::rngs::StdRng;
use tempfile::TempDir;

/// Distinct L2-normalized documents so self-retrieval is well defined.
fn random_docs(num_docs: usize, tokens: usize, dim: usize) -> Vec<Array2<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..num_docs)
        .map(|_| {
            let mut emb: Array2<f32> =
                Array2::random_using((tokens, dim), StandardNormal, &mut rng);
            for mut row in emb.axis_iter_mut(Axis(0)) {
                let norm = row.dot(&row).sqrt().max(1e-12);
                row /= norm;
            }
            emb
        })
        .collect()
}

fn config(binary: bool) -> IndexConfig {
    IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        binary,
        ..Default::default()
    }
}

fn params() -> SearchParameters {
    SearchParameters {
        top_k: 3,
        n_ivf_probe: 16,
        ..Default::default()
    }
}

#[test]
fn binary_index_stores_one_bit_per_dimension() {
    let (dim, nbits) = (64usize, 4usize);
    let docs = random_docs(40, 8, dim);

    let float_dir = TempDir::new().unwrap();
    let bin_dir = TempDir::new().unwrap();
    let float_ix =
        MmapIndex::create_with_kmeans(&docs, float_dir.path().to_str().unwrap(), &config(false))
            .unwrap();
    let bin_ix =
        MmapIndex::create_with_kmeans(&docs, bin_dir.path().to_str().unwrap(), &config(true))
            .unwrap();

    // Float stores dim*nbits/8 bytes/token; binary stores ceil(dim/8).
    assert!(!float_ix.metadata.binary);
    assert!(bin_ix.metadata.binary);
    assert_eq!(float_ix.mmap_residuals.ncols(), dim * nbits / 8); // 32
    assert_eq!(bin_ix.mmap_residuals.ncols(), binary::packed_dim(dim)); // 8

    // The document store is 4x narrower here (and 32x versus raw f32).
    assert_eq!(
        float_ix.mmap_residuals.ncols() / bin_ix.mmap_residuals.ncols(),
        nbits
    );
}

#[test]
fn binary_index_retrieves_the_query_document() {
    let docs = random_docs(50, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config(true)).unwrap();

    // Use each document's own tokens as the query; the document itself should
    // rank first under asymmetric binary MaxSim (q is dotted with sign(q)).
    let mut hits = 0;
    for (doc_id, doc) in docs.iter().enumerate() {
        let result = index.search(doc, &params(), None).unwrap();
        if result.passage_ids.first() == Some(&(doc_id as i64)) {
            hits += 1;
        }
    }
    let recall_at_1 = hits as f32 / docs.len() as f32;
    assert!(recall_at_1 >= 0.9, "binary recall@1 too low: {recall_at_1}");
}

#[test]
fn binary_reconstruct_returns_signs_not_garbage() {
    let docs = random_docs(20, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config(true)).unwrap();

    // reconstruct() on a binary index must decode the stored ±1 signs, matching
    // sign(original), not misread them as residual codes.
    let recon = index.reconstruct(&[3]).unwrap();
    let doc = &recon[0];
    assert_eq!(doc.dim(), docs[3].dim());
    for (got, orig) in doc.iter().zip(docs[3].iter()) {
        assert_eq!(*got, if *orig >= 0.0 { 1.0 } else { -1.0 });
    }
}

#[test]
fn binary_index_updates_via_start_from_scratch_rebuild() {
    // While raw embeddings are retained (num_documents <= start_from_scratch),
    // update() serves binary indexes through the full rebuild-from-raw path:
    // documents can be added and the index must stay binary.
    let dim = 64usize;
    // One sequential pool so the update docs are distinct from the initial
    // docs (random_docs is deterministically seeded — two calls would produce
    // identical embeddings and tie self-retrieval between old and new IDs).
    let all = random_docs(24, 8, dim);
    let (docs, more) = all.split_at(20);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    MmapIndex::create_with_kmeans(docs, path, &config(true)).unwrap();

    let mut index = MmapIndex::load(path).unwrap();
    let doc_ids = index
        .update(more, &next_plaid::update::UpdateConfig::default())
        .unwrap();
    assert_eq!(doc_ids, vec![20, 21, 22, 23]);

    let reloaded = MmapIndex::load(path).unwrap();
    assert!(
        reloaded.metadata.binary,
        "rebuild-from-raw flipped the index back to residual"
    );
    assert_eq!(reloaded.num_documents(), 24);
    assert_eq!(reloaded.mmap_residuals.ncols(), binary::packed_dim(dim));

    // The added documents must be retrievable under binary MaxSim.
    let result = reloaded.search(&more[0], &params(), None).unwrap();
    assert_eq!(result.passage_ids.first(), Some(&20));
}

#[test]
fn binary_index_rejects_update_above_start_from_scratch() {
    // Past the raw-embeddings threshold the only update paths left are the
    // buffer/centroid-expansion appends, which re-encode through the residual
    // codec. They must be refused before mutating anything on disk.
    let dim = 64usize;
    let docs = random_docs(20, 8, dim);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let create_config = next_plaid::IndexConfig {
        start_from_scratch: 5,
        ..config(true)
    };
    MmapIndex::create_with_kmeans(&docs, path, &create_config).unwrap();

    let mut index = MmapIndex::load(path).unwrap();
    let more = random_docs(4, 8, dim);
    let update_config = next_plaid::update::UpdateConfig {
        start_from_scratch: 5,
        ..Default::default()
    };
    let err = index.update(&more, &update_config).unwrap_err();
    assert!(
        err.to_string().contains("binary"),
        "expected binary-index rejection, got: {err}"
    );

    let reloaded = MmapIndex::load(path).unwrap();
    assert!(reloaded.metadata.binary);
    assert_eq!(
        reloaded.num_documents(),
        20,
        "failed update mutated the index"
    );
}

#[test]
fn binary_index_rejects_incremental_update() {
    // update_index re-encodes through the residual codec; appending residual
    // rows to a 1-bit sign store would corrupt it and flip metadata.binary
    // off. Every update entry point must refuse — update_append reached
    // update_index unguarded once.
    let dim = 64usize;
    let docs = random_docs(20, 8, dim);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    MmapIndex::create_with_kmeans(&docs, path, &config(true)).unwrap();

    let more = random_docs(4, 8, dim);
    let err = MmapIndex::update_append(&more, path, &next_plaid::update::UpdateConfig::default())
        .unwrap_err();
    assert!(
        err.to_string().contains("binary"),
        "expected binary-index rejection, got: {err}"
    );

    // The failed attempt must not have flipped the on-disk flag.
    let reloaded = MmapIndex::load(path).unwrap();
    assert!(reloaded.metadata.binary);
}

#[test]
fn encode_index_chunk_binary_packs_signs_with_ivf_codes() {
    // encode_index_chunk with `binary: true` must produce the same artifact as
    // create_index_files' binary path: rows of packed sign bits (not residual
    // codes) plus one centroid code per token for IVF. Guards the fast path
    // that skips residual computation entirely in binary mode.
    let dim = 64usize;
    let docs = random_docs(10, 8, dim);
    let centroids = next_plaid::compute_kmeans(
        &docs,
        &next_plaid::kmeans::ComputeKmeansConfig {
            seed: 42,
            num_partitions: Some(16),
            force_cpu: true,
            ..Default::default()
        },
    )
    .unwrap();
    let artifacts = next_plaid::prepare_codec_artifacts(&docs, centroids, &config(true)).unwrap();

    let chunk = next_plaid::encode_index_chunk(&docs, &artifacts.codec, true, true).unwrap();

    let total_tokens: usize = docs.iter().map(|d| d.nrows()).sum();
    assert_eq!(chunk.codes.len(), total_tokens);
    assert_eq!(chunk.residuals.ncols(), binary::packed_dim(dim));
    assert_eq!(chunk.residuals.nrows(), total_tokens);
    assert_eq!(
        chunk.doclens,
        docs.iter().map(|d| d.nrows() as i64).collect::<Vec<_>>()
    );

    // The packed rows must be the embeddings' sign bits...
    let mut flat = Array2::<f32>::zeros((total_tokens, dim));
    let mut offset = 0;
    for doc in &docs {
        flat.slice_mut(ndarray::s![offset..offset + doc.nrows(), ..])
            .assign(doc);
        offset += doc.nrows();
    }
    assert_eq!(chunk.residuals, binary::binarize(&flat.view()));

    // ...and the codes the nearest-centroid assignment the IVF is built from.
    let want_codes: Vec<i64> = artifacts
        .codec
        .compress_into_codes_cpu(&flat)
        .iter()
        .map(|&c| c as i64)
        .collect();
    assert_eq!(chunk.codes.to_vec(), want_codes);
}

#[cfg(feature = "_cuda")]
mod cuda_codec_budget_tests {
    use super::*;

    fn bounded_budget(dim: usize, num_centroids: usize) -> usize {
        let fixed = num_centroids * dim * std::mem::size_of::<f32>();
        let row = dim * std::mem::size_of::<f32>()
            + num_centroids * std::mem::size_of::<f32>()
            + std::mem::size_of::<u32>()
            + dim * std::mem::size_of::<f32>();
        fixed + row * 3
    }

    fn assert_cuda_available() {
        next_plaid::cuda::get_global_context().expect("requires an available CUDA device");
    }

    /// Every file in an index directory, name-sorted, for byte-exact comparison
    /// of two index artifacts or of one directory before and after an operation.
    fn snapshot_index_dir(dir: &TempDir) -> Vec<(String, Vec<u8>)> {
        let mut entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    std::fs::read(entry.path()).unwrap(),
                )
            })
            .collect();
        entries.sort();
        entries
    }

    /// Bounded multi-batch append to a small residual index: the returned IDs
    /// and document counts must be correct, and every persisted artifact must
    /// byte-match an unbounded control append of the same documents.
    #[test]
    #[ignore = "requires an available CUDA device"]
    fn update_append_with_gpu_memory_budget_matches_unbounded_multi_batch() {
        use next_plaid::Metadata;

        assert_cuda_available();
        let all = random_docs(16, 8, 64);
        let (docs, more) = all.split_at(10);

        let bounded_dir = TempDir::new().unwrap();
        let control_dir = TempDir::new().unwrap();
        let bounded_path = bounded_dir.path().to_str().unwrap();
        let control_path = control_dir.path().to_str().unwrap();
        // CPU k-means keeps codec creation deterministic and identical across
        // both dirs; the appended compression is the GPU path under test.
        let create_config = IndexConfig {
            force_cpu: true,
            ..config(false)
        };
        MmapIndex::create_with_kmeans(docs, bounded_path, &create_config).unwrap();
        MmapIndex::create_with_kmeans(docs, control_path, &create_config).unwrap();

        // Budget for ~3 compression rows per batch: 48 appended tokens force
        // well over a dozen CUDA batches (the unbounded control uses one).
        let metadata = Metadata::load_from_path(bounded_dir.path()).unwrap();
        let budget = bounded_budget(metadata.embedding_dim, metadata.num_partitions);

        let update_config = next_plaid::update::UpdateConfig::default();
        let bounded_ids = MmapIndex::update_append_with_gpu_memory_budget(
            more,
            bounded_path,
            &update_config,
            Some(budget),
        )
        .unwrap();
        let control_ids = MmapIndex::update_append(more, control_path, &update_config).unwrap();

        let want_ids: Vec<i64> = (10..16).collect();
        assert_eq!(bounded_ids, want_ids);
        assert_eq!(control_ids, want_ids);

        assert_eq!(
            snapshot_index_dir(&bounded_dir),
            snapshot_index_dir(&control_dir),
            "bounded append artifacts diverged from the unbounded control"
        );

        let reloaded = Metadata::load_from_path(bounded_dir.path()).unwrap();
        assert_eq!(reloaded.num_documents, 16);
        assert_eq!(reloaded.num_embeddings, 16 * 8);
        assert_eq!(reloaded.avg_doclen, 8.0);
    }

    /// An undersized budget must propagate as `Error::Config` and leave the
    /// index metadata and artifacts byte-identical.
    #[test]
    #[ignore = "requires an available CUDA device"]
    fn update_append_with_gpu_memory_budget_config_error_leaves_index_unchanged() {
        assert_cuda_available();
        let all = random_docs(14, 8, 64);
        let (docs, more) = all.split_at(10);

        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap();
        let create_config = IndexConfig {
            force_cpu: true,
            ..config(false)
        };
        MmapIndex::create_with_kmeans(docs, path, &create_config).unwrap();
        let before = snapshot_index_dir(&dir);

        // 1 byte is smaller than the resident centroid matrix: validation must
        // fail before any chunk write.
        let error = MmapIndex::update_append_with_gpu_memory_budget(
            more,
            path,
            &next_plaid::update::UpdateConfig::default(),
            Some(1),
        )
        .unwrap_err();
        assert!(
            matches!(error, next_plaid::Error::Config(_)),
            "expected the undersized budget to propagate as Error::Config, got: {error}"
        );

        assert_eq!(
            snapshot_index_dir(&dir),
            before,
            "failed append mutated index metadata or artifacts"
        );
    }

    /// Successful initial-create public path under a bounded multi-batch CUDA
    /// budget: preflight → codec artifacts → encode chunk → write index files,
    /// then validate the resulting metadata and artifacts against an unbounded
    /// control run and through `MmapIndex::load`.
    #[test]
    #[ignore = "requires an available CUDA device"]
    fn initial_index_public_path_with_gpu_memory_budget_matches_unbounded() {
        use next_plaid::Metadata;

        assert_cuda_available();
        let docs = random_docs(12, 8, 64);
        let total_tokens: usize = docs.iter().map(|d| d.nrows()).sum();

        // Deterministic CPU k-means so the bounded and unbounded runs share
        // the exact same codec inputs.
        let centroids = next_plaid::compute_kmeans(
            &docs,
            &next_plaid::kmeans::ComputeKmeansConfig {
                seed: 42,
                num_partitions: Some(16),
                force_cpu: true,
                ..Default::default()
            },
        )
        .unwrap();
        let cfg = config(false);

        // Preflight validates the heuristic codec shape (which can exceed the
        // explicit 16 centroids), so size the budget for the larger of the two
        // while still forcing ~3 compression rows per CUDA batch.
        let preflight_centroids =
            next_plaid::kmeans::estimate_num_partitions(&docs).min(total_tokens);
        let budget = bounded_budget(64, preflight_centroids.max(16));

        let run_public_path = |budget: Option<usize>, dir: &TempDir| -> Metadata {
            let path = dir.path().to_str().unwrap();
            next_plaid::preflight_codec_gpu_memory_budget(&docs, false, false, budget).unwrap();
            let artifacts = next_plaid::prepare_codec_artifacts_with_gpu_memory_budget(
                &docs,
                centroids.clone(),
                &cfg,
                budget,
            )
            .unwrap();
            let chunk = next_plaid::encode_index_chunk_with_gpu_memory_budget(
                &docs,
                &artifacts.codec,
                false,
                false,
                budget,
            )
            .unwrap();
            next_plaid::write_index_from_encoded_chunks(&[chunk], &artifacts, path, &cfg).unwrap()
        };

        let bounded_dir = TempDir::new().unwrap();
        let control_dir = TempDir::new().unwrap();
        let bounded_metadata = run_public_path(Some(budget), &bounded_dir);
        run_public_path(None, &control_dir);

        assert_eq!(bounded_metadata.num_chunks, 1);
        assert_eq!(bounded_metadata.num_documents, 12);
        assert_eq!(bounded_metadata.num_embeddings, total_tokens);
        assert_eq!(bounded_metadata.avg_doclen, 8.0);
        assert!(!bounded_metadata.binary);

        assert_eq!(
            snapshot_index_dir(&bounded_dir),
            snapshot_index_dir(&control_dir),
            "bounded initial create artifacts diverged from the unbounded control"
        );

        let index = MmapIndex::load(bounded_dir.path().to_str().unwrap()).unwrap();
        assert_eq!(index.num_documents(), 12);
    }

    #[test]
    #[ignore = "requires an available CUDA device"]
    fn encode_index_chunk_with_gpu_memory_budget_binary_matches_legacy_multi_batch() {
        assert_cuda_available();
        let docs = random_docs(10, 8, 64);
        let centroids = next_plaid::compute_kmeans(
            &docs,
            &next_plaid::kmeans::ComputeKmeansConfig {
                seed: 42,
                num_partitions: Some(16),
                force_cpu: true,
                ..Default::default()
            },
        )
        .unwrap();
        let artifacts =
            next_plaid::prepare_codec_artifacts(&docs, centroids, &config(true)).unwrap();
        let budget = bounded_budget(64, artifacts.codec.num_centroids());

        let legacy = next_plaid::encode_index_chunk(&docs, &artifacts.codec, false, true).unwrap();
        let bounded = next_plaid::encode_index_chunk_with_gpu_memory_budget(
            &docs,
            &artifacts.codec,
            false,
            true,
            Some(budget),
        )
        .unwrap();

        assert_eq!(bounded.codes, legacy.codes);
        assert_eq!(bounded.residuals, legacy.residuals);
        assert_eq!(bounded.doclens, legacy.doclens);
    }

    #[test]
    #[ignore = "requires an available CUDA device"]
    fn encode_index_chunk_with_gpu_memory_budget_residual_matches_legacy_multi_batch() {
        assert_cuda_available();
        let docs = random_docs(10, 8, 64);
        let centroids = next_plaid::compute_kmeans(
            &docs,
            &next_plaid::kmeans::ComputeKmeansConfig {
                seed: 42,
                num_partitions: Some(16),
                force_cpu: true,
                ..Default::default()
            },
        )
        .unwrap();
        let artifacts =
            next_plaid::prepare_codec_artifacts(&docs, centroids, &config(false)).unwrap();
        let budget = bounded_budget(64, artifacts.codec.num_centroids());

        let legacy = next_plaid::encode_index_chunk(&docs, &artifacts.codec, false, false).unwrap();
        let bounded = next_plaid::encode_index_chunk_with_gpu_memory_budget(
            &docs,
            &artifacts.codec,
            false,
            false,
            Some(budget),
        )
        .unwrap();

        assert_eq!(bounded.codes, legacy.codes);
        assert_eq!(bounded.residuals, legacy.residuals);
        assert_eq!(bounded.doclens, legacy.doclens);
    }
}
