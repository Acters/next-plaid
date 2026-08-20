# Dreb runtime validation — upstream v1.7.0 integration

Date: 2026-08-20

## Source identity

- Upstream release: `v1.7.0` / `00e26aa`
- Dreb stale-source baseline: `588b3fc`
- Merge commit: `faeea13`
- Core benchmark harness: `1286943`
- Read-only sidecar tests: `18f8c70`
- Exact-root indexing: `4110d66`
- Post-review correctness hardening: `156c337`

The Dreb fork remains intentional. The integrated runtime retains the versioned `serve --stdio` protocol, `restrict_to_dir`, strict read-only index loading, source-freshness checks, model-cache validation, project/model locks, exact-root indexing, and per-run CUDA codec/pooling controls.

## Validation

The final source passed:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked --features cuda
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked --features cuda -- -D warnings
```

The read-only suite covers missing, stale, wrong-dtype, wrong-size, invalid-value, and checksum-mismatched inverse-norm sidecars. Read-only fallback performs a search and verifies ordered IDs/scores without writing. Normal loading repairs invalid optional sidecars. Persistent state checks fail closed on source-walk errors and no longer classify metadata I/O failures as generation changes.

The parent Dreb extension suite passed 90 tests with five environment-gated skips. Real-runtime gates verified:

- persistent versus one-shot ordered path/span/score parity;
- extension execution through the persistent backend;
- fresh exact-root automatic indexing;
- edit, addition, and deletion stale-source refresh;
- persistent reuse after refresh.

## Performance and retrieval parity

The reproducible A/B evidence is stored in the parent repository at:

```text
extensions/colgrep/benchmark/upstream-v1.7-ab-2026-08-20/
```

Pooled local results on the 5,702-document Dreb extensions index:

- Core semantic search: 27.07 ms → 4.71 ms (**5.75×**)
- Persistent semantic-only: 41.14 ms → 17.55 ms (**2.34×**)
- Persistent hybrid: 62.68 ms → 43.00 ms (**1.46×**)

On 20 Dreb-specific relevance queries, hit@15 and MRR@15 were unchanged in both semantic and hybrid modes. Every top-1 result was unchanged. v1.7's approximate residual path changes near-tie ordering and scores slightly; the maximum shared-result score delta was 0.00571.

See the evidence README for methodology, raw distributions, caveats, and file-level results.
