# dreb-colgrep-runtime validation — 2026-08-02

## Runtime identity

- Branch: `dreb-colgrep-runtime`
- Commit: `b2ce382`
- Base: upstream `1.6.5` (`76092e1`)
- Binary: `target/release/colgrep`, 68 MiB, SHA-256 `faed5a68e896f1586a46388e651aafc9089f8b0ee610a87c785f6e151ba69a45`
- Wrapper: [`dreb-extensions/extensions/colgrep/scripts/colgrep-cuda13-wrapper.sh`](https://github.com/Acters/dreb-extensions/blob/main/extensions/colgrep/scripts/colgrep-cuda13-wrapper.sh), SHA-256 `dbf9b25482381e08246cd909cc4a2c081650a1401fff989b22bbeb3a786cbc10`
- ONNX Runtime: `onnxruntime-opt-cuda 1.28.0-1`
- CUDA: `13.3.1-1`
- cuDNN: `9.24.0.43-1.1`

## Automated validation

Passed after the final runtime commit:

- `cargo fmt --all -- --check`
- `cargo test --workspace --locked`
  - colgrep library: 701 passed, 1 ignored
  - colgrep binary: 93 passed
  - next-plaid: 140 passed
  - read-only load: 8 passed
  - all integration suites passed
- `cargo clippy --workspace --all-targets --locked -- -D warnings`
- `cargo build --release --locked -p colgrep --features cuda`
- `git diff --check`

## Live runtime checks

Target project: a local dreb checkout

Index: `$XDG_DATA_HOME/colgrep/indices/<project-id>`

### Health/search/shutdown

The wrapper launched `serve --stdio` against the existing index. Health returned 37,284 documents and the expected project/model identity. A natural-language search returned the semantic-search implementation first. Shutdown completed cleanly with empty stderr.

### Persistent versus one-shot parity

Persistent stdio and one-shot CLI returned identical ordered `(file, line, end_line, score)` tuples for five-result searches in these modes:

- natural language
- exact identifier query
- semantic-only
- code-only
- include glob (`*.ts`)
- restricted directory (`src/dreb/packages/semantic-search`)

### Lock and stale-index behavior

- Server startup behind an externally held exclusive writer lock failed after the bounded startup wait with `Failed to acquire shared index lock while loading`.
- A request before a writer lock succeeded; the same request during the writer lock returned protocol error `index_busy`; after release, the same request succeeded again.
- Setting `state.json` `dirty: true` after server startup caused the next request to return protocol error `stale_index`; shutdown still completed cleanly. The state file was restored after the test.

### Query latency

Twenty post-warmup benchmark queries from the extension query set, run through the real server protocol:

- mean: 80.61ms
- median: 65.78ms
- p95: 93.43ms
- min: 46.04ms
- max: 359.29ms

The max sample is the initial model/session warm request in this process. Median is the steadier query estimate for this small sample; larger controlled comparisons should use randomized A/B measurement.

### SQLite side files

The live read-only server run changed only `metadata.db-shm` mtime. The main database, vector files, manifests, metadata, and index state were unchanged. This SQLite shared-memory file is expected filesystem state for a live read connection, but persistent-search no-mutation tests should compare both before/after-close state and live-server expectations rather than asserting that SQLite never touches its shared-memory file.
