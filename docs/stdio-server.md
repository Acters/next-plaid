# Persistent stdio search protocol

`colgrep serve --stdio` keeps one existing ColGREP model and index loaded and serves sequential search requests over newline-delimited JSON. It is intended for local tools that issue repeated queries and want to avoid rebuilding the ONNX session for every CLI invocation.

The server does not listen on a network socket, index files, repair indexes, or emit non-protocol data on stdout. Diagnostics use stderr.

## Start the server

Build the project index first, then start the server for the same canonical project/model pair:

```bash
colgrep init --model lightonai/LateOn-Code-edge /path/to/project
colgrep serve --stdio --model lightonai/LateOn-Code-edge /path/to/project
```

Startup fails if the path or matching index is missing, the configuration cannot be read, or the index is dirty, incomplete, or being built. The model, tokenizer, ONNX session, SQLite filtering data, and mmap vector index are loaded once.

## Framing and lifecycle

Each stdin line is one UTF-8 JSON request. Each stdout line is one JSON response. The current protocol version is `1`.

- Request IDs must be JSON strings or numbers and are echoed unchanged.
- Requests are processed sequentially.
- Request lines are limited to 1 MiB.
- Search result payloads are limited to 16 MiB.
- `top_k` is limited to 1,000.
- EOF or a successful `shutdown` request exits cleanly.
- Invalid requests receive an error response; they do not terminate the server.

The server acquires a shared index lock for each search. An index writer cannot replace mmap or SQLite files during a request. If a writer is already active, the response uses `index_busy`. If the persisted index changed between requests, the response uses `stale_index`; restart the server to load the new generation. Before searching, the server also runs the incremental planner's project-tree scan with its mtime/size fast path. Added, edited, or deleted indexable files produce `stale_source`; stop the server, run `colgrep init`, and start a fresh server rather than returning results from the stale source snapshot.

## Requests

### Health

```json
{"version":1,"id":"ready","op":"health"}
```

Example response:

```json
{"version":1,"id":"ready","ok":true,"op":"health","status":"ok","documents":37284,"project_root":"/path/to/project","model":"lightonai/LateOn-Code-edge"}
```

### Search

```json
{"version":1,"id":"q-1","op":"search","query":"database retry logic","top_k":10,"semantic_only":false,"code_only":true,"include":["*.{rs,ts}"],"exclude":["*test*"],"restrict_to_dir":"src"}
```

Search fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `query` | string | yes | Semantic query. Empty strings are rejected. |
| `top_k` | integer | no | Result count; configured CLI default when omitted. |
| `semantic_only` | boolean | no | Disable FTS5 hybrid ranking. |
| `code_only` | boolean | no | Exclude document/config formats. |
| `include` | string array | no | Include file globs. `include_patterns` is an alias. |
| `exclude` | string array | no | Exclude file globs. `exclude_patterns` is an alias. |
| `restrict_to_dir` | string | no | Relative, normalized descendant directory under the server root. Filters the already-loaded parent index without resolving or accessing the requested path. |

`restrict_to_dir` rejects empty or current-directory-only values, absolute paths, root/prefix components, and any `..` parent traversal. Harmless `.` components and repeated separators are normalized before filtering; the request never canonicalizes or accesses this path on the filesystem.

A successful response has the same ordered `SearchResult` values as `colgrep search --json --no-update` for the corresponding options. For example, a server rooted at `/path/to/project` with `restrict_to_dir: "src"` uses the same loaded-index subdirectory filter as a one-shot search run from `/path/to/project/src`.

```json
{"version":1,"id":"q-1","ok":true,"op":"search","results":[{"unit":{"file":"/path/to/project/src/retry.rs"},"score":1.23}]}
```

The abbreviated result above omits normal `CodeUnit` fields for readability.

### Shutdown

```json
{"version":1,"id":"stop","op":"shutdown"}
```

The response is flushed before the process exits:

```json
{"version":1,"id":"stop","ok":true,"op":"shutdown","status":"shutting_down"}
```

## Errors

Errors preserve the request ID when it was valid:

```json
{"version":1,"id":"q-1","ok":false,"op":"search","error":{"code":"stale_index","message":"stale-index: the index changed after the server loaded it; restart the server"}}
```

Important codes include:

- `invalid_json`, `unsupported_version`, `missing_id`, `invalid_id`
- `missing_operation`, `unsupported_operation`, `missing_query`, `empty_query`
- `top_k_too_large`, `invalid_glob`, `glob_too_large`, `invalid_restriction`
- `index_busy`, `stale_index`, `stale_source`, `search_failed`
- `response_too_large`, `serialization_error`

Clients should restart the process on `stale_index`, broken pipes, malformed responses, or unexpected process exit. On `stale_source`, restarting alone is insufficient: terminate the process, incrementally update with `colgrep init`, and then restart. For `index_busy`, retry after the writer completes or restart after the indexing operation.

## Index updates

The safest client lifecycle is:

1. Shut down the persistent server.
2. Run `colgrep init` as a one-shot process.
3. Start a new server lazily on the next search.

This releases mmap state before publication and ensures the next query loads the new index generation.
