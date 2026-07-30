use std::io::{self, BufRead, Write};
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;
use serde_json::{Map, Value};

use colgrep::{prepare_glob_patterns, SearchResult, MAX_GLOB_PATTERNS};

use super::search::SearchEngine;

pub const PROTOCOL_VERSION: u64 = 1;
const MAX_REQUEST_LINE_BYTES: usize = 1024 * 1024;
const MAX_RESULT_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOP_K: usize = 1000;

/// The small interface used by the protocol loop. Keeping the loop generic
/// makes request/response behavior testable without loading ONNX models.
pub(crate) trait SearchService {
    fn search(
        &self,
        query: &str,
        top_k: usize,
        semantic_only: bool,
        code_only: bool,
        include_patterns: &[String],
        exclude_patterns: &[String],
    ) -> Result<Vec<SearchResult>>;

    fn health(&self) -> HealthInfo;

    fn default_top_k(&self) -> usize;
}

#[derive(Debug, Clone)]
pub(crate) struct HealthInfo {
    pub documents: usize,
    pub project_root: String,
    pub model: String,
}

impl SearchService for SearchEngine {
    fn search(
        &self,
        query: &str,
        top_k: usize,
        semantic_only: bool,
        code_only: bool,
        include_patterns: &[String],
        exclude_patterns: &[String],
    ) -> Result<Vec<SearchResult>> {
        self.search(
            query,
            top_k,
            semantic_only,
            code_only,
            include_patterns,
            exclude_patterns,
        )
    }

    fn health(&self) -> HealthInfo {
        HealthInfo {
            documents: self.document_count(),
            project_root: self.project_root().display().to_string(),
            model: self.model().to_string(),
        }
    }

    fn default_top_k(&self) -> usize {
        self.configured_top_k().min(MAX_TOP_K)
    }
}

/// Start the explicitly selected stdio transport.
pub(crate) fn cmd_serve(stdio: bool, path: &Path, model: Option<&str>) -> Result<()> {
    if !stdio {
        anyhow::bail!("`colgrep serve` currently requires --stdio");
    }

    let engine = SearchEngine::load_existing(path, model)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    run_stdio(&mut reader, &mut writer, &engine)
}

/// Run the versioned newline-delimited JSON protocol until shutdown or EOF.
pub(crate) fn run_stdio<R: BufRead, W: Write, S: SearchService>(
    reader: &mut R,
    writer: &mut W,
    service: &S,
) -> Result<()> {
    loop {
        let line = match read_bounded_line(reader)? {
            None => return Ok(()),
            Some(BoundedLine::TooLong) => {
                write_response(
                    writer,
                    error_response(
                        Value::Null,
                        None,
                        "request_too_large",
                        format!(
                            "request line exceeds the {} byte limit",
                            MAX_REQUEST_LINE_BYTES
                        ),
                    ),
                )?;
                continue;
            }
            Some(BoundedLine::Line(line)) => line,
        };

        let request = match serde_json::from_slice::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    writer,
                    error_response(Value::Null, None, "invalid_json", error.to_string()),
                )?;
                continue;
            }
        };

        let id = request
            .id
            .as_ref()
            .filter(|id| is_valid_request_id(id))
            .cloned()
            .unwrap_or(Value::Null);
        let response = match validate_request(request, service.default_top_k().min(MAX_TOP_K)) {
            Ok(request) => handle_request(service, request),
            Err(error) => error_response(id, error.op, error.code, error.message),
        };
        let should_shutdown = response.get("op").and_then(Value::as_str) == Some("shutdown")
            && response.get("ok") == Some(&Value::Bool(true));

        write_response(writer, response)?;
        if should_shutdown {
            return Ok(());
        }
    }
}

#[derive(Debug, Deserialize)]
struct Request {
    version: Option<u64>,
    id: Option<Value>,
    op: Option<String>,
    query: Option<String>,
    top_k: Option<usize>,
    semantic_only: Option<bool>,
    code_only: Option<bool>,
    #[serde(default, alias = "include_patterns")]
    include: Vec<String>,
    #[serde(default, alias = "exclude_patterns")]
    exclude: Vec<String>,
}

struct ProtocolError {
    op: Option<String>,
    code: &'static str,
    message: String,
}

fn is_valid_request_id(id: &Value) -> bool {
    id.is_string() || id.is_number()
}

fn validate_globs(
    patterns: &[String],
    field: &'static str,
) -> std::result::Result<(), ProtocolError> {
    if patterns.len() > MAX_GLOB_PATTERNS {
        return Err(ProtocolError {
            op: Some("search".to_string()),
            code: "glob_too_large",
            message: format!(
                "{field} glob pattern count {} exceeds the limit of {MAX_GLOB_PATTERNS}",
                patterns.len()
            ),
        });
    }

    match prepare_glob_patterns(patterns) {
        Ok(_) => Ok(()),
        Err(error) => {
            let message = error.to_string();
            let code = if message.contains("expansion")
                || message.contains("nesting")
                || message.contains("length")
                || message.contains("count")
            {
                "glob_too_large"
            } else {
                "invalid_glob"
            };
            Err(ProtocolError {
                op: Some("search".to_string()),
                code,
                message: format!("invalid {field} glob set: {message}"),
            })
        }
    }
}

fn validate_request(
    request: Request,
    default_top_k: usize,
) -> std::result::Result<Request, ProtocolError> {
    if request.id.is_none() {
        return Err(ProtocolError {
            op: request.op.clone(),
            code: "missing_id",
            message: "request must contain an id".to_string(),
        });
    }
    let Some(request_id) = request.id.as_ref() else {
        unreachable!("missing request id was handled above");
    };
    if !is_valid_request_id(request_id) {
        return Err(ProtocolError {
            op: request.op.clone(),
            code: "invalid_id",
            message: "request id must be a string or number".to_string(),
        });
    }
    if request.version != Some(PROTOCOL_VERSION) {
        return Err(ProtocolError {
            op: request.op.clone(),
            code: "unsupported_version",
            message: format!("expected protocol version {PROTOCOL_VERSION}"),
        });
    }

    let Some(op) = request.op.as_deref() else {
        return Err(ProtocolError {
            op: None,
            code: "missing_operation",
            message: "request must contain an op".to_string(),
        });
    };

    match op {
        "search" => {
            let Some(query) = request.query.as_deref() else {
                return Err(ProtocolError {
                    op: Some(op.to_string()),
                    code: "missing_query",
                    message: "search requests must contain a query".to_string(),
                });
            };
            if query.is_empty() {
                return Err(ProtocolError {
                    op: Some(op.to_string()),
                    code: "empty_query",
                    message: "search query must not be empty".to_string(),
                });
            }
            if request.top_k.unwrap_or(default_top_k) > MAX_TOP_K {
                return Err(ProtocolError {
                    op: Some(op.to_string()),
                    code: "top_k_too_large",
                    message: format!("top_k must be at most {MAX_TOP_K}"),
                });
            }
            validate_globs(&request.include, "include")?;
            validate_globs(&request.exclude, "exclude")?;
        }
        "health" | "shutdown" => {}
        _ => {
            return Err(ProtocolError {
                op: Some(op.to_string()),
                code: "unsupported_operation",
                message: format!("unsupported operation: {op}"),
            });
        }
    }

    Ok(request)
}

enum SerializedResultsError {
    TooLarge { bytes: usize },
    Serialization(String),
}

fn serialize_results(
    results: Vec<SearchResult>,
) -> std::result::Result<Value, SerializedResultsError> {
    if results.iter().any(|result| !result.score.is_finite()) {
        return Err(SerializedResultsError::Serialization(
            "search result contains a non-finite score".to_string(),
        ));
    }

    // Serialize to bytes first. Only after the byte cap passes do we parse the
    // payload into a protocol Value, preventing an oversized result from
    // constructing an unbounded JSON value tree.
    let bytes = serde_json::to_vec(&results)
        .map_err(|error| SerializedResultsError::Serialization(error.to_string()))?;
    if bytes.len() > MAX_RESULT_PAYLOAD_BYTES {
        return Err(SerializedResultsError::TooLarge { bytes: bytes.len() });
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| SerializedResultsError::Serialization(error.to_string()))
}

fn handle_request<S: SearchService>(service: &S, request: Request) -> Value {
    let id = request.id.unwrap_or(Value::Null);
    let op = request.op.unwrap_or_default();

    match op.as_str() {
        "search" => match service.search(
            request.query.as_deref().unwrap_or_default(),
            request
                .top_k
                .unwrap_or_else(|| service.default_top_k().min(MAX_TOP_K)),
            request.semantic_only.unwrap_or(false),
            request.code_only.unwrap_or(false),
            &request.include,
            &request.exclude,
        ) {
            Ok(results) => match serialize_results(results) {
                Ok(results) => {
                    let mut response = object_with_id(id, true, Some("search"));
                    response.insert("results".to_string(), results);
                    Value::Object(response)
                }
                Err(SerializedResultsError::TooLarge { bytes }) => error_response(
                    id,
                    Some("search".to_string()),
                    "response_too_large",
                    format!(
                        "serialized search results are {bytes} bytes; limit is {MAX_RESULT_PAYLOAD_BYTES} bytes"
                    ),
                ),
                Err(SerializedResultsError::Serialization(error)) => error_response(
                    id,
                    Some("search".to_string()),
                    "serialization_error",
                    format!("failed to serialize search results: {error}"),
                ),
            },
            Err(error) => {
                let message = error.to_string();
                let code = if message.starts_with("index-busy:") {
                    "index_busy"
                } else if message.starts_with("stale-index:") {
                    "stale_index"
                } else {
                    "search_failed"
                };
                error_response(id, Some("search".to_string()), code, message)
            }
        },
        "health" => {
            let health = service.health();
            let mut response = object_with_id(id, true, Some("health"));
            response.insert("status".to_string(), Value::String("ok".to_string()));
            response.insert(
                "documents".to_string(),
                Value::Number(health.documents.into()),
            );
            response.insert(
                "project_root".to_string(),
                Value::String(health.project_root),
            );
            response.insert("model".to_string(), Value::String(health.model));
            Value::Object(response)
        }
        "shutdown" => {
            let mut response = object_with_id(id, true, Some("shutdown"));
            response.insert(
                "status".to_string(),
                Value::String("shutting_down".to_string()),
            );
            Value::Object(response)
        }
        _ => unreachable!("request was validated before dispatch"),
    }
}

fn object_with_id(id: Value, ok: bool, op: Option<&str>) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert(
        "version".to_string(),
        Value::Number(PROTOCOL_VERSION.into()),
    );
    object.insert("id".to_string(), id);
    object.insert("ok".to_string(), Value::Bool(ok));
    if let Some(op) = op {
        object.insert("op".to_string(), Value::String(op.to_string()));
    }
    object
}

fn error_response(id: Value, op: Option<String>, code: &str, message: String) -> Value {
    let mut response = object_with_id(id, false, op.as_deref());
    response.insert(
        "error".to_string(),
        serde_json::json!({"code": code, "message": message}),
    );
    Value::Object(response)
}

fn write_response<W: Write>(writer: &mut W, response: Value) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, &response)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

enum BoundedLine {
    Line(Vec<u8>),
    TooLong,
}

/// Read one line while retaining at most MAX_REQUEST_LINE_BYTES in memory.
/// Oversized lines are drained through their newline so the next request can
/// still be handled safely.
fn read_bounded_line<R: BufRead>(reader: &mut R) -> io::Result<Option<BoundedLine>> {
    let mut line = Vec::new();
    let mut too_long = false;

    loop {
        let (take, has_newline, eof) = {
            let buffer = reader.fill_buf()?;
            if buffer.is_empty() {
                (0, false, true)
            } else {
                let (take, has_newline) = match buffer.iter().position(|byte| *byte == b'\n') {
                    Some(position) => (position + 1, true),
                    None => (buffer.len(), false),
                };
                if !too_long {
                    let remaining = MAX_REQUEST_LINE_BYTES.saturating_sub(line.len());
                    let append_len = take.min(remaining);
                    line.extend_from_slice(&buffer[..append_len]);
                    if append_len < take {
                        too_long = true;
                    }
                }
                (take, has_newline, false)
            }
        };

        reader.consume(take);
        if eof {
            if line.is_empty() && !too_long {
                return Ok(None);
            }
            break;
        }
        if has_newline {
            break;
        }
    }

    if too_long {
        return Ok(Some(BoundedLine::TooLong));
    }

    while line
        .last()
        .is_some_and(|byte| *byte == b'\n' || *byte == b'\r')
    {
        line.pop();
    }
    Ok(Some(BoundedLine::Line(line)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::Cursor;
    use std::path::PathBuf;

    use colgrep::index::state::FileInfo;
    use colgrep::{IndexState, INDEX_FORMAT_VERSION};
    use tempfile::TempDir;

    use super::super::search::{check_index_generation, validate_server_index_state};

    #[derive(Debug, PartialEq, Eq)]
    struct SearchCall {
        query: String,
        top_k: usize,
        semantic_only: bool,
        code_only: bool,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
    }

    struct MockService {
        searches: RefCell<Vec<SearchCall>>,
        backend_error: bool,
        busy_error: bool,
        stale_error: bool,
        serialization_error: bool,
        large_result: bool,
        default_top_k: usize,
    }

    impl Default for MockService {
        fn default() -> Self {
            Self {
                searches: RefCell::new(Vec::new()),
                backend_error: false,
                busy_error: false,
                stale_error: false,
                serialization_error: false,
                large_result: false,
                default_top_k: 15,
            }
        }
    }

    impl SearchService for MockService {
        fn search(
            &self,
            query: &str,
            top_k: usize,
            semantic_only: bool,
            code_only: bool,
            include_patterns: &[String],
            exclude_patterns: &[String],
        ) -> Result<Vec<SearchResult>> {
            self.searches.borrow_mut().push(SearchCall {
                query: query.to_string(),
                top_k,
                semantic_only,
                code_only,
                include_patterns: include_patterns.to_vec(),
                exclude_patterns: exclude_patterns.to_vec(),
            });
            if self.backend_error {
                return Err(anyhow::anyhow!("backend unavailable"));
            }
            if self.busy_error {
                return Err(anyhow::anyhow!(
                    "index-busy: the index is being updated; retry the request"
                ));
            }
            if self.stale_error {
                return Err(anyhow::anyhow!(
                    "stale-index: the index changed after the server loaded it"
                ));
            }
            if self.serialization_error || self.large_result {
                let mut unit = colgrep::CodeUnit::new(
                    "u".to_string(),
                    PathBuf::from("src/main.rs"),
                    1,
                    2,
                    colgrep::Language::Rust,
                    colgrep::UnitType::Function,
                    None,
                );
                if self.large_result {
                    unit.code = "x".repeat(MAX_RESULT_PAYLOAD_BYTES + 1);
                }
                return Ok(vec![SearchResult {
                    unit,
                    score: if self.serialization_error {
                        f32::NAN
                    } else {
                        1.0
                    },
                }]);
            }
            Ok(Vec::new())
        }

        fn health(&self) -> HealthInfo {
            HealthInfo {
                documents: 3,
                project_root: "/project".to_string(),
                model: "model".to_string(),
            }
        }

        fn default_top_k(&self) -> usize {
            self.default_top_k
        }
    }

    fn run(input: &str, service: &MockService) -> Vec<Value> {
        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();
        run_stdio(&mut reader, &mut output, service).unwrap();
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn protocol_error_matrix() {
        let cases = [
            "not json",
            r#"{"version":2,"id":"version","op":"health"}"#,
            r#"{"version":1,"op":"health"}"#,
            r#"{"version":1,"id":true,"op":"health"}"#,
            r#"{"version":1,"id":"operation"}"#,
            r#"{"version":1,"id":"unknown","op":"unknown"}"#,
            r#"{"version":1,"id":"query","op":"search"}"#,
            r#"{"version":1,"id":"empty","op":"search","query":""}"#,
            r#"{"version":1,"id":"limit","op":"search","query":"x","top_k":1001}"#,
            r#"{"version":1,"id":"glob","op":"search","query":"x","include":["["]}"#,
        ];
        let expected = [
            "invalid_json",
            "unsupported_version",
            "missing_id",
            "invalid_id",
            "missing_operation",
            "unsupported_operation",
            "missing_query",
            "empty_query",
            "top_k_too_large",
            "invalid_glob",
        ];
        let service = MockService::default();
        let mut input = cases.join("\n");
        input.push_str("\n{\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n");
        let responses = run(&input, &service);
        let codes: Vec<&str> = responses
            .iter()
            .take(expected.len())
            .map(|response| response["error"]["code"].as_str().unwrap())
            .collect();
        assert_eq!(codes, expected);
        assert!(service.searches.borrow().is_empty());
    }

    #[test]
    fn bounded_globs_preserve_common_braces_and_reject_pathological_expansion() {
        assert_eq!(
            prepare_glob_patterns(&["*.{ts,tsx}".to_string()]).unwrap(),
            vec!["*.ts", "*.tsx"]
        );

        let pathological = (0..9).map(|_| "{a,b}").collect::<Vec<_>>().join("");
        let error = validate_globs(&[pathological], "include").unwrap_err();
        assert_eq!(error.code, "glob_too_large");

        let too_many = (0..=MAX_GLOB_PATTERNS)
            .map(|index| format!("file{index}.rs"))
            .collect::<Vec<_>>();
        let error = validate_globs(&too_many, "exclude").unwrap_err();
        assert_eq!(error.code, "glob_too_large");
    }

    #[test]
    fn default_top_k_is_forwarded_and_explicit_options_are_preserved() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":17,\"op\":\"search\",\"query\":\"auth\",\"semantic_only\":true,\"code_only\":true,\"include\":[\"*.rs\"],\"exclude\":[\"*test*\"]}\n\
             {\"version\":1,\"id\":\"explicit\",\"op\":\"search\",\"query\":\"auth\",\"top_k\":7}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );

        assert_eq!(responses[0]["id"], 17);
        assert_eq!(responses[0]["results"], serde_json::json!([]));
        assert_eq!(
            service.searches.borrow().as_slice(),
            &[
                SearchCall {
                    query: "auth".to_string(),
                    top_k: 15,
                    semantic_only: true,
                    code_only: true,
                    include_patterns: vec!["*.rs".to_string()],
                    exclude_patterns: vec!["*test*".to_string()],
                },
                SearchCall {
                    query: "auth".to_string(),
                    top_k: 7,
                    semantic_only: false,
                    code_only: false,
                    include_patterns: vec![],
                    exclude_patterns: vec![],
                },
            ]
        );
    }

    #[test]
    fn success_and_health_envelopes_are_complete() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":\"search\",\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"health\",\"op\":\"health\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );

        assert_eq!(
            responses[0],
            serde_json::json!({
                "version": 1,
                "id": "search",
                "ok": true,
                "op": "search",
                "results": [],
            })
        );
        assert_eq!(
            responses[1],
            serde_json::json!({
                "version": 1,
                "id": "health",
                "ok": true,
                "op": "health",
                "status": "ok",
                "documents": 3,
                "project_root": "/project",
                "model": "model",
            })
        );
        assert_eq!(responses[2]["version"], 1);
        assert_eq!(responses[2]["id"], "stop");
        assert_eq!(responses[2]["ok"], true);
        assert_eq!(responses[2]["op"], "shutdown");
        assert_eq!(responses[2]["status"], "shutting_down");
    }

    #[test]
    fn backend_errors_are_protocol_errors() {
        let service = MockService {
            backend_error: true,
            ..Default::default()
        };
        let responses = run(
            "{\"version\":1,\"id\":1,\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );
        assert_eq!(responses[0]["version"], 1);
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[0]["op"], "search");
        assert_eq!(responses[0]["error"]["code"], "search_failed");
    }

    #[test]
    fn stale_and_busy_errors_have_distinct_protocol_codes() {
        let stale_service = MockService {
            stale_error: true,
            ..Default::default()
        };
        let stale = run(
            "{\"version\":1,\"id\":\"stale\",\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &stale_service,
        );
        assert_eq!(stale[0]["version"], 1);
        assert_eq!(stale[0]["id"], "stale");
        assert_eq!(stale[0]["op"], "search");
        assert_eq!(stale[0]["error"]["code"], "stale_index");

        let busy_service = MockService {
            busy_error: true,
            ..Default::default()
        };
        let busy = run(
            "{\"version\":1,\"id\":7,\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &busy_service,
        );
        assert_eq!(busy[0]["version"], 1);
        assert_eq!(busy[0]["id"], 7);
        assert_eq!(busy[0]["op"], "search");
        assert_eq!(busy[0]["error"]["code"], "index_busy");
    }

    #[test]
    fn result_serialization_errors_are_protocol_errors() {
        let service = MockService {
            serialization_error: true,
            ..Default::default()
        };
        let responses = run(
            "{\"version\":1,\"id\":1,\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[0]["error"]["code"], "serialization_error");
    }

    #[test]
    fn oversized_result_payload_is_a_protocol_error() {
        let service = MockService {
            large_result: true,
            ..Default::default()
        };
        let responses = run(
            "{\"version\":1,\"id\":1,\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );
        assert_eq!(responses[0]["version"], 1);
        assert_eq!(responses[0]["id"], 1);
        assert_eq!(responses[0]["op"], "search");
        assert_eq!(responses[0]["error"]["code"], "response_too_large");
    }

    #[test]
    fn shutdown_stops_processing_following_requests() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n\
             {\"version\":1,\"id\":1,\"op\":\"search\",\"query\":\"never\"}\n",
            &service,
        );
        assert_eq!(responses.len(), 1);
        assert!(service.searches.borrow().is_empty());
    }

    #[test]
    fn stale_generation_is_rejected_but_search_count_is_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let mut state = IndexState {
            index_format_version: INDEX_FORMAT_VERSION,
            ..Default::default()
        };
        state.files.insert(
            PathBuf::from("src/main.rs"),
            FileInfo {
                content_hash: 1,
                mtime: 2,
                size: 3,
            },
        );
        state.save(temp_dir.path()).unwrap();
        let vector_dir = temp_dir.path().join("index");
        std::fs::create_dir_all(&vector_dir).unwrap();
        std::fs::write(vector_dir.join("metadata.json"), r#"{"documents":1}"#).unwrap();
        let expected = IndexState::load(temp_dir.path())
            .unwrap()
            .generation_with_index_dir(temp_dir.path())
            .unwrap();

        let mut count_only = IndexState::load(temp_dir.path()).unwrap();
        count_only.search_count = 1;
        count_only.save(temp_dir.path()).unwrap();
        check_index_generation(temp_dir.path(), expected).unwrap();

        let mut changed = IndexState::load(temp_dir.path()).unwrap();
        changed
            .files
            .get_mut(&PathBuf::from("src/main.rs"))
            .unwrap()
            .size = 4;
        changed.save(temp_dir.path()).unwrap();
        let error = check_index_generation(temp_dir.path(), expected).unwrap_err();
        assert!(error.to_string().contains("stale-index"));
    }

    #[test]
    fn vector_metadata_rewrite_stales_generation_without_state_change() {
        let temp_dir = TempDir::new().unwrap();
        let mut state = IndexState {
            index_format_version: INDEX_FORMAT_VERSION,
            ..Default::default()
        };
        state.files.insert(
            PathBuf::from("src/main.rs"),
            FileInfo {
                content_hash: 1,
                mtime: 2,
                size: 3,
            },
        );
        state.save(temp_dir.path()).unwrap();
        let vector_dir = temp_dir.path().join("index");
        std::fs::create_dir_all(&vector_dir).unwrap();
        let metadata_path = vector_dir.join("metadata.json");
        std::fs::write(&metadata_path, r#"{"documents":1}"#).unwrap();
        let expected = IndexState::load(temp_dir.path())
            .unwrap()
            .generation_with_index_dir(temp_dir.path())
            .unwrap();

        // Same state.json and same-length metadata rewrite: only the vector
        // publication fingerprint changes.
        std::fs::write(&metadata_path, r#"{"documents":2}"#).unwrap();
        let error = check_index_generation(temp_dir.path(), expected).unwrap_err();
        assert!(error.to_string().starts_with("stale-index:"));
    }

    #[test]
    fn startup_rejects_dirty_or_invalid_state() {
        let valid_dir = TempDir::new().unwrap();
        let mut valid = IndexState {
            index_format_version: INDEX_FORMAT_VERSION,
            ..Default::default()
        };
        valid.files.insert(
            PathBuf::from("src/main.rs"),
            FileInfo {
                content_hash: 1,
                mtime: 2,
                size: 3,
            },
        );
        valid.save(valid_dir.path()).unwrap();
        std::fs::create_dir_all(valid_dir.path().join("index")).unwrap();
        std::fs::write(
            valid_dir.path().join("index/metadata.json"),
            r#"{"documents":1}"#,
        )
        .unwrap();
        assert!(validate_server_index_state(valid_dir.path()).is_ok());
        std::fs::write(valid_dir.path().join(".building"), "").unwrap();
        let error = validate_server_index_state(valid_dir.path()).unwrap_err();
        assert!(error.to_string().contains("still being built"));

        let temp_dir = TempDir::new().unwrap();
        let dirty = IndexState {
            index_format_version: INDEX_FORMAT_VERSION,
            dirty: true,
            ..Default::default()
        };
        dirty.save(temp_dir.path()).unwrap();
        let error = validate_server_index_state(temp_dir.path()).unwrap_err();
        assert!(error.to_string().contains("dirty"));

        let invalid_dir = TempDir::new().unwrap();
        let invalid = IndexState::default();
        invalid.save(invalid_dir.path()).unwrap();
        let error = validate_server_index_state(invalid_dir.path()).unwrap_err();
        assert!(error.to_string().contains("Invalid index state"));
    }

    #[test]
    fn eof_after_a_request_exits_cleanly() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":\"eof\",\"op\":\"health\"}\n",
            &service,
        );
        assert_eq!(responses[0]["id"], "eof");
        assert_eq!(responses[0]["ok"], true);
        assert!(responses[0].get("pool_factor").is_none());
    }

    #[test]
    fn oversized_lines_are_drained_and_do_not_kill_the_server() {
        let service = MockService::default();
        let mut input = "x".repeat(MAX_REQUEST_LINE_BYTES + 1);
        input.push('\n');
        input.push_str("{\"version\":1,\"id\":\"ok\",\"op\":\"health\"}\n");
        let responses = run(&input, &service);
        assert_eq!(responses[0]["error"]["code"], "request_too_large");
        assert_eq!(responses[1]["id"], "ok");
    }
}
