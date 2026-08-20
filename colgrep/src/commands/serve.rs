use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};

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
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        query: &str,
        top_k: usize,
        semantic_only: bool,
        code_only: bool,
        include_patterns: &[String],
        exclude_patterns: &[String],
        subdir_filter: Option<&Path>,
    ) -> Result<Vec<SearchResult>>;

    fn health(&self) -> HealthInfo;

    fn default_top_k(&self) -> usize;

    /// Estimate the serialized search-result payload before JSON materialization.
    ///
    /// The default is deliberately conservative. Implementations that return a
    /// finite estimate avoid the protocol's unbounded fallback serialization.
    fn estimate_serialized_results_bytes(&self, results: &[SearchResult]) -> Option<usize> {
        let _ = results;
        None
    }
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
        subdir_filter: Option<&Path>,
    ) -> Result<Vec<SearchResult>> {
        self.search(
            query,
            top_k,
            semantic_only,
            code_only,
            include_patterns,
            exclude_patterns,
            subdir_filter,
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

    fn estimate_serialized_results_bytes(&self, results: &[SearchResult]) -> Option<usize> {
        Some(estimate_search_results_json_bytes(results))
    }
}

fn estimate_search_results_json_bytes(results: &[SearchResult]) -> usize {
    // Exact field/number overhead plus string lengths. Escaped JSON strings are
    // longer than their raw UTF-8 values, so inflate string payloads by a fixed
    // factor and add a final 25% safety margin.
    let per_result = 512usize;
    let estimated = results.iter().fold(2usize, |total, result| {
        let unit = &result.unit;
        let strings = [
            unit.name.len(),
            unit.qualified_name.len(),
            unit.file.to_string_lossy().len(),
            unit.signature.len(),
            unit.code.len(),
            unit.docstring.as_deref().map_or(0, str::len),
            unit.return_type.as_deref().map_or(0, str::len),
            unit.extends.as_deref().map_or(0, str::len),
            unit.parent_class.as_deref().map_or(0, str::len),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
        .saturating_add(unit.parameters.iter().map(String::len).sum::<usize>())
        .saturating_add(unit.calls.iter().map(String::len).sum::<usize>())
        .saturating_add(unit.called_by.iter().map(String::len).sum::<usize>())
        .saturating_add(unit.variables.iter().map(String::len).sum::<usize>())
        .saturating_add(unit.imports.iter().map(String::len).sum::<usize>());
        total
            .saturating_add(per_result)
            .saturating_add(strings.saturating_mul(2))
    });
    estimated.saturating_add(estimated / 4)
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
    restrict_to_dir: Option<PathBuf>,
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

fn normalize_restrict_to_dir(value: &Path) -> std::result::Result<PathBuf, String> {
    if value.as_os_str().is_empty() {
        return Err("restrict_to_dir must name a descendant directory".to_string());
    }
    if value.to_string_lossy().contains('\0') {
        return Err("restrict_to_dir must not contain a NUL byte".to_string());
    }

    let mut normalized = PathBuf::new();
    for component in value.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => {
                return Err("restrict_to_dir must be a relative path".to_string());
            }
            Component::ParentDir => {
                return Err("restrict_to_dir must not contain parent traversal".to_string());
            }
            Component::CurDir => {}
            Component::Normal(component) => normalized.push(component),
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err("restrict_to_dir must name a descendant directory".to_string());
    }
    Ok(normalized)
}

fn validate_request(
    mut request: Request,
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
            request.restrict_to_dir = match request.restrict_to_dir.as_deref() {
                Some(value) => {
                    Some(
                        normalize_restrict_to_dir(value).map_err(|message| ProtocolError {
                            op: Some(op.to_string()),
                            code: "invalid_restriction",
                            message,
                        })?,
                    )
                }
                None => None,
            };
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

fn serialize_results<S: SearchService>(
    service: &S,
    results: Vec<SearchResult>,
) -> std::result::Result<Value, SerializedResultsError> {
    if results.iter().any(|result| !result.score.is_finite()) {
        return Err(SerializedResultsError::Serialization(
            "search result contains a non-finite score".to_string(),
        ));
    }

    // Reject an obviously oversized result before serializing it. If the
    // service cannot provide a useful estimate, retain bounded verification
    // with the existing byte-cap serialization path.
    if let Some(estimated_bytes) = service.estimate_serialized_results_bytes(&results) {
        if estimated_bytes > MAX_RESULT_PAYLOAD_BYTES {
            return Err(SerializedResultsError::TooLarge {
                bytes: estimated_bytes,
            });
        }
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
            request.restrict_to_dir.as_deref(),
        ) {
            Ok(results) => match serialize_results(service, results) {
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
                } else if message.starts_with("stale-source:") {
                    "stale_source"
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

    use super::super::search::{checked_index_state, validate_server_index_state};

    #[derive(Debug, PartialEq, Eq)]
    struct SearchCall {
        query: String,
        top_k: usize,
        semantic_only: bool,
        code_only: bool,
        include_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
        subdir_filter: Option<PathBuf>,
    }

    struct MockService {
        searches: RefCell<Vec<SearchCall>>,
        backend_error: bool,
        busy_error: bool,
        stale_error: bool,
        stale_source_error: bool,
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
                stale_source_error: false,
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
            subdir_filter: Option<&Path>,
        ) -> Result<Vec<SearchResult>> {
            self.searches.borrow_mut().push(SearchCall {
                query: query.to_string(),
                top_k,
                semantic_only,
                code_only,
                include_patterns: include_patterns.to_vec(),
                exclude_patterns: exclude_patterns.to_vec(),
                subdir_filter: subdir_filter.map(Path::to_path_buf),
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
            if self.stale_source_error {
                return Err(anyhow::anyhow!(
                    "stale-source: project files changed after indexing"
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

        fn estimate_serialized_results_bytes(&self, results: &[SearchResult]) -> Option<usize> {
            Some(estimate_search_results_json_bytes(results))
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
            r#"{"version":1,"id":"absolute","op":"search","query":"x","restrict_to_dir":"/outside"}"#,
            r#"{"version":1,"id":"parent","op":"search","query":"x","restrict_to_dir":"src/../outside"}"#,
            r#"{"version":1,"id":"empty","op":"search","query":"x","restrict_to_dir":""}"#,
            r#"{"version":1,"id":"current","op":"search","query":"x","restrict_to_dir":"."}"#,
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
            "invalid_restriction",
            "invalid_restriction",
            "invalid_restriction",
            "invalid_restriction",
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
    fn restriction_errors_echo_the_request_as_normal_search_errors() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":\"bad-path\",\"op\":\"search\",\"query\":\"x\",\"restrict_to_dir\":\"../outside\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &service,
        );
        assert_eq!(responses[0]["version"], 1);
        assert_eq!(responses[0]["id"], "bad-path");
        assert_eq!(responses[0]["ok"], false);
        assert_eq!(responses[0]["op"], "search");
        assert_eq!(responses[0]["error"]["code"], "invalid_restriction");
        assert_eq!(responses[1]["ok"], true);
        assert!(service.searches.borrow().is_empty());
    }

    #[test]
    fn restriction_paths_are_normalized_without_filesystem_access() {
        assert_eq!(
            normalize_restrict_to_dir(Path::new("./src//nested/./code")).unwrap(),
            PathBuf::from("src/nested/code")
        );
        for value in ["", ".", "./", "../outside", "src/../../outside", "/outside"] {
            assert!(
                normalize_restrict_to_dir(Path::new(value)).is_err(),
                "{value}"
            );
        }
        #[cfg(windows)]
        {
            assert!(normalize_restrict_to_dir(Path::new(r"C:\\outside")).is_err());
            assert!(normalize_restrict_to_dir(Path::new(r"\\server\\share")).is_err());
        }
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

        let exact_pattern_count = (0..MAX_GLOB_PATTERNS)
            .map(|index| format!("file{index}.rs"))
            .collect::<Vec<_>>();
        assert!(validate_globs(&exact_pattern_count, "include").is_ok());
        let too_many = (0..=MAX_GLOB_PATTERNS)
            .map(|index| format!("file{index}.rs"))
            .collect::<Vec<_>>();
        let error = validate_globs(&too_many, "exclude").unwrap_err();
        assert_eq!(error.code, "glob_too_large");

        let exact_expansion = (0..8).map(|_| "{a,b}").collect::<Vec<_>>().join("");
        assert!(validate_globs(&[exact_expansion], "include").is_ok());
        let over_expansion = (0..9).map(|_| "{a,b}").collect::<Vec<_>>().join("");
        let error = validate_globs(&[over_expansion], "include").unwrap_err();
        assert_eq!(error.code, "glob_too_large");

        let exact_length = "x".repeat(4096);
        assert!(validate_globs(&[exact_length], "include").is_ok());
        let too_long = "x".repeat(4097);
        let error = validate_globs(&[too_long], "include").unwrap_err();
        assert_eq!(error.code, "glob_too_large");

        let nested_over_limit = (0..15).map(|_| "{a,").collect::<String>() + "a" + &"}".repeat(15);
        let error = validate_globs(&[nested_over_limit], "include").unwrap_err();
        assert!(
            matches!(error.code, "glob_too_large" | "invalid_glob"),
            "deep nesting must be rejected before execution: {}",
            error.message
        );
    }

    #[test]
    fn parseable_requests_with_invalid_field_types_are_protocol_errors() {
        let cases = [
            r#"{"version":1,"id":"query-number","op":"search","query":1}"#,
            r#"{"version":1,"id":"top-k-negative","op":"search","query":"x","top_k":-1}"#,
            r#"{"version":1,"id":"include-string","op":"search","query":"x","include":"*.rs"}"#,
            r#"{"version":1,"id":"semantic-string","op":"search","query":"x","semantic_only":"yes"}"#,
        ];
        let service = MockService::default();
        let mut input = cases.join("\n");
        input.push_str("\n{\"version\":1,\"id\":\"after\",\"op\":\"health\"}\n");
        let responses = run(&input, &service);

        assert_eq!(responses.len(), cases.len() + 1);
        for (index, response) in responses.iter().take(cases.len()).enumerate() {
            assert_eq!(response["ok"], false, "case {index}");
            assert_eq!(response["error"]["code"], "invalid_json", "case {index}");
            assert_eq!(response["id"], serde_json::Value::Null, "case {index}");
        }
        assert_eq!(responses[cases.len()]["id"], "after");
        assert_eq!(responses[cases.len()]["ok"], true);
        assert!(service.searches.borrow().is_empty());
    }

    #[test]
    fn default_top_k_is_forwarded_and_explicit_options_are_preserved() {
        let service = MockService::default();
        let responses = run(
            "{\"version\":1,\"id\":17,\"op\":\"search\",\"query\":\"auth\",\"semantic_only\":true,\"code_only\":true,\"include\":[\"*.rs\"],\"exclude\":[\"*test*\"],\"restrict_to_dir\":\"./src//nested\"}\n\
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
                    subdir_filter: Some(PathBuf::from("src/nested")),
                },
                SearchCall {
                    query: "auth".to_string(),
                    top_k: 7,
                    semantic_only: false,
                    code_only: false,
                    include_patterns: vec![],
                    exclude_patterns: vec![],
                    subdir_filter: None,
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
    fn stale_source_index_and_busy_errors_have_distinct_protocol_codes() {
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

        let stale_source_service = MockService {
            stale_source_error: true,
            ..Default::default()
        };
        let stale_source = run(
            "{\"version\":1,\"id\":\"source\",\"op\":\"search\",\"query\":\"x\"}\n\
             {\"version\":1,\"id\":\"stop\",\"op\":\"shutdown\"}\n",
            &stale_source_service,
        );
        assert_eq!(stale_source[0]["version"], 1);
        assert_eq!(stale_source[0]["id"], "source");
        assert_eq!(stale_source[0]["op"], "search");
        assert_eq!(stale_source[0]["error"]["code"], "stale_source");

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
        checked_index_state(temp_dir.path(), expected).unwrap();

        let mut changed = IndexState::load(temp_dir.path()).unwrap();
        changed
            .files
            .get_mut(&PathBuf::from("src/main.rs"))
            .unwrap()
            .size = 4;
        changed.save(temp_dir.path()).unwrap();
        let error = checked_index_state(temp_dir.path(), expected).unwrap_err();
        assert!(error.to_string().contains("stale-index"));
    }

    #[test]
    fn metadata_io_failures_are_not_misclassified_as_stale_generation() {
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
        let expected = validate_server_index_state(temp_dir.path()).unwrap();

        std::fs::write(temp_dir.path().join("state.json"), b"not json").unwrap();
        let state_error = checked_index_state(temp_dir.path(), expected).unwrap_err();
        assert!(!format!("{state_error:#}").contains("stale-index:"));

        state.save(temp_dir.path()).unwrap();
        std::fs::remove_file(vector_dir.join("metadata.json")).unwrap();
        let metadata_error = checked_index_state(temp_dir.path(), expected).unwrap_err();
        assert!(!format!("{metadata_error:#}").contains("stale-index:"));
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
        let error = checked_index_state(temp_dir.path(), expected).unwrap_err();
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
