use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use colgrep::{ensure_model, Config, Searcher};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct QueryRecord {
    id: String,
    query: String,
}

#[derive(Debug, Serialize)]
struct ResultIdentity {
    file: String,
    line: usize,
    end_line: usize,
    score: f32,
}

#[derive(Debug, Serialize)]
struct Record {
    warmup: bool,
    cycle: usize,
    id: String,
    elapsed_ms: f64,
    count: usize,
}

#[derive(Debug, Serialize)]
struct Summary {
    n: usize,
    mean_ms: f64,
    median_ms: f64,
    p05_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
}

#[derive(Debug, Serialize)]
struct Output {
    label: String,
    project: String,
    model: String,
    query_count: usize,
    warmup_cycles: usize,
    measured_cycles: usize,
    top_k: usize,
    load_ms: f64,
    encode_ms: Summary,
    search_ms: Summary,
    results: std::collections::BTreeMap<String, Vec<ResultIdentity>>,
    records: Vec<Record>,
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = (sorted.len() - 1) as f64 * q;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn summarize(values: &[f64]) -> Summary {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Summary {
        n: values.len(),
        mean_ms: mean,
        median_ms: quantile(&sorted, 0.5),
        p05_ms: quantile(&sorted, 0.05),
        p95_ms: quantile(&sorted, 0.95),
        min_ms: sorted[0],
        max_ms: *sorted.last().unwrap(),
        stddev_ms: variance.sqrt(),
    }
}

fn relative_file(project: &Path, file: &Path) -> String {
    file.strip_prefix(project)
        .unwrap_or(file)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn parse_usize(value: &str, label: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("Invalid {label}: {value}"))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 9 {
        anyhow::bail!(
            "usage: dreb_core_benchmark LABEL PROJECT MODEL QUERIES OUT WARMUP_CYCLES MEASURED_CYCLES TOP_K"
        );
    }
    let label = args[1].clone();
    let project = fs::canonicalize(&args[2]).context("Project does not exist")?;
    let model = args[3].clone();
    let query_path = PathBuf::from(&args[4]);
    let output_path = PathBuf::from(&args[5]);
    let warmup_cycles = parse_usize(&args[6], "warmup cycles")?;
    let measured_cycles = parse_usize(&args[7], "measured cycles")?;
    let top_k = parse_usize(&args[8], "top k")?;
    if measured_cycles == 0 || top_k == 0 {
        anyhow::bail!("measured cycles and top k must be positive");
    }
    let queries: Vec<QueryRecord> =
        serde_json::from_slice(&fs::read(query_path).context("Failed to read queries")?)?;
    if queries.is_empty() {
        anyhow::bail!("Query list is empty");
    }

    let config = Config::load().unwrap_or_default();
    let model_path = ensure_model(Some(&model), true)?;
    let load_started = Instant::now();
    let searcher =
        Searcher::load_read_only_with_quantized(&project, &model, &model_path, !config.use_fp32())?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let mut embeddings = Vec::with_capacity(queries.len());
    let mut encode_times = Vec::with_capacity(queries.len());
    for query in &queries {
        let started = Instant::now();
        let embedding = searcher.encode_query(&query.query)?;
        encode_times.push(started.elapsed().as_secs_f64() * 1000.0);
        embeddings.push(embedding);
    }

    let mut records = Vec::new();
    let mut results = std::collections::BTreeMap::new();
    for (warmup, cycles) in [(true, warmup_cycles), (false, measured_cycles)] {
        for cycle in 0..cycles {
            let rotation = (cycle * 7 + usize::from(warmup)) % queries.len();
            for offset in 0..queries.len() {
                let index = (offset + rotation) % queries.len();
                let started = Instant::now();
                let found = searcher.search_with_embedding(&embeddings[index], top_k, None)?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                records.push(Record {
                    warmup,
                    cycle,
                    id: queries[index].id.clone(),
                    elapsed_ms,
                    count: found.len(),
                });
                if !warmup {
                    results.insert(
                        queries[index].id.clone(),
                        found
                            .iter()
                            .map(|result| ResultIdentity {
                                file: relative_file(&project, &result.unit.file),
                                line: result.unit.line,
                                end_line: result.unit.end_line,
                                score: result.score,
                            })
                            .collect(),
                    );
                }
            }
        }
    }
    let search_times: Vec<f64> = records
        .iter()
        .filter(|record| !record.warmup)
        .map(|record| record.elapsed_ms)
        .collect();
    let output = Output {
        label,
        project: project.display().to_string(),
        model,
        query_count: queries.len(),
        warmup_cycles,
        measured_cycles,
        top_k,
        load_ms,
        encode_ms: summarize(&encode_times),
        search_ms: summarize(&search_times),
        results,
        records,
    };
    fs::write(output_path, serde_json::to_vec_pretty(&output)?)?;
    println!("{}", serde_json::to_string_pretty(&output.search_ms)?);
    Ok(())
}
