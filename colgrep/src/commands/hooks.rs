use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use colgrep::{find_parent_index, index_exists, Config, DEFAULT_MODEL};

/// Maximum number of files for a "small project" where we enable colgrep
/// even without a pre-existing index, so the first search auto-creates one quickly.
const SMALL_PROJECT_FILE_LIMIT: usize = 50;

/// Check if colgrep context should be injected.
/// Returns true if:
/// - An index (for the currently selected model) already exists for this project
///   or a parent project, OR
/// - The project is small enough that auto-indexing on first search is fast
fn should_inject_colgrep_context(project_root: &Path) -> bool {
    let model = Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    index_exists(project_root, &model)
        || matches!(find_parent_index(project_root, &model), Ok(Some(_)))
        || is_small_project(project_root)
}

/// Quick check whether the project has few enough files that colgrep can
/// index it on-the-fly without noticeable delay. Walks respecting .gitignore
/// and stops counting as soon as we exceed the threshold.
fn is_small_project(root: &Path) -> bool {
    let walker = WalkBuilder::new(root)
        .hidden(true) // skip hidden files/dirs
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .git_exclude(true)
        .max_depth(Some(10))
        .build();

    let mut count = 0usize;
    for entry in walker {
        let Ok(entry) = entry else { continue };
        // Only count files, not directories
        if entry.file_type().is_some_and(|ft| ft.is_file()) {
            count += 1;
            if count > SMALL_PROJECT_FILE_LIMIT {
                return false;
            }
        }
    }
    count > 0
}

/// Claude Code session hook - outputs JSON reminder for semantic search
pub fn cmd_session_hook() -> Result<()> {
    let cwd = std::env::current_dir()?;

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        let response = serde_json::json!({});
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }

    // Output the hook response with additional context
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": concat!(
                "This machine has colgrep installed - an open-source semantic code-search CLI ",
                "(https://github.com/lightonai/next-plaid) that runs fully locally. ",
                "The user set up this hook via `colgrep install` and prefers colgrep over Grep/Glob ",
                "for code search: it combines semantic and regex matching over a pre-built index. ",
                "Its CLI behaves exactly like grep (`-e`, `-F`, `-w`, `-s`, `-r`, `--include`, `--exclude`), ",
                "with an optional semantic query on top.\n\n",
                "USAGE (start broad with a semantic query, then narrow with regex/paths/filters):\n",
                "- Semantic: `colgrep \"error handling logic\"`\n",
                "- Regex only: `colgrep -e \"async fn\\s+\\w+\"`\n",
                "- Hybrid (regex pre-filter + semantic ranking): `colgrep -e \"Result<\" \"error handling\"`\n",
                "- Pattern modes: `-F` (fixed string) | `-w` (whole word) | `-s` (case-sensitive)\n",
                "- Scope to dirs/files: `colgrep \"auth\" ./src/api ./main.rs`\n",
                "- Include: `--include=\"*.{ts,tsx}\"`, `--include=\"src/**/*.rs\"` | exclude: `--exclude=\"*_test.go\"`, `--exclude-dir=vendor`\n",
                "- Skip docs/config files: `--code-only`\n",
                "- Output: `-k 20` (more results) | `-l` (file names only) | `-c` (full function bodies) | `-n 10` (context lines) | `--json`\n",
                "- Pure semantic (no keyword boost): `colgrep --semantic-only \"how is the index kept in sync\"`\n",
                "- Ranking weight: `--alpha 0.8` (keyword vs semantic balance, default 0.6)\n",
                "- Combine: `colgrep -e \"impl.*Auth\" \"trait implementation\" ./src --include=\"*.rs\" -k 20 -c`\n\n",
                "NOTES:\n",
                "- The first query may take 30-90s (model load + index build); later ones take <5s. ",
                "Run it in the foreground and wait for the result rather than launching a parallel grep for the same thing.\n",
                "- If there are no results, broaden the semantic terms or try regex-only mode (-e).\n",
                "- Plain Grep/Glob remain fine for exact strings or filename lookups when that fits better.\n\n",
                "When you spawn subagents that will search code, mention that colgrep is available and how to invoke it.\n",
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Claude Code task hook - outputs JSON reminder to include colgrep in agent prompts
/// This is triggered when the Task tool is used to spawn agents
pub fn cmd_task_hook() -> Result<()> {
    let cwd = std::env::current_dir()?;

    // Check if we should inject colgrep context
    if !should_inject_colgrep_context(&cwd) {
        // Return empty response - don't tell model about colgrep
        let response = serde_json::json!({});
        println!("{}", serde_json::to_string(&response)?);
        return Ok(());
    }

    // Output the hook response with detailed agent instructions
    let response = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": concat!(
                "Reminder from the user's `colgrep install` hook: if this agent will search code, ",
                "include a note like this in its prompt:\n",
                "---\n",
                "This project has `colgrep` installed - an open-source semantic code-search CLI ",
                "(https://github.com/lightonai/next-plaid) that runs fully locally. ",
                "The user prefers it over Grep/Glob for code search; its CLI behaves exactly like grep, ",
                "with an optional semantic query on top.\n",
                "- Semantic: `colgrep \"error handling\" -k 10` | pure semantic: `--semantic-only`\n",
                "- Regex + semantic: `colgrep -e \"fn.*test\" \"unit tests\"`\n",
                "- Regex only: `colgrep -e \"async fn\"` | scope: `colgrep \"query\" ./src`\n",
                "- Filters: `--include=\"*.{ts,tsx}\"`, `--exclude=\"*_test.go\"`, `--exclude-dir=vendor`, `--code-only`\n",
                "- Output: `-l` (file names only) | `-c` (full function bodies) | `-k 20` (more results)\n",
                "The first query may take 30-90s (index build); later ones take <5s - run it in the ",
                "foreground and wait. If there are no results, broaden the terms or use regex-only mode. ",
                "Plain grep stays fine for exact-string lookups.\n",
                "---"
            )
        }
    });

    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
