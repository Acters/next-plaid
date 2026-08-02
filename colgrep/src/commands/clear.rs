use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use colgrep::{
    acquire_index_lock, find_parent_index, get_colgrep_data_dir, get_index_dir_for_project, Config,
    ProjectMetadata, DEFAULT_MODEL,
};

/// Remove an index while retaining exclusive synchronization on a stable lock
/// inode in the parent directory. Deleting the directory itself removes the
/// lock file, so the guard must live outside the index directory.
fn clear_index_directory(index_path: &Path) -> Result<()> {
    let Some(name) = index_path.file_name() else {
        anyhow::bail!(
            "Cannot clear index directory without a name: {}",
            index_path.display()
        );
    };
    let Some(parent) = index_path.parent() else {
        anyhow::bail!(
            "Cannot clear index directory without a parent: {}",
            index_path.display()
        );
    };

    // The .lock suffix is also ignored by colgrep's index directory discovery.
    let lock_dir = parent.join(format!(".{}.clear-lock", name.to_string_lossy()));
    std::fs::create_dir_all(&lock_dir).with_context(|| {
        format!(
            "Failed to create clear lock directory at {}",
            lock_dir.display()
        )
    })?;
    let _clear_guard = acquire_index_lock(&lock_dir).with_context(|| {
        format!(
            "Failed to acquire clear lock for index {}",
            index_path.display()
        )
    })?;

    // Coordinate with regular readers/writers for the index itself. Keep both
    // guards held until deletion and lock cleanup are complete.
    let _index_guard = acquire_index_lock(index_path)?;
    std::fs::remove_dir_all(index_path)
        .with_context(|| format!("Failed to clear index at {}", index_path.display()))?;
    std::fs::remove_dir_all(&lock_dir).with_context(|| {
        format!(
            "Cleared index but failed to remove clear lock directory {}",
            lock_dir.display()
        )
    })?;
    Ok(())
}

fn current_model() -> String {
    Config::load()
        .ok()
        .and_then(|c| c.get_default_model().map(|s| s.to_string()))
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

pub fn cmd_clear(path: &PathBuf, all: bool) -> Result<()> {
    if all {
        // Clear all indexes (every model, every project)
        let data_dir = get_colgrep_data_dir()?;
        if !data_dir.exists() {
            println!("No indexes found.");
            return Ok(());
        }

        // Collect index directories and their project paths
        let index_dirs: Vec<_> = std::fs::read_dir(&data_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        if index_dirs.is_empty() {
            println!("No indexes found.");
            return Ok(());
        }

        // Delete each index and log the project path
        for entry in &index_dirs {
            let index_path = entry.path();
            let (project_path, model) = match ProjectMetadata::load(&index_path) {
                Ok(m) => (m.project_path.display().to_string(), m.model),
                Err(_) => (index_path.display().to_string(), None),
            };

            clear_index_directory(&index_path)?;
            match model {
                Some(m) => println!("🗑️  Cleared index for {} [{}]", project_path, m),
                None => println!("🗑️  Cleared index for {}", project_path),
            }
        }

        println!("\n✅ Cleared {} index(es)", index_dirs.len());
    } else {
        // Clear index for current project, scoped to the active model only.
        // Other models' indexes for the same project are left intact.
        let path = std::fs::canonicalize(path)?;
        let model = current_model();
        let index_dir = get_index_dir_for_project(&path, &model)?;

        if index_dir.exists() {
            // Exact match found - clear it.
            clear_index_directory(&index_dir)?;
            println!("🗑️  Cleared index for {} [{}]", path.display(), model);
        } else if let Some(parent_info) = find_parent_index(&path, &model)? {
            // We're in a subdirectory of an indexed project (for this model) -
            // clear the parent index.
            clear_index_directory(&parent_info.index_dir)?;
            println!(
                "🗑️  Cleared index for {} [{}] (parent of current directory)",
                parent_info.project_path.display(),
                model
            );
        } else {
            println!("No index found for {} [{}]", path.display(), model);
            return Ok(());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use colgrep::try_acquire_index_read_lock;

    #[test]
    fn clear_removes_index_and_allows_recreation() {
        let temp = tempfile::tempdir().unwrap();
        let index_dir = temp.path().join("index-1");
        std::fs::create_dir_all(&index_dir).unwrap();
        let prepared_guard = acquire_index_lock(&index_dir).unwrap();
        drop(prepared_guard);
        assert!(index_dir.join(".lock").exists());

        clear_index_directory(&index_dir).unwrap();

        assert!(!index_dir.exists());
        assert!(!temp.path().join(".index-1.clear-lock").exists());

        std::fs::create_dir_all(&index_dir).unwrap();
        let guard = acquire_index_lock(&index_dir).unwrap();
        assert!(try_acquire_index_read_lock(&index_dir).unwrap().is_none());
        drop(guard);
        assert!(try_acquire_index_read_lock(&index_dir).unwrap().is_some());
    }
}
