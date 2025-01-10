use std::{path::PathBuf, process::Command};

use crate::build::remove_directory;

pub(super) fn local_repo_requires_update(
    local_repo_path: &PathBuf,
    repo_url: &str,
    repo_tag: &str,
) -> crate::Result<bool> {
    if !local_repo_path.exists() {
        crate::trace!("Directory does not exist: {}", local_repo_path.display());
        return Ok(true);
    }
    crate::trace!("Directory exists: {}", local_repo_path.display());

    // Check if it's a git repository
    let is_git_repo = Command::new("git")
        .current_dir(local_repo_path)
        .arg("rev-parse")
        .arg("--is-inside-work-tree")
        .status()?
        .success();

    if !is_git_repo {
        crate::trace!("Directory is not a git repo: {}", local_repo_path.display());
        return Ok(true);
    }

    crate::trace!("Directory is git repo: {} ", local_repo_path.display());

    // Check if it's the correct repository
    let remote_url = String::from_utf8(
        Command::new("git")
            .current_dir(local_repo_path)
            .args(["config", "--get", "remote.origin.url"])
            .output()?
            .stdout,
    )?;

    if remote_url.trim() != repo_url {
        crate::trace!("Incorrect remote URL: {} != {}", remote_url, repo_url);
        return Ok(true);
    }
    crate::trace!("{} == {} ", remote_url.trim(), repo_url);

    // Fetch the latest tags
    let _ = Command::new("git")
        .current_dir(local_repo_path)
        .args(["fetch", "--tags"])
        .status();

    // Check if the current HEAD is at the specified tag
    let is_at_tag = Command::new("git")
        .current_dir(local_repo_path)
        .args(["describe", "--tags", "--exact-match"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim() == repo_tag)
        .unwrap_or(false);

    if !is_at_tag {
        crate::trace!("Local repo not at tag {}", repo_tag);
        return Ok(true);
    }

    crate::trace!("{} == local repo tag - no update required", repo_tag);
    Ok(false)
}

pub(super) fn update_local_repo(
    local_repo_path: &PathBuf,
    repo_url: &str,
    repo_tag: &str,
) -> crate::Result<()> {
    remove_directory(local_repo_path)?;
    crate::trace!("Cloning {repo_url}  at tag {repo_tag}");
    Command::new("git")
        .arg("clone")
        .arg("--depth=1") // Shallow clone to save bandwidth and time
        .arg(format!("--branch={}", repo_tag))
        .arg("--recursive")
        .arg(repo_url)
        .arg(local_repo_path)
        .status()?;
    crate::trace!("Successfully cloned {repo_url}  at tag {repo_tag}");
    Ok(())
}
