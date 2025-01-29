// Internal modules
mod git;
mod make;

// Internal imports
use crate::{get_target_directory, logging::LoggingConfig};
use std::path::PathBuf;

/// Clones and builds a repository at a specified tag with appropriate platform-specific optimizations.
///
/// This function handles the complete process of:
/// - Checking if a repository needs updating or building
/// - Cloning/updating the repository to the specified tag
/// - Building the repository with provided configuration
/// - Cleaning up on build failures
///
/// # Arguments
///
/// * `target_sub_path` - Subdirectory within the workspace's target directory where the repo will be cloned
/// * `repo_url` - URL of the git repository to clone
/// * `repo_tag` - Specific git tag to checkout
/// * `executable_name` - Name of the executable that should be built
/// * `builder_args` - Arguments to pass to the make command (e.g., ["llama-server", "BUILD_TYPE=Release"])
/// * `cuda_arg` - Optional CUDA-specific argument for the make command when building with CUDA support
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if build succeeds, Error if any step fails
///
/// # Errors
///
/// Returns error if:
/// - Git operations fail (clone, checkout)
/// - Build process fails
/// - Executable is not found after build
/// - Directory operations fail (create, remove)
///
/// # Example
///
/// ```no_run
/// use llm_devices::build_repo;
///
/// let result = build_repo(
///     "llama.cpp",
///     "https://github.com/ggerganov/llama.cpp",
///     "b3943",
///     "llama-server",
///     &["llama-server", "BUILD_TYPE=Release", "-j"],
///     &Some("GGML_CUDA=1")
/// );
/// ```
pub fn build_repo(
    target_sub_path: &str,
    repo_url: &str,
    repo_tag: &str,
    executable_name: &str,
    builder_args: &[&str],
    cuda_arg: &Option<&str>,
) -> crate::Result<()> {
    let mut logger = LoggingConfig {
        logger_name: format!("{}.build", target_sub_path),
        build_log: true,
        level: tracing::Level::TRACE,
        ..Default::default()
    };
    logger.load_logger()?;

    let local_repo_path = get_target_directory()?.join(target_sub_path);

    let (local_repo_requires_update, local_repo_requires_build) =
        check(&local_repo_path, repo_url, repo_tag, executable_name)?;

    if !local_repo_requires_update && !local_repo_requires_build {
        crate::trace!("No build required for {}", target_sub_path);

        return Ok(());
    }

    if local_repo_requires_update {
        git::update_local_repo(&local_repo_path, repo_url, repo_tag)?;
    }

    if local_repo_requires_build {
        match make::build_local_repo(&local_repo_path, builder_args, cuda_arg) {
            Ok(_) => (),
            Err(e) => {
                remove_directory(&local_repo_path)?;
                crate::bail!("Failed to build {}: {}", target_sub_path, e);
            }
        }
    }

    match check(&local_repo_path, repo_url, repo_tag, executable_name)? {
        (false, false) => {
            crate::trace!("Build succeeded for {}", target_sub_path);
        }
        _ => {
            remove_directory(&local_repo_path)?;
            crate::bail!("Build failed for {}", target_sub_path);
        }
    }

    Ok(())
}

fn check(
    local_repo_path: &PathBuf,
    repo_url: &str,
    repo_tag: &str,
    executable_name: &str,
) -> crate::Result<(bool, bool)> {
    let local_repo_requires_update =
        git::local_repo_requires_update(&local_repo_path, repo_url, repo_tag)?;
    let local_repo_requires_build = if local_repo_requires_update {
        true
    } else {
        make::local_repo_requires_build(&local_repo_path, executable_name)
    };

    Ok((local_repo_requires_update, local_repo_requires_build))
}

fn remove_directory(local_repo_path: &PathBuf) -> crate::Result<()> {
    if !local_repo_path.exists() {
        crate::trace!(
            "Directory {} does not exist, skipping removal",
            local_repo_path.display()
        );
        return Ok(());
    }

    match std::fs::remove_dir_all(local_repo_path) {
        Ok(_) => {
            crate::trace!(
                "Successfully removed directory {}",
                local_repo_path.display()
            );
            Ok(())
        }
        Err(e) => {
            crate::bail!(
                "Failed to remove directory {}: {}",
                local_repo_path.display(),
                e
            )
        }
    }
}
