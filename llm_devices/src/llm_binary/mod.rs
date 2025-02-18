// Internal modules
mod binary;
mod build;
mod download;
mod zip;

// Internal imports
use crate::{get_target_directory, logging::LoggingConfig};
use binary::executable_is_ok;
use build::cmake_is_available;
use std::path::PathBuf;

// Public exports
pub use binary::{get_bin_dir, get_bin_path, LLAMA_CPP_SERVER_EXECUTABLE};

pub const TARGET_DIR_SUB_PATH: &str = "llama_cpp";

fn get_target_dir_sub_dir() -> crate::Result<PathBuf> {
    Ok(get_target_directory()?.join(TARGET_DIR_SUB_PATH))
}

/// Downloads release at a specified tag with appropriate platform-specific optimizations.
///
/// This function handles the complete process of:
/// - Checking if a repository needs updating or building
/// - Cloning/updating the repository to the specified tag
/// - Building the repository with provided configuration
/// - Cleaning up on build failures
///
/// # Arguments
///
/// * `try_build` - Attempt to build the repository if true, otherwise install the binary
/// * `repo_url` - URL of the git repository to clone
/// * `repo_tag` - Specific git tag to checkout
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
/// use llm_devices::build_or_install;
///
/// let result = build_or_install(
///     "llama_cpp",
///     "https://github.com/ggml-org/llama.cpp",
///     "b3943",
/// );
/// ```
pub fn build_or_install(try_build: bool, repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    let mut logger = LoggingConfig {
        logger_name: format!("{}.build", get_target_dir_sub_dir()?.display()),
        build_log: true,
        level: tracing::Level::TRACE,
        ..Default::default()
    };
    logger.load_logger()?;

    match build_or_install_inner(try_build, repo_url, repo_tag) {
        Ok(_) => (),
        Err(e) => {
            crate::error!("Build failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn build_or_install_inner(try_build: bool, repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    if executable_is_ok(repo_tag)? {
        return Ok(());
    } else {
        remove_directory(&get_target_dir_sub_dir()?)?;
    }

    if try_build {
        if cmake_is_available() {
            if let Err(e) = build_binary(repo_url, repo_tag) {
                crate::error!("Build failed: {}", e);
            }
        } else {
            crate::error!("Cmake not found, installing binary instead");
            if let Err(e) = install_binary(repo_url, repo_tag) {
                crate::error!("Install failed: {}", e);
            }
        }
    } else {
        if let Err(e) = install_binary(repo_url, repo_tag) {
            crate::error!("Install failed: {}", e);
        }
    };

    if executable_is_ok(repo_tag)? {
        crate::trace!(
            "Build succeeded for {}",
            get_target_dir_sub_dir()?.display()
        );
    } else {
        remove_directory(&get_target_dir_sub_dir()?)?;
        crate::bail!("Build failed for {}", get_target_dir_sub_dir()?.display());
    };
    Ok(())
}

pub(super) fn install_binary(repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    download::binary(repo_url, repo_tag)?;
    zip::extract_zip()?;
    binary::set_binary(repo_tag)?;

    Ok(())
}

pub(super) fn build_binary(repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    download::source(repo_url, repo_tag)?;
    zip::extract_zip()?;
    build::cmake_project_buildsystem(&get_target_dir_sub_dir()?)?;
    build::cmake_build_project(&get_target_dir_sub_dir()?)?;
    binary::set_binary(repo_tag)?;

    Ok(())
}

fn remove_directory(dir_path: &PathBuf) -> crate::Result<()> {
    if !dir_path.exists() {
        crate::trace!(
            "Directory {} does not exist, skipping removal",
            dir_path.display()
        );
        return Ok(());
    }

    match std::fs::remove_dir_all(dir_path) {
        Ok(_) => {
            crate::trace!("Successfully removed directory {}", dir_path.display());
            Ok(())
        }
        Err(e) => {
            crate::bail!("Failed to remove directory {}: {}", dir_path.display(), e)
        }
    }
}
