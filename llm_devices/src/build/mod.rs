use std::path::{Path, PathBuf};

use crate::logging::LoggingConfig;

mod git;
mod make;

pub fn run(
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

/// Hack to resolve this cargo issue
/// https://github.com/rust-lang/cargo/issues/9661
pub fn get_target_directory() -> crate::Result<PathBuf> {
    // First, check CARGO_TARGET_DIR environment variable
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return Ok(PathBuf::from(target_dir));
    }
    // Next, check OUT_DIR and traverse up to find 'target'
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        if let Some(target_dir) = find_target_in_ancestors(&PathBuf::from(out_dir)) {
            return Ok(target_dir);
        }
    }

    // If that fails, check CARGO_MANIFEST_DIR and traverse up to find 'target'
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        if let Some(target_dir) = find_target_in_ancestors(&PathBuf::from(manifest_dir)) {
            return Ok(target_dir);
        }
    }

    // As a last resort, use the compile-time CARGO_MANIFEST_DIR
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match find_target_in_ancestors(&manifest_dir) {
        Some(target_dir) => Ok(target_dir),
        None => crate::bail!(
            "Could not find target directory in ancestors of {}",
            manifest_dir.display()
        ),
    }
}

fn find_target_in_ancestors(start_dir: &Path) -> Option<PathBuf> {
    start_dir
        .ancestors()
        .find(|path| path.join("target").is_dir())
        .map(|path| path.join("target"))
}
