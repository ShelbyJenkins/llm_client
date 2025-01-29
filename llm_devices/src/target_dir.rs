use std::path::{Path, PathBuf};

/// Resolves the Cargo target directory path.
///
/// This function attempts to find the target directory through multiple methods:
/// 1. Checks the `CARGO_TARGET_DIR` environment variable
/// 2. Traverses up from the `OUT_DIR` to find a 'target' directory
/// 3. Traverses up from `CARGO_MANIFEST_DIR` to find a 'target' directory
/// 4. Falls back to using the compile-time `CARGO_MANIFEST_DIR`
///
/// This helps resolve issues with cargo workspace target directory resolution
/// (see https://github.com/rust-lang/cargo/issues/9661).
///
/// # Returns
///
/// * `Result<PathBuf>` - Path to the cargo target directory if found
///
/// # Errors
///
/// Returns an error if unable to locate the target directory through any method.
///
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_directory() {
        let result = get_target_directory();
        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("target"));
    }
}
