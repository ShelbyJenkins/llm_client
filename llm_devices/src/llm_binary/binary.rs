#[cfg(target_os = "windows")]
pub const LLAMA_CPP_SERVER_EXECUTABLE: &str = "llama-server.exe";
#[cfg(target_os = "linux")]
pub const LLAMA_CPP_SERVER_EXECUTABLE: &str = "llama-server";
#[cfg(target_os = "macos")]
pub const LLAMA_CPP_SERVER_EXECUTABLE: &str = "llama-server";

pub fn get_bin_dir() -> crate::Result<std::path::PathBuf> {
    find_binary_dir(&super::get_target_dir_sub_dir()?)
}

pub fn get_bin_path() -> crate::Result<std::path::PathBuf> {
    Ok(get_bin_dir()?.join(LLAMA_CPP_SERVER_EXECUTABLE))
}

pub(super) fn get_version_path() -> crate::Result<std::path::PathBuf> {
    Ok(get_bin_dir()?.join("version"))
}

pub(super) fn get_version() -> crate::Result<String> {
    std::fs::read_to_string(get_version_path()?)
        .map_err(|e| crate::anyhow!("Failed to read version file: {}", e))
        .map(|s| s.trim().to_string())
}

pub fn executable_is_ok(repo_tag: &str) -> crate::Result<bool> {
    match get_version() {
        Ok(version) => {
            if version == repo_tag {
                let binary_path = get_bin_path()?;
                if !binary_path.exists() {
                    crate::trace!("binary_path does not exist: {}", binary_path.display());
                    return Ok(false);
                } else {
                    return Ok(true);
                }
            } else {
                Ok(false)
            }
        }
        Err(e) => {
            crate::trace!("{}", e);
            Ok(false)
        }
    }
}

pub(super) fn set_binary(repo_tag: &str) -> crate::Result<()> {
    let version_file = get_bin_dir()?.join("version");
    std::fs::write(&version_file, repo_tag)
        .map_err(|e| crate::anyhow!("Failed to write version file: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(get_bin_path()?)?.permissions();
        perms.set_mode(0o755); // rwxr-xr-x
        std::fs::set_permissions(get_bin_path()?, perms)?;
    }

    Ok(())
}

fn find_binary_dir(dir: &std::path::PathBuf) -> crate::Result<std::path::PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map_or(false, |n| n == LLAMA_CPP_SERVER_EXECUTABLE)
        {
            return Ok(dir.to_owned());
        } else if path.is_dir() {
            // Recursively search subdirectories
            if let Ok(found) = find_binary_dir(&path) {
                return Ok(found);
            }
        }
    }
    let zip_path = super::zip::get_zip_file_path()?;
    crate::bail!(
        "Could not find {} in {}",
        LLAMA_CPP_SERVER_EXECUTABLE,
        zip_path.display()
    )
}
