pub(super) fn source(repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    let url = format!("{repo_url}/archive/refs/tags/{repo_tag}.zip");
    download(&url)
}

pub(super) fn binary(repo_url: &str, repo_tag: &str) -> crate::Result<()> {
    let url = if cfg!(target_os = "linux") {
        format!("{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-ubuntu-x64.zip")
    } else if cfg!(target_os = "macos") {
        match std::env::consts::ARCH {
            "aarch64" => format!(
                "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-macos-arm64.zip"
            ),
            "x86_64" => format!(
                "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-macos-x64.zip"
            ),
            arch => panic!("Unsupported architecture on macOS: {}", arch),
        }
    } else if cfg!(target_os = "windows") {
        format!(
            "{repo_url}/releases/download/{repo_tag}/llama-{repo_tag}-bin-win-cuda-cu12.4-x64.zip"
        )
    } else {
        panic!("Unsupported operating system")
    };
    download(&url)
}

fn download(url: &str) -> crate::Result<()> {
    std::fs::create_dir_all(super::get_target_dir_sub_dir()?)
        .map_err(|e| crate::anyhow!("Failed to create installer directory: {}", e))?;

    let zip_path = super::zip::get_zip_file_path()?;
    crate::trace!("Downloading from {} to {}", url, zip_path.display());
    match std::process::Command::new("curl")
        .args([
            "-sL",
            "-o",
            super::zip::get_zip_file_path()?.to_str().unwrap(),
            &url,
        ])
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                crate::bail!("Download failed\nstderr: {}\n stdout: {}", stderr, stdout);
            }
            crate::trace!("Download completed successfully");
            Ok(())
        }
        Err(e) => {
            panic!("Download failed with error: {}", e);
        }
    }
}
