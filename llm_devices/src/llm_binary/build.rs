use crate::CpuConfig;
use std::{path::PathBuf, process::Command};

pub const BUILD_ARGS: [&str; 1] = ["-DLLAMA_LLGUIDANCE=ON"];
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub const CUDA_ARGS: [&str; 1] = ["-DGGML_CUDA=ON"];

pub fn cmake_is_available() -> bool {
    match Command::new("cmake").arg("--version").output() {
        Ok(output) => {
            if output.status.success() {
                crate::trace!("Cmake found: {}", String::from_utf8_lossy(&output.stdout));
                return true;
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            crate::trace!(
                "Cmake check failed:\nstderr: {}\n stdout: {}",
                stderr,
                stdout
            );
            return false;
        }
        Err(e) => {
            crate::trace!("Failed to check cmake version: {}", e);
            return false;
        }
    }
}

pub(super) fn cmake_project_buildsystem(local_repo_path: &PathBuf) -> Result<(), crate::Error> {
    let mut builder = Command::new("cmake");
    builder.arg("-B").arg("build").args(BUILD_ARGS);

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        match crate::init_nvml_wrapper() {
            Ok(_) => {
                builder.args(CUDA_ARGS);
            }
            Err(_) => {
                crate::trace!("No CUDA detected - building without CUDA support");
            }
        }
    }

    builder.current_dir(local_repo_path);
    crate::trace!("Running cmake command: {:?}", builder);
    match builder.output() {
        Ok(output) => {
            if !output.status.success() {
                crate::bail!(
                    "Cmake command failed with exit code: {}\nStderr: {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            } else {
                crate::trace!("Cmake command completed successfully");
            }
        }
        Err(e) => crate::bail!("Failed to execute cmake command: {}", e),
    };
    Ok(())
}

pub(super) fn cmake_build_project(local_repo_path: &PathBuf) -> Result<(), crate::Error> {
    let num_jobs = CpuConfig::default().num_cpus().saturating_sub(1);
    let mut builder = Command::new("cmake");
    builder
        .arg("--build")
        .arg("build")
        .arg("--config")
        .arg("Release")
        .arg("-j")
        .arg(num_jobs.to_string())
        .arg("-t")
        .arg(super::binary::LLAMA_CPP_SERVER_EXECUTABLE)
        .current_dir(local_repo_path);

    crate::trace!("Running cmake command: {:?}", builder);
    match builder.output() {
        Ok(output) => {
            if !output.status.success() {
                crate::bail!(
                    "Cmake command failed with exit code: {}\nStderr: {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            } else {
                crate::trace!("Cmake command completed successfully");
            }
        }
        Err(e) => crate::bail!("Failed to execute cmake command: {}", e),
    };
    Ok(())
}
