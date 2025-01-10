use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(any(target_os = "linux", target_os = "windows"))]
use crate::devices::cuda::init_nvml_wrapper;

pub(super) fn local_repo_requires_build(local_repo_path: &Path, executable_name: &str) -> bool {
    let executable = local_repo_path.join(executable_name);
    if !executable.exists() || !executable.is_file() {
        crate::trace!(
            "Did not find {executable_name} with path: {}",
            executable.display()
        );
        true
    } else {
        crate::trace!(
            "Found {executable_name} with path: {}",
            executable.display()
        );
        false
    }
}

#[allow(unused_variables)]
pub(super) fn build_local_repo(
    local_repo_path: &PathBuf,
    builder_args: &[&str],
    cuda_arg: &Option<&str>,
) -> crate::Result<()> {
    let mut builder = Command::new("make");
    builder
        .args(builder_args)
        // .args(["llama-server", "BUILD_TYPE=Release", "-j"])
        .current_dir(local_repo_path);

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    handle_cuda_arg(&mut builder, cuda_arg);

    crate::trace!("Running make command: {:?}", builder);
    let output = builder
        .output()
        .map_err(|e| crate::anyhow!("Failed to execute make command: {}", e))?;

    if output.status.success() {
        crate::trace!("Make command completed successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        crate::bail!(
            "Make command failed with exit code: {}\nStderr: {}",
            output.status,
            stderr
        )
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn handle_cuda_arg(builder: &mut Command, cuda_arg: &Option<&str>) {
    if let Some(cuda_arg) = cuda_arg {
        match init_nvml_wrapper() {
            Ok(_) => {
                builder.arg(cuda_arg);
                // builder.arg("GGML_CUDA=1");
            }
            Err(_) => {
                crate::trace!("No CUDA detected - building without CUDA support");
            }
        }
    }
}
