use cargo_metadata::MetadataCommand;

use std::{env, path::PathBuf, process::Command};

macro_rules! p {
    ($($arg:tt)*) => {
        println!("cargo:warning={}", format!($($arg)*));
    }
}

fn main() {
    let start_time = std::time::Instant::now();
    println!("cargo:rerun-if-changed=build.rs");
    if cfg!(feature = "llama_cpp_backend") {
        let llama_cpp_dir = get_target_dir().join("llama_cpp");
        // println!("cargo::rerun-if-changed={}", llama_cpp_dir.display());

        p!("Starting build process...");
        if local_repo_requires_build() {
            build(&llama_cpp_dir);
        };
    }
    p!(
        "Build process completed in {} seconds",
        start_time.elapsed().as_secs_f32()
    );
}

fn get_package_metadata() -> cargo_metadata::Package {
    let metadata = MetadataCommand::new().exec().unwrap();
    metadata
        .packages
        .iter()
        .find(|p| p.name == env!("CARGO_PKG_NAME"))
        .unwrap()
        .to_owned()
}

fn get_target_dir() -> PathBuf {
    // Hack to resolve this cargo issue
    // https://github.com/rust-lang/cargo/issues/9661
    let start_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    start_dir
        .ancestors()
        .find(|path| {
            // Check if this path's directory name is 'target'
            if let Some(dir_name) = path.file_name() {
                dir_name == "target"
            } else {
                false
            }
        })
        .expect("Could not find 'target' directory")
        .to_owned()
}

fn local_repo_requires_build() -> bool {
    let llama_cpp_dir = get_target_dir().join("llama_cpp");
    p!("Starting llama.cpp repo setup...");
    let package = get_package_metadata();
    let llama_cpp_repo = package.metadata["llama_cpp_backend"]["repo"]
        .as_str()
        .unwrap();
    let tag = package.metadata["llama_cpp_backend"]["tag"]
        .as_str()
        .unwrap();

    p!(
        "Extracted repo info - URL: {}, Revision: {}",
        llama_cpp_repo,
        tag
    );

    if local_repo_requires_update(&llama_cpp_dir, llama_cpp_repo, tag) {
        Command::new("git")
            .arg("clone")
            .arg("--depth=1") // Shallow clone to save bandwidth and time
            .arg(format!("--branch={}", tag))
            .arg("--recursive")
            .arg(llama_cpp_repo)
            .arg(&llama_cpp_dir)
            .status()
            .unwrap();
        p!("Successfully cloned llama.cpp repo at tag {}", tag);
        true
    } else {
        false
    }
}

fn local_repo_requires_update(llama_cpp_dir: &PathBuf, llama_cpp_repo: &str, tag: &str) -> bool {
    if llama_cpp_dir.exists() {
        p!("Directory exists: {} ", llama_cpp_dir.display());
        // Check if it's a git repository
        let is_git_repo = Command::new("git")
            .current_dir(llama_cpp_dir)
            .arg("rev-parse")
            .arg("--is-inside-work-tree")
            .status()
            .unwrap()
            .success();
        if is_git_repo {
            p!("Directory is git repo: {} ", llama_cpp_dir.display());
            // Check if it's the correct repository
            let remote_url = String::from_utf8(
                Command::new("git")
                    .current_dir(llama_cpp_dir)
                    .args(["config", "--get", "remote.origin.url"])
                    .output()
                    .unwrap()
                    .stdout,
            )
            .unwrap();
            if remote_url.trim() == llama_cpp_repo {
                p!("{} == {} ", remote_url.trim(), llama_cpp_repo);
                // Fetch the latest tags
                Command::new("git")
                    .current_dir(llama_cpp_dir)
                    .args(["fetch", "--tags"])
                    .status()
                    .unwrap();

                // Check if the current HEAD is at the specified tag
                let is_at_tag = Command::new("git")
                    .current_dir(llama_cpp_dir)
                    .args(["describe", "--tags", "--exact-match"])
                    .output()
                    .map(|output| String::from_utf8_lossy(&output.stdout).trim() == tag)
                    .unwrap_or(false);

                if is_at_tag {
                    p!("{tag} == local repo tag - no update required");
                    return false;
                }
            }
        }
        // If any check fails, remove the directory to prepare for a fresh clone
        std::fs::remove_dir_all(llama_cpp_dir).unwrap();
        p!("Removed {}", llama_cpp_dir.display());
    }
    true
}

fn build(llama_cpp_dir: &PathBuf) {
    let mut builder = Command::new("make");
    builder
        .args(["llama-server", "BUILD_TYPE=Release", "-j"])
        .current_dir(llama_cpp_dir);

    if cfg!(not(target_os = "macos")) {
        match init_nvml_wrapper() {
            Ok(_) => {
                builder.arg("GGML_CUDA=1");
            }
            Err(_) => {
                p!("No CUDA detected - building without CUDA support");
            }
        }
    }

    p!("Running make command: {:?}", builder);
    let status = builder.status().unwrap();
    if status.success() {
        p!("Make command completed successfully");
    } else {
        // On failure, remove the directory to prepare for a fresh clone
        std::fs::remove_dir_all(llama_cpp_dir).unwrap();
        p!("Removed {}", llama_cpp_dir.display());
        panic!("Make command failed with exit code: {}", status);
    }
}

#[cfg(not(target_os = "macos"))]
fn init_nvml_wrapper() -> anyhow::Result<nvml_wrapper::Nvml> {
    let library_names = vec![
        "libnvidia-ml.so",   // For Linux
        "libnvidia-ml.so.1", // For WSL
        "nvml.dll",          // For Windows
    ];
    for library_name in library_names {
        match nvml_wrapper::Nvml::builder()
            .lib_path(library_name.as_ref())
            .init()
        {
            Ok(nvml) => return Ok(nvml),
            Err(_) => {
                continue;
            }
        }
    }
    anyhow::bail!("Failed to initialize nvml_wrapper::Nvml")
}
