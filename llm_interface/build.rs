use cargo_metadata::MetadataCommand;

macro_rules! p {
    ($($arg:tt)*) => {
        println!("cargo:warning={}", format!($($arg)*))
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Skip building when building docs
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    if cfg!(feature = "llama_cpp_backend") {
        let start_time = std::time::Instant::now();
        let package = get_package_metadata();
        let repo_url = package.metadata["llama_cpp_backend"]["repo"]
            .as_str()
            .expect("Excpected llama_cpp_backend.repo as a string");
        let repo_tag = package.metadata["llama_cpp_backend"]["tag"]
            .as_str()
            .expect("Excpected llama_cpp_backend.tag as a string");

        match llm_devices::build_or_install(true, repo_url, repo_tag) {
            Ok(_) => p!(
                "Successfully built llama_cpp in {} seconds",
                start_time.elapsed().as_secs_f32()
            ),
            Err(e) => {
                p!(
                    "Failed to build llama_cpp: {} in {} seconds",
                    e,
                    start_time.elapsed().as_secs_f32()
                );
                std::process::exit(1);
            }
        }
    }
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
