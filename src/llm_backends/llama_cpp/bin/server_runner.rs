use anyhow::Result;
use clap::value_parser;
use llm_client::{llama_cpp::server::ServerProcess, llm_backends::llama_cpp::DEFAULT_N_GPU_LAYERS};
use llm_utils::models::gguf::GGUFModelBuilder;
use std::{
    env,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    process::Command,
    thread,
    time::Duration,
};

// cargo run -p llm_client --bin server_runner start --n_gpu_layers 22 --model_url "https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"

// cargo run -p llm_client --bin server_runner stop

#[tokio::main]
pub async fn main() -> Result<()> {
    let exe_path = env::current_exe().expect("Failed to get the current executable path");

    // Get the directory of the executable
    let exe_dir = exe_path
        .parent()
        .expect("Failed to get the directory of the executable");

    // Create a path for the new file in the same directory
    let mut file_path = PathBuf::from(exe_dir);
    file_path.push("server.pid");

    let matches = clap::Command::new("Server Manager")
        .subcommand(
            clap::Command::new("start")
                .about("Starts the server")
                .arg(
                    clap::Arg::new("model_url")
                        .help("The model URL")
                        .long("model_url")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("hf_token")
                        .help("Your hugging face token")
                        .long("hf_token")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("threads")
                        .value_parser(value_parser!(u16))
                        .help("threads")
                        .long("threads")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("ctx_size")
                        .value_parser(value_parser!(u16))
                        .help("ctx_size")
                        .long("ctx")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("n_gpu_layers")
                        .value_parser(value_parser!(u16))
                        .help("n_gpu_layers")
                        .long("n_gpu_layers")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("embedding")
                        .help("embedding")
                        .long("embedding")
                        .required(false),
                ),
        )
        .subcommand(clap::Command::new("stop").about("Stops the server"))
        .get_matches();

    match matches.subcommand() {
        Some(("start", cmd)) => {
            let model_url = cmd.get_one::<String>("model_url").unwrap();
            let hf_token = cmd
                .get_one::<String>("hf_token")
                .map(|token| token.to_owned());

            let threads = cmd.get_one::<u16>("threads").copied();
            let ctx_size = cmd.get_one::<u32>("ctx_size").copied();
            let n_gpu_layers = cmd.get_one("n_gpu_layers").copied();
            let embedding = cmd.get_one::<bool>("embedding").copied();

            let mut server_process = start_server_cli(
                model_url,
                hf_token,
                threads,
                ctx_size,
                n_gpu_layers,
                embedding,
            )
            .await?;

            let pid = server_process.process.id();
            let mut file: File = File::create(file_path).expect("Failed to create PID file");
            writeln!(file, "{}", pid).expect("Failed to write to PID file");
            loop {
                if let Ok(Some(status)) = server_process.process.try_wait() {
                    println!("Server exited with status: {}", status);
                    break;
                } else {
                    println!("Server is still running...");
                    thread::sleep(Duration::from_secs(1000));
                }
            }
        }
        Some(("stop", _)) => {
            let mut file = File::open(file_path).expect("Failed to open PID file");
            let mut pid_str = String::new();
            file.read_to_string(&mut pid_str)
                .expect("Failed to read PID file");
            let pid: u32 = pid_str.trim().parse().expect("Failed to parse PID");
            Command::new("kill")
                .arg(pid.to_string())
                .status()
                .expect("Failed to kill process");
            ServerProcess::kill_all_servers();
        }
        _ => println!("No valid subcommand was provided."),
    }
    Ok(())
}

async fn start_server_cli(
    model_url: &str,
    hf_token: Option<String>,
    threads: Option<u16>,
    ctx_size: Option<u32>,
    n_gpu_layers: Option<u16>,
    embedding_enabled: Option<bool>,
) -> Result<ServerProcess> {
    let model = GGUFModelBuilder::new(hf_token.clone())
        .from_quant_file_url(model_url)
        .load()
        .await?;
    ServerProcess::start_server_backend(
        &model.local_model_path,
        threads.unwrap_or(1),
        ctx_size.unwrap_or(model.metadata.context_length),
        n_gpu_layers.unwrap_or(DEFAULT_N_GPU_LAYERS),
        embedding_enabled.unwrap_or(false),
    )
    .await
}
