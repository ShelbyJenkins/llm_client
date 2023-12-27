use llm_client::providers::llama_cpp::server;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

// cargo run -p llm_client --bin server_runner start --model_url "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q8_0.gguf"
// cargo run -p llm_client --bin server_runner stop

#[tokio::main]
pub async fn main() {
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
                    clap::Arg::new("model_token")
                        .help("HF token")
                        .long("model_token")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("threads")
                        .help("threads")
                        .long("threads")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("ctx_size")
                        .help("ctx_size")
                        .long("ctx")
                        .required(false),
                )
                .arg(
                    clap::Arg::new("n_gpu_layers")
                        .help("n_gpu_layers")
                        .long("n_gpu_layers")
                        .required(false),
                ),
        )
        .subcommand(clap::Command::new("stop").about("Stops the server"))
        .get_matches();

    match matches.subcommand() {
        Some(("start", cmd)) => {
            let model_url = cmd.get_one::<String>("model_url").unwrap();
            let model_token = cmd.get_one::<String>("model_token");

            let threads = cmd.get_one::<u16>("threads").copied();
            let ctx_size = cmd.get_one::<u16>("ctx_size").copied();
            let n_gpu_layers = cmd.get_one::<u16>("n_gpu_layers").copied();

            let mut child = server::start_server_cli(
                model_url,
                model_token.cloned(),
                threads,
                ctx_size,
                n_gpu_layers,
            )
            .await;

            let pid = child.id();
            let mut file: File = File::create(file_path).expect("Failed to create PID file");
            writeln!(file, "{}", pid).expect("Failed to write to PID file");
            loop {
                if let Ok(Some(status)) = child.try_wait() {
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
        }
        _ => println!("No valid subcommand was provided."),
    }
}
