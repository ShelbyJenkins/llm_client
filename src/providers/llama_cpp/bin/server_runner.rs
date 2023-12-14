use llm_client::providers::llama_cpp::server;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

// cargo run -p llm_client --bin server_runner start --model_id TheBloke/zephyr-7B-alpha-GGUF --model_filename zephyr-7b-alpha.Q5_K_M.gguf
// cargo run -p llm_client --bin server_runner start -- --model_id TheBloke/zephyr-7B-alpha-GGUF --model_filename zephyr-7b-alpha.Q2_K.gguf
// cargo run -p llm_client --bin server_runner start TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_0.gguf
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
                    clap::Arg::new("model_id")
                        .help("The model URL")
                        .long("model_id")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("model_filename")
                        .help("The model version")
                        .long("model_filename")
                        .required(true),
                )
                .arg(
                    clap::Arg::new("model_token")
                        .help("HF token")
                        .long("model_token")
                        .required(false),
                ),
        )
        .subcommand(clap::Command::new("stop").about("Stops the server"))
        .get_matches();

    match matches.subcommand() {
        Some(("start", cmd)) => {
            let model_id = cmd.get_one::<String>("model_id").unwrap();
            let model_filename = cmd.get_one::<String>("model_filename").unwrap();
            let model_token = cmd.get_one::<String>("model_token");

            server::kill_existing();
            let mut child =
                server::server_process(model_id, model_filename, model_token.cloned()).await;
            let pid = child.id();
            let mut file = File::create(file_path).expect("Failed to create PID file");
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
