use anyhow::Result;
use llm_client::llama_cpp::server::ServerProcess;
use std::{env, fs::File, io::Read, path::PathBuf, process::Command};

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
        .subcommand(clap::Command::new("stop").about("Stops the server"))
        .get_matches();

    match matches.subcommand() {
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
