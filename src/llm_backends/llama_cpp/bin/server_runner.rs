use anyhow::Result;
use llm_client::llm_backends::llama_cpp::server::kill_all_servers;
use std::{env, fs::File, io::Read, path::PathBuf, process::Command};

// cargo run -p llm_client --bin server_runner stop

pub fn main() -> Result<()> {
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
            match kill_processes_from_file(&file_path) {
                Ok(_) => println!("Successfully killed the process from file."),
                Err(e) => eprintln!("An error occurred killing process from file: {}", e),
            }
            kill_all_servers();
        }
        _ => println!("No valid subcommand was provided."),
    }
    Ok(())
}

fn kill_processes_from_file(file_path: &PathBuf) -> Result<()> {
    let mut file = File::open(file_path)?;
    let mut pid_str = String::new();
    file.read_to_string(&mut pid_str)?;
    let pid: u32 = pid_str.trim().parse()?;
    Command::new("kill").arg(pid.to_string()).status()?;
    Ok(())
}
