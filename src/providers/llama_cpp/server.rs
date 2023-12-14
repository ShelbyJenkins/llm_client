use super::model_loader::download_model;
use core::panic;
use std::fs;
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
pub const HOST: &str = "localhost";
pub const PORT: &str = "8080";
const LLAMA_PATH: &str = "/workspaces/test/llm_client/src/providers/llama_cpp/llama_cpp";

pub async fn start_server(
    model_id: &str,
    model_filename: &str,
    model_token: Option<String>,
) -> std::process::Child {
    let server_process = server_process(model_id, model_filename, model_token).await;
    let server_addr = format!("{}:{}", HOST, PORT);
    let timeout = Duration::from_secs(30);
    let start_time = Instant::now();
    println!("Starting server with process PID: {}", server_process.id());
    while Instant::now().duration_since(start_time) < timeout {
        match TcpStream::connect(server_addr.clone()) {
            Ok(_) => {
                println!("Server is ready.");
                break;
            }
            Err(_) => {
                thread::sleep(Duration::from_secs(1));
            }
        }
    }
    if Instant::now().duration_since(start_time) >= timeout {
        panic!("Timeout reached. Server did not start in time.");
    }
    server_process
}

pub fn test_server() {
    let server_addr = format!("{}:{}", HOST, PORT);
    let timeout = Duration::from_secs(5);
    let start_time = Instant::now();
    while Instant::now().duration_since(start_time) < timeout {
        match TcpStream::connect(server_addr.clone()) {
            Ok(_) => {
                println!("Server tested. Server is ready to serve.");
                break;
            }
            Err(_) => {
                thread::sleep(Duration::from_secs(1));
            }
        }
    }
    if Instant::now().duration_since(start_time) >= timeout {
        panic!("Timeout reached. Server is not running. Please start the server with llm_cpp::start_server.");
    }
}

pub fn kill_existing() {
    // Run nvidia-smi to get all GPU-using PIDs
    let output = Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()
        .expect("Failed to execute nvidia-smi");

    let pids = String::from_utf8_lossy(&output.stdout);

    for pid in pids.lines() {
        let _ = Command::new("kill").arg(pid).status();
    }
}

// c N, --ctx-size N: Set the size of the prompt context.
pub async fn server_process(
    model_id: &str,
    model_filename: &str,
    model_token: Option<String>,
) -> std::process::Child {
    kill_existing();
    let llama_path = fs::canonicalize(LLAMA_PATH).expect("Failed to canonicalize path");

    let llama_path_str = llama_path
        .to_str()
        .expect("Failed to convert path to string");

    // let model_path = format!("{}/{}/{}", MODEL_PATH, model_id, model_file_name);

    let path = download_model(model_id, model_filename, model_token).await;
    let path = match path {
        Some(path) => path.to_str().unwrap().to_string(),
        None => panic!("Failed to download model"),
    };

    // Start the server
    Command::new("./server")
        .current_dir(llama_path_str)
        .arg("--threads")
        .arg("12")
        .arg("--model")
        .arg(path)
        .arg("--ctx-size")
        .arg("9001")
        .arg("--n-gpu-layers")
        .arg("23")
        .arg("--host")
        .arg(HOST)
        .arg("--port")
        .arg(PORT)
        .arg("--timeout")
        .arg("60")
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start server")
}
