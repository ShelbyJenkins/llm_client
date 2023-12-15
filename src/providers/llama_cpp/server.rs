use super::model_loader::download_model;
use core::panic;
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
pub const HOST: &str = "localhost";
pub const PORT: &str = "8080";

pub async fn start_server(
    model_id: &str,
    model_filename: &str,
    model_token: Option<String>,
    threads: Option<u16>,
    ctx_size: Option<u16>,
    n_gpu_layers: Option<u16>,
) -> std::process::Child {
    let threads = threads.unwrap_or(2).to_string();
    let ctx_size = ctx_size.unwrap_or(9001).to_string();
    let n_gpu_layers = n_gpu_layers.unwrap_or(6).to_string();

    let model_path = download_model(model_id, model_filename, model_token).await;

    let model_path = match model_path {
        Some(model_path) => model_path.to_str().unwrap().to_owned(),
        None => panic!("Failed to download model"),
    };
    let server_process = server_process(&model_path, &threads, &ctx_size, &n_gpu_layers).await;

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

pub async fn server_process(
    model_path: &str,
    threads: &str,
    ctx_size: &str,
    n_gpu_layers: &str,
) -> std::process::Child {
    kill_existing();

    // Start the server
    Command::new("./server")
        .current_dir(super::get_llama_cpp_path())
        .arg("--threads")
        .arg(threads) //12
        .arg("--model")
        .arg(model_path)
        .arg("--ctx-size")
        .arg(ctx_size) // 9001
        .arg("--n-gpu-layers")
        .arg(n_gpu_layers) // 23
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
