use super::get_model_info;
use super::model_loader::{check_requested_model_against_given_model, download_model};
use super::models::LlamaDef;
use core::panic;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const HOST: &str = "localhost";
pub const PORT: &str = "8080";
const SERVER_RETRIES: u8 = 3;
const SERVER_TIMEOUT_MS: u64 = 200;

lazy_static! {
    #[derive(Debug)]
    pub static ref SERVER_ADDRESS: String = format!("{}:{}", HOST, PORT);
}

enum ServerStatus {
    RunningRequested,
    RunningOther,
    Stopped,
}

pub async fn start_server_cli(
    model_url: &str,
    model_token: Option<String>,
    threads: Option<u16>,
    ctx_size: Option<u16>,
    n_gpu_layers: Option<u16>,
) -> std::process::Child {
    let def = LlamaDef::new(
        model_url,
        super::models::LlamaPromptFormat::None,
        threads,
        ctx_size,
        n_gpu_layers,
    );

    kill_server();

    start_server(
        &def.model_id,
        &def.model_filename,
        model_token,
        def.threads,
        def.max_tokens_for_model,
        def.n_gpu_layers,
    )
    .await
}

pub async fn check_and_or_start_server(
    llama_def: &LlamaDef,
) -> Result<(), Box<dyn std::error::Error>> {
    match check_server_connection_and_model(llama_def, Duration::from_millis(650)).await {
        ServerStatus::RunningRequested => return Ok(()),
        // Ok(ServerStatus::RunningRequested) => {}
        ServerStatus::RunningOther => {}
        ServerStatus::Stopped => {}
    }
    start_server(
        &llama_def.model_id,
        &llama_def.model_filename,
        None,
        llama_def.threads,
        llama_def.max_tokens_for_model,
        llama_def.n_gpu_layers,
    )
    .await;
    match check_server_connection_and_model(llama_def, Duration::from_millis(650)).await {
        ServerStatus::RunningRequested => Ok(()),
        // Ok(ServerStatus::RunningRequested) => {}
        _ => panic!("Failed to start server"),
    }
}

async fn check_server_connection_and_model(
    llama_def: &LlamaDef,
    timeout: Duration,
) -> ServerStatus {
    let start_time = Instant::now();
    let mut status: bool = false;
    while (Instant::now().duration_since(start_time) < timeout) && !status {
        match TcpStream::connect(SERVER_ADDRESS.clone()) {
            Ok(_) => status = true,
            Err(_) => thread::sleep(Duration::from_millis(SERVER_TIMEOUT_MS)),
        };
    }
    if status {
        let mut attempts: u8 = 0;

        while attempts < SERVER_RETRIES {
            match get_model_info().await {
                Ok(model) => {
                    if check_requested_model_against_given_model(
                        &llama_def.model_id,
                        &llama_def.model_filename,
                        PathBuf::from(model),
                    )
                    .await
                    {
                        return ServerStatus::RunningRequested;
                    } else {
                        return ServerStatus::RunningOther;
                    }
                }
                Err(_) => {
                    attempts += 1;
                    thread::sleep(Duration::from_millis(SERVER_TIMEOUT_MS));
                }
            }
        }
    }
    ServerStatus::Stopped
}

pub async fn start_server(
    model_id: &str,
    model_filename: &str,
    model_token: Option<String>,
    threads: u16,
    ctx_size: u16,
    n_gpu_layers: u16,
) -> std::process::Child {
    let model_path = download_model(model_id, model_filename, model_token).await;

    let model_path = match model_path {
        Some(model_path) => model_path.to_str().unwrap().to_owned(),
        None => panic!("Failed to download model"),
    };
    let server_process = server_process(&model_path, threads, ctx_size, n_gpu_layers).await;

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

pub fn kill_server() {
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
    threads: u16,
    ctx_size: u16,
    n_gpu_layers: u16,
) -> std::process::Child {
    kill_server();

    // Start the server
    Command::new("./server")
        .current_dir(super::get_llama_cpp_path())
        .arg("--threads")
        .arg(threads.to_string()) //12
        .arg("--model")
        .arg(model_path)
        .arg("--ctx-size")
        .arg(ctx_size.to_string()) // 9001
        .arg("--n-gpu-layers")
        .arg(n_gpu_layers.to_string()) // 23
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

#[cfg(test)]
mod tests {

    use super::{check_and_or_start_server, kill_server};
    use crate::providers::llama_cpp::models::{
        LlamaDef, DEFAULT_CTX_SIZE, DEFAULT_N_GPU_LAYERS, DEFAULT_THREADS, TEST_LLM_URL_1,
        TEST_LLM_URL_2, TEST_PROMPT_TEMPLATE_1, TEST_PROMPT_TEMPLATE_2,
    };

    #[tokio::test]
    async fn test_server() -> Result<(), Box<dyn std::error::Error>> {
        kill_server();
        let llama_def = LlamaDef::new(
            TEST_LLM_URL_1,
            TEST_PROMPT_TEMPLATE_1,
            Some(DEFAULT_CTX_SIZE),
            Some(DEFAULT_THREADS),
            Some(DEFAULT_N_GPU_LAYERS),
        );
        check_and_or_start_server(&llama_def).await?;

        let llama_def = LlamaDef::new(
            TEST_LLM_URL_2,
            TEST_PROMPT_TEMPLATE_2,
            Some(DEFAULT_CTX_SIZE),
            Some(DEFAULT_THREADS),
            Some(DEFAULT_N_GPU_LAYERS),
        );
        check_and_or_start_server(&llama_def).await?;
        kill_server();
        Ok(())
    }
}
