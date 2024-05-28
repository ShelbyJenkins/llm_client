use super::*;
use std::{
    net::TcpStream,
    process::Command,
    thread,
    time::{Duration, Instant},
};

pub const HOST: &str = "localhost";
pub const PORT: &str = "8080";
pub fn server_address() -> String {
    format!("{}:{}", HOST, PORT)
}
const STATUS_CHECK_TIME_MS: u64 = 650;
const STATUS_RETRY_TIMEOUT_MS: u64 = 200;
const START_UP_CHECK_TIME_S: u64 = 30;
const START_UP_RETRY_TIME_S: u64 = 5;

#[derive(PartialEq)]
enum ServerStatus {
    Running,
    RunningRequested,
    Stopped,
}

impl LlamaBackend {
    pub async fn start_server(&mut self) -> Result<Option<ServerProcess>> {
        if test_connection(
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        )
        .await
            == ServerStatus::Running
        {
            println!("Server is already running.");
            {
                if self
                    .check_server_config(3, Duration::from_millis(STATUS_RETRY_TIMEOUT_MS))
                    .await
                    == ServerStatus::RunningRequested
                {
                    return Ok(None);
                } else {
                    ServerProcess::kill_all_servers();
                }
            }
        }

        let mut server_process = ServerProcess::start_server_backend(
            self.model
                .as_ref()
                .expect("model not found")
                .local_model_path
                .as_str(),
            self.threads,
            self.ctx_size,
            self.n_gpu_layers,
            false,
        )
        .await?;

        if test_connection(
            Duration::from_secs(START_UP_CHECK_TIME_S),
            Duration::from_secs(START_UP_RETRY_TIME_S),
        )
        .await
            == ServerStatus::Stopped
        {
            server_process.kill_server_process();
            panic!("Failed to start server")
        }
        println!("Server successfully started.");
        match self
            .check_server_config(5, Duration::from_secs(START_UP_RETRY_TIME_S))
            .await
        {
            ServerStatus::RunningRequested => Ok(Some(server_process)),
            _ => {
                server_process.kill_server_process();
                panic!("Failed to start server with correct model.")
            }
        }
    }

    async fn check_server_config(&self, conn_attempts: u8, retry_time: Duration) -> ServerStatus {
        let mut attempts: u8 = 0;
        let requested_model = self
            .model
            .as_ref()
            .expect("model not found")
            .local_model_path
            .clone();
        while attempts < conn_attempts {
            let running_model = self.get_model_info().await;
            match running_model {
                Ok(running_model) => {
                    if running_model == requested_model.as_str() {
                        return ServerStatus::RunningRequested;
                    } else {
                        println!("error in check_server_config:\n running model: {running_model}\n requested_model: {requested_model}");
                        return ServerStatus::Running;
                    }
                }
                Err(e) => {
                    println!("error in check_server_config:\n{e}");
                    attempts += 1;
                    thread::sleep(retry_time);
                }
            }
        }
        ServerStatus::Stopped
    }

    async fn get_model_info(&self) -> Result<String> {
        let request = LlamaCompletionsRequestArgs::default()
            .prompt("test")
            .n_predict(1u16)
            .build()?;

        let response = self
            .client()
            .completions()
            .create(request)
            .await
            .map_err(|error| anyhow!(format!("get_model_info failed with error: {}.", error)))?;
        Ok(response.model)
    }
}

async fn test_connection(test_time: Duration, retry_time: Duration) -> ServerStatus {
    let start_time = Instant::now();
    let server_status = ServerStatus::Stopped;
    while (Instant::now().duration_since(start_time) < test_time)
        && server_status == ServerStatus::Stopped
    {
        match TcpStream::connect(server::server_address()) {
            Ok(_) => {
                return ServerStatus::Running;
            }
            Err(_) => thread::sleep(retry_time),
        };
    }
    ServerStatus::Stopped
}

pub struct ServerProcess {
    pub process: std::process::Child,
}
impl Drop for ServerProcess {
    fn drop(&mut self) {
        self.kill_server_process();
    }
}
impl ServerProcess {
    pub async fn start_server_backend(
        model_path: &str,
        threads: u16,
        ctx_size: u32,
        n_gpu_layers: u16,
        embedding: bool,
    ) -> Result<Self> {
        let process =
            Self::start_command(model_path, threads, ctx_size, n_gpu_layers, embedding).await;
        println!("Starting server with process PID: {}", process.id());

        Ok(ServerProcess { process })
    }
    pub async fn start_command(
        model_path: &str,
        threads: u16,
        ctx_size: u32,
        n_gpu_layers: u16,
        embedding: bool,
    ) -> std::process::Child {
        let mut command = Command::new("./server");

        command
            .current_dir(get_llama_cpp_path())
            .arg("--n-gpu-layers")
            .arg(n_gpu_layers.to_string())
            .arg("--threads")
            .arg(threads.to_string())
            .arg("--model")
            .arg(model_path)
            .arg("--ctx-size")
            .arg(ctx_size.to_string())
            .arg("--timeout")
            .arg("600")
            .arg("--host")
            .arg(HOST)
            .arg("--port")
            .arg(PORT);

        if embedding {
            command.arg("--embedding");
        }

        command.spawn().expect("Failed to start server")
    }

    pub fn kill_server_process(&mut self) {
        self.process
            .kill()
            .expect("Failed to kill server. This shouldn't ever panic.");

        Self::kill_all_servers();
        thread::sleep(Duration::from_secs(1));
    }
    pub fn kill_all_servers() {
        // pgrep -f '^./server --n-gpu-layers'
        let output = Command::new("pgrep")
            .arg("-f")
            .arg("^./server --n-gpu-layers")
            .output()
            .expect("Failed to execute pgrep");
        let pids = String::from_utf8_lossy(&output.stdout);
        for pid in pids.lines() {
            Command::new("kill")
                .arg(pid)
                .status()
                .expect("Failed to kill process");
        }
        thread::sleep(Duration::from_secs(1));
    }
}

pub fn get_llama_cpp_path() -> String {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let llama_path =
        std::fs::canonicalize(manifest_dir.join(LLAMA_PATH)).expect("Failed to canonicalize path");

    llama_path
        .to_str()
        .expect("Failed to convert path to string")
        .to_string()
}

#[cfg(test)]
mod tests {

    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_server() -> Result<()> {
        let mut backend = LlamaBackend::new();
        backend.setup().await?;
        std::mem::drop(backend);
        if test_connection(
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        )
        .await
            == ServerStatus::Running
        {
            panic!("Server should be stopped");
        }
        let mut backend = LlamaBackend::new();
        backend.setup().await?;
        backend.server_process.unwrap().kill_server_process();

        if test_connection(
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        )
        .await
            == ServerStatus::Running
        {
            panic!("Server should be stopped");
        }
        Ok(())
    }
}
