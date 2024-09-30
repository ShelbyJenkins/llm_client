use super::{
    completion::{LlamaCppCompletionRequest, LlamaCppCompletionResponse},
    devices::LlamaCppDeviceMap,
    LlamaCppConfig,
};
use crate::llms::{api::client::ApiClient, local::LocalLlmConfig};

const STATUS_CHECK_TIME_MS: u64 = 650;
const STATUS_RETRY_TIMEOUT_MS: u64 = 200;
const START_UP_CHECK_TIME_S: u64 = 30;
const START_UP_RETRY_TIME_S: u64 = 5;

/// Hack to resolve this cargo issue
/// https://github.com/rust-lang/cargo/issues/9661
fn get_llama_cpp_path() -> crate::Result<std::path::PathBuf> {
    let start_dir = std::path::PathBuf::from(env!("OUT_DIR"));
    let target_dir = start_dir
        .ancestors()
        .find(|path| {
            // Check if this path's directory name is 'target'
            if let Some(dir_name) = path.file_name() {
                dir_name == "target"
            } else {
                false
            }
        })
        .ok_or(crate::anyhow!("Could not find llama_cpp path"))?;
    let path = target_dir.join("llama_cpp");
    crate::info!("llama_cpp_dir: {}", path.display());
    Ok(path)
}

#[derive(PartialEq)]
pub enum ServerStatus {
    Running,
    RunningRequested,
    Stopped,
}

pub struct LlamaCppServer {
    pub local_config: LocalLlmConfig,
    pub(crate) server_config: LlamaCppDeviceMap,
    pub server_process: Option<std::process::Child>,
    pub host: String,
    pub path: String,
    pub port: Option<String>,
}

impl LlamaCppServer {
    pub fn new(config: &LlamaCppConfig, local_config: LocalLlmConfig) -> crate::Result<Self> {
        let path = if let Some(port) = &config.api_config.port {
            format!("{}:{}", config.api_config.host, port)
        } else {
            config.api_config.host.clone()
        };
        Ok(Self {
            server_process: None,
            server_config: LlamaCppDeviceMap::new(&local_config.device_config)?,
            local_config,
            path,
            host: config.api_config.host.clone(),
            port: config.api_config.port.clone(),
        })
    }

    pub(crate) async fn start_server(
        &mut self,
        client: &ApiClient<LlamaCppConfig>,
    ) -> crate::Result<ServerStatus> {
        match self
            .connect_with_timeouts(
                std::time::Duration::from_millis(STATUS_CHECK_TIME_MS),
                std::time::Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
                client,
            )
            .await?
        {
            ServerStatus::RunningRequested => return Ok(ServerStatus::RunningRequested),
            ServerStatus::Stopped => (),
            ServerStatus::Running => self.shutdown(),
        };

        let original = if !self.local_config.device_config.use_gpu {
            let original = std::env::var("CUDA_VISIBLE_DEVICES").ok();
            std::env::set_var("CUDA_VISIBLE_DEVICES", "");
            original
        } else {
            None
        };

        self.server_process = Some(self.start_server_backend()?);

        match self
            .connect_with_timeouts(
                std::time::Duration::from_secs(START_UP_CHECK_TIME_S),
                std::time::Duration::from_secs(START_UP_RETRY_TIME_S),
                client,
            )
            .await?
        {
            ServerStatus::RunningRequested => {
                if !self.local_config.device_config.use_gpu {
                    match original {
                        Some(value) => std::env::set_var("CUDA_VISIBLE_DEVICES", value),
                        None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
                    }
                }
                crate::trace!(
                    "Started server with process PID: {}",
                    self.server_process
                        .as_ref()
                        .expect("Server process not created")
                        .id()
                );
                Ok(ServerStatus::RunningRequested)
            }
            ServerStatus::Stopped => {
                self.shutdown();
                tracing::error!("Failed to start server");
                panic!("Failed to start server")
            }
            ServerStatus::Running => {
                self.shutdown();
                tracing::error!("Failed to start server with correct model.");
                panic!("Failed to start server with correct model.")
            }
        }
    }

    fn start_server_backend(&self) -> crate::Result<std::process::Child> {
        let path = get_llama_cpp_path()?;
        let mut command = std::process::Command::new("./llama-server");
        command.current_dir(path);
        self.server_config.populate_args(&mut command);
        command
            .arg("--model")
            .arg(&self.local_config.device_config.local_model_path)
            .arg("--ctx-size")
            .arg(self.local_config.inference_ctx_size.to_string())
            .arg("--timeout")
            .arg("600")
            .arg("--host")
            .arg(&self.host)
            // .arg("--log-disable")
            .arg("--verbose");

        if let Some(port) = &self.port {
            command.arg("--port").arg(port);
        }
        crate::trace!("Starting server with command: {:?}", command);
        let process = command.spawn().expect("Failed to start server");

        Ok(process)
    }

    async fn connect_with_timeouts(
        &self,
        test_duration: std::time::Duration,
        retry_timeout: std::time::Duration,
        client: &ApiClient<LlamaCppConfig>,
    ) -> crate::Result<ServerStatus> {
        if self.test_connection(test_duration, retry_timeout) == ServerStatus::Running {
            tracing::info!("Server is running.");

            match self.check_server_config(3, retry_timeout, client).await {
                Ok(ServerStatus::RunningRequested) => {
                    return Ok(ServerStatus::RunningRequested);
                }
                Ok(ServerStatus::Running) => {
                    return Ok(ServerStatus::Running);
                }
                Ok(ServerStatus::Stopped) => {
                    return Ok(ServerStatus::Stopped);
                }
                Err(_) => {
                    return Ok(ServerStatus::Stopped);
                }
            }
        } else {
            Ok(ServerStatus::Stopped)
        }
    }

    fn test_connection(
        &self,
        test_time: std::time::Duration,
        retry_time: std::time::Duration,
    ) -> ServerStatus {
        let start_time = std::time::Instant::now();

        while std::time::Instant::now().duration_since(start_time) < test_time {
            match std::net::TcpStream::connect(&self.path) {
                Ok(_) => {
                    return ServerStatus::Running;
                }
                Err(_) => std::thread::sleep(retry_time),
            };
        }
        ServerStatus::Stopped
    }

    async fn check_server_config(
        &self,
        conn_attempts: u8,
        retry_time: std::time::Duration,
        client: &ApiClient<LlamaCppConfig>,
    ) -> crate::Result<ServerStatus> {
        let mut attempts: u8 = 0;
        while attempts < conn_attempts {
            let request = LlamaCppCompletionRequest {
                prompt: vec![0u32],
                n_predict: Some(0),
                ..Default::default()
            };
            let result: Result<LlamaCppCompletionResponse, crate::llms::api::error::ClientError> =
                client.post("/completion", request).await;
            match result {
                Ok(res) => {
                    if &self.local_config.device_config.local_model_path == &res.model {
                        return Ok(ServerStatus::RunningRequested);
                    } else {
                        tracing::info!(
                       "error in check_server_config:\n running model: {}\n requested_model: {:?}", res.model, &self.local_config.device_config.local_model_path
                        );
                        return Ok(ServerStatus::Running);
                    }
                }
                Err(e) => {
                    tracing::info!("error in check_server_config:\n{e}");
                    attempts += 1;
                    std::thread::sleep(retry_time);
                }
            }
        }
        Ok(ServerStatus::Stopped)
    }

    pub fn shutdown(&self) {
        let process = if let Some(server_process) = &self.server_process {
            server_process
        } else {
            kill_all_servers();
            std::thread::sleep(std::time::Duration::from_millis(100));
            return;
        };

        let pid = process.id();
        match kill_server_process_command(pid) {
            Ok(_) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                crate::warn!("Failed to kill server process: {}", e);
            }
        };

        match server_pid_exists(pid) {
            Ok(true) => {
                crate::error!("Failed to kill server process");
                kill_all_servers();
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Ok(false) => (),
            Err(e) => {
                crate::warn!("Failed to check if server process exists: {e}");
            }
        }
    }
}

pub fn kill_server_process_command(pid: u32) -> crate::Result<()> {
    match std::process::Command::new("kill")
        .arg(pid.to_string())
        .status()
    {
        Ok(_) => Ok(()),
        Err(e) => {
            crate::bail!(
                "std::process::Command::new(\"kill\") failed to kill server process: {}",
                e
            )
        }
    }
}

pub fn server_pid_exists(pid: u32) -> crate::Result<bool> {
    let pid: String = pid.to_string();
    match get_all_server_pids() {
        Ok(pids) => {
            for p in pids {
                if p.contains(&pid) {
                    return Ok(true);
                }
            }
        }
        Err(e) => {
            crate::warn!("Failed to check if server process exists: {e}");
        }
    };
    match get_all_server_pids() {
        Ok(pids) => {
            for p in pids {
                if p.contains(&pid) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Err(e) => {
            crate::bail!("Failed to check if server process exists: {}", e)
        }
    }
}

pub fn get_all_server_pids() -> crate::Result<Vec<String>> {
    // pgrep -f '^./llama-server'
    let output = std::process::Command::new("pgrep")
        .arg("-f")
        .arg("^./llama-server")
        .output()?;
    let pids = String::from_utf8_lossy(&output.stdout);
    let mut pid_vec = Vec::new();
    for pid in pids.lines() {
        pid_vec.push(pid.to_owned());
    }
    Ok(pid_vec)
}

pub fn kill_all_servers() {
    let pids = match get_all_server_pids() {
        Ok(pids) => pids,
        Err(e) => {
            crate::error!("Failed to get all server pids: {e}");
            return;
        }
    };
    for pid in pids {
        std::process::Command::new("kill")
            .arg(pid)
            .status()
            .expect("Failed to kill process");
    }
}

impl Drop for LlamaCppServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::LlmInterface;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_server() {
        let builder = LlmInterface::llama_cpp();
        let loaded = builder.init().await.unwrap();
        std::mem::drop(loaded);

        let new_builder = LlmInterface::llama_cpp();

        let new_server = LlamaCppServer::new(
            &new_builder.config.clone(),
            new_builder.local_config.clone(),
        )
        .unwrap();
        if new_server.test_connection(
            std::time::Duration::from_millis(STATUS_CHECK_TIME_MS),
            std::time::Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        ) == ServerStatus::Running
        {
            panic!("Server should be stopped after dropping");
        }

        let _loaded = new_builder.init().await.unwrap();

        new_server.shutdown();

        if new_server.test_connection(
            std::time::Duration::from_millis(STATUS_CHECK_TIME_MS),
            std::time::Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        ) == ServerStatus::Running
        {
            panic!("Server should be stopped after killing");
        }
    }
}
