// Internal modules
mod config;
mod health;
mod models;
mod status;

// Internal imports
use crate::llms::{api::ApiClient, local::llama_cpp::LlamaCppConfig};
use llm_devices::{get_target_directory, DeviceConfig};
use status::{server_status, ServerStatus};
use std::process::Command;

// Public exports
pub use config::LlamaCppServerConfig;

const STATUS_CHECK_TIME_MS: u64 = 650;
const STATUS_RETRY_TIMEOUT_MS: u64 = 200;
const START_UP_CHECK_TIME_S: u64 = 180;
const START_UP_RETRY_TIME_S: u64 = 5;

pub struct LlamaCppServer {
    pub device_config: DeviceConfig,
    pub(crate) server_config: LlamaCppServerConfig,
    pub server_process: Option<std::process::Child>,
    pub host: String,
    pub server_http_path: String,
    pub port: Option<String>,
    pub inference_ctx_size: u64,
}

impl LlamaCppServer {
    pub fn new(
        device_config: DeviceConfig,
        host: &str,
        port: &Option<String>,
        inference_ctx_size: u64,
    ) -> crate::Result<Self> {
        let server_http_path = if let Some(port) = port {
            format!("{}:{}", &host, port)
        } else {
            host.to_owned()
        };

        Ok(Self {
            server_process: None,
            server_config: LlamaCppServerConfig::new(&device_config)?,
            server_http_path,
            host: host.to_owned(),
            port: port.as_deref().map(|p| p.to_owned()),
            inference_ctx_size,
            device_config,
        })
    }

    pub(super) async fn start_server(
        &mut self,
        client: &ApiClient<LlamaCppConfig>,
    ) -> Result<(), crate::Error> {
        match server_status(
            &self.device_config.local_model_path,
            &self.server_http_path,
            std::time::Duration::from_millis(STATUS_CHECK_TIME_MS),
            std::time::Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
            client,
        )
        .await?
        {
            ServerStatus::RunningRequested => return Ok(()),
            ServerStatus::Offline => (),
            ServerStatus::RunningModel(model_id) => match kill_server_from_model(&model_id) {
                Ok(_) => (),
                Err(e) => {
                    crate::error!(
                        "Failed to kill LlamaCppServer with model ID: {} {}",
                        model_id,
                        e
                    );
                    kill_all_servers()?;
                }
            },
        };

        let original = if !self.device_config.use_gpu {
            let original = std::env::var("CUDA_VISIBLE_DEVICES").ok();
            std::env::set_var("CUDA_VISIBLE_DEVICES", "");
            original
        } else {
            None
        };

        self.server_process = Some(self.start_server_backend()?);

        match server_status(
            &self.device_config.local_model_path,
            &self.server_http_path,
            std::time::Duration::from_secs(START_UP_CHECK_TIME_S),
            std::time::Duration::from_secs(START_UP_RETRY_TIME_S),
            client,
        )
        .await?
        {
            ServerStatus::RunningRequested => {
                if !self.device_config.use_gpu {
                    match original {
                        Some(value) => std::env::set_var("CUDA_VISIBLE_DEVICES", value),
                        None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
                    }
                }
                crate::trace!(
                    "Started LlamaCppServer with process PID: {}",
                    self.server_process
                        .as_ref()
                        .expect("LlamaCppServer process not created")
                        .id()
                );
                Ok(())
            }
            ServerStatus::Offline => {
                self.shutdown()?;
                crate::bail!("Failed to start LlamaCppServer");
            }
            ServerStatus::RunningModel(model_id) => {
                match kill_server_from_model(&model_id) {
                    Ok(_) => (),
                    Err(e) => {
                        crate::error!(
                            "Failed to kill LlamaCppServer with model ID: {} {}",
                            model_id,
                            e
                        );
                        kill_all_servers()?;
                    }
                };
                crate::bail!("Failed to start LlamaCppServer with correct model.");
            }
        }
    }

    fn start_server_backend(&self) -> crate::Result<std::process::Child> {
        let path = get_target_directory()?.join("llama_cpp");
        let mut command = std::process::Command::new("./llama-server");
        command.current_dir(path);
        self.server_config.populate_args(&mut command);
        command
            .arg("--model")
            .arg(&self.device_config.local_model_path)
            .arg("--ctx-size")
            .arg(self.inference_ctx_size.to_string())
            .arg("--timeout")
            .arg("600")
            .arg("--host")
            .arg(&self.host)
            .arg("--log-disable")
            .arg("--verbose");

        if let Some(port) = &self.port {
            command.arg("--port").arg(port);
        }
        crate::info!("Starting LlamaCppServer with command: {:?}", command);
        let process = command.spawn().expect("Failed to start LlamaCppServer");

        Ok(process)
    }

    pub(super) fn shutdown(&self) -> crate::Result<()> {
        let process = if let Some(server_process) = &self.server_process {
            server_process
        } else {
            crate::error!("LlamaCppServer process not started. No need to shutdown.");
            return Ok(());
        };

        let pid = process.id();
        match kill_server_from_pid(pid) {
            Ok(_) => {
                crate::info!("LlamaCppServer process with PID: {} killed", pid);
                Ok(())
            }
            Err(e) => {
                crate::bail!("Failed to kill LlamaCppServer process: {}", e);
            }
        }
    }
}

pub fn kill_server_from_model(model_id: &str) -> crate::Result<()> {
    let pid = if let Some(pid) = get_server_pid_by_model(model_id)? {
        pid
    } else {
        return Ok(());
    };

    match kill_server_from_pid(pid) {
        Ok(_) => Ok(()),
        Err(e) => {
            crate::bail!("Failed to kill LlamaCppServer process: {}", e);
        }
    }
}

pub fn kill_server_from_pid(pid: u32) -> crate::Result<()> {
    match std::process::Command::new("kill")
        .arg(pid.to_string())
        .status()
    {
        Ok(_) => (),
        Err(e) => {
            crate::bail!(
                "std::process::Command::new(\"kill\") failed to kill LlamaCppServer process: {}",
                e
            )
        }
    };
    std::thread::sleep(std::time::Duration::from_millis(100));
    match server_pid_exists(pid) {
        Ok(true) => {
            crate::bail!(
                "std::process::Command::new(\"kill\") failed to kill LlamaCppServer process with PID: {}",
                pid
            )
        }
        Ok(false) => Ok(()),
        Err(e) => {
            crate::bail!("Failed to check if LlamaCppServer process exists: {e}");
        }
    }
}

pub fn kill_all_servers() -> crate::Result<()> {
    crate::info!("Killing all LlamaCppServer processes");
    let pids = match get_all_server_pids() {
        Ok(pids) => pids,
        Err(e) => {
            crate::bail!("Failed to get all LlamaCppServer pids: {e}");
        }
    };
    crate::info!("Killing LlamaCppServer processes with PIDs: {:?}", pids);
    for pid in pids {
        std::process::Command::new("kill")
            .arg(pid)
            .status()
            .expect("Failed to kill process");
    }
    std::thread::sleep(std::time::Duration::from_millis(250));
    let pids = match get_all_server_pids() {
        Ok(pids) => pids,
        Err(e) => {
            crate::bail!("Failed to get all LlamaCppServer pids: {e}");
        }
    };
    if !pids.is_empty() {
        crate::bail!(
            "Failed to kill all LlamaCppServer processes with PIDs: {:?}",
            pids
        );
    } else {
        crate::info!("All LlamaCppServer processes killed");
        Ok(())
    }
}

pub fn get_server_pid_by_model(model_id: &str) -> crate::Result<Option<u32>> {
    // pgrep -f '^./llama-server.*Meta-Llama-3.1-8B-Instruct'
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("wmic")
            .args(&[
                "process",
                "where",
                &format!("commandline like '%{}%'", model_id),
                "get",
                "processid",
            ])
            .output()?;
        let output_str = String::from_utf8_lossy(&output.stdout);
        Ok(output_str
            .lines()
            .skip(1) // Skip header
            .next()
            .and_then(|pid_str| pid_str.trim().parse::<u32>().ok()))
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let output = Command::new("pgrep")
            .args(&["-f", &format!("./llama-server.*{}", model_id)])
            .output()?;
        let pid = String::from_utf8_lossy(&output.stdout);
        Ok(pid.lines().next().and_then(|s| s.parse::<u32>().ok()))
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err(io::Error::new(io::ErrorKind::Other, "Unsupported operating system").into())
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
            Ok(false)
        }
        Err(e) => {
            crate::bail!("Failed to check if LlamaCppServer process exists: {}", e)
        }
    }
}

pub fn get_all_server_pids() -> crate::Result<Vec<String>> {
    // pgrep -f '^./llama-server'
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("tasklist")
            .args(&["/FO", "CSV", "/NH", "/FI", "IMAGENAME eq llama-server.exe"])
            .output()?;
        let output_str = String::from_utf8_lossy(&output.stdout);
        Ok(output_str
            .lines()
            .filter_map(|line| {
                line.split(',')
                    .nth(1)
                    .map(|pid| pid.trim_matches('"').to_string())
            })
            .collect())
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let output = Command::new("pgrep")
            .args(&["-f", "^./llama-server"])
            .output()?;
        let pids = String::from_utf8_lossy(&output.stdout);
        Ok(pids.lines().map(|pid| pid.to_string()).collect())
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        Err(io::Error::new(io::ErrorKind::Other, "Unsupported operating system").into())
    }
}

impl Drop for LlamaCppServer {
    fn drop(&mut self) {
        match self.shutdown() {
            Ok(_) => (),
            Err(e) => crate::error!("Failed to shutdown LlamaCppServer: {}", e),
        }
    }
}
