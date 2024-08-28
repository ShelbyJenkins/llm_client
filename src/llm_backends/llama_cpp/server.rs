use super::*;
use std::{
    net::TcpStream,
    path::Path,
    process::Command,
    thread,
    time::{Duration, Instant},
};

const STATUS_CHECK_TIME_MS: u64 = 650;
const STATUS_RETRY_TIMEOUT_MS: u64 = 200;
const START_UP_CHECK_TIME_S: u64 = 30;
const START_UP_RETRY_TIME_S: u64 = 5;
const LLAMA_PATH: &str = "src/llm_backends/llama_cpp/llama_cpp";

#[derive(PartialEq)]
pub enum ServerStatus {
    Running,
    RunningRequested,
    Stopped,
}

pub struct LlamaServerConfig {
    pub threads: u16,
    pub ctx_size: u32,
    pub n_gpu_layers: u16,
    pub host: String,
    pub port: String,
    pub llm_loader: OsLlmLoader,
    pub server_process: Option<std::process::Child>,
}

impl Default for LlamaServerConfig {
    fn default() -> Self {
        Self {
            threads: 8,
            ctx_size: 4096,
            n_gpu_layers: 12,
            host: "localhost".to_string(),
            port: "8080".to_string(),
            llm_loader: OsLlmLoader::new(),
            server_process: None,
        }
    }
}

impl LlamaServerConfig {
    /// Used for setting the context limits of the model, and also for calculating vram usage.
    pub fn ctx_size(mut self, ctx_size: u32) -> Self
    where
        Self: Sized,
    {
        self.ctx_size = ctx_size;
        self
    }

    /// The number of CPU threads to use. If loading purely in vram, this can be set to 1.
    pub fn threads(mut self, threads: u16) -> Self
    where
        Self: Sized,
    {
        self.threads = threads;
        self
    }

    /// If using the `available_vram` method, will automatically be set to max.
    pub fn n_gpu_layers(mut self, n_gpu_layers: u16) -> Self
    where
        Self: Sized,
    {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn load_model(&mut self) -> Result<OsLlm> {
        if let Some(preset_loader) = &mut self.llm_loader.preset_loader {
            if let Some(ctx_size) = preset_loader.use_ctx_size {
                self.ctx_size = ctx_size; // If the preset loader has a ctx_size set, we use that.
            } else {
                preset_loader.use_ctx_size = Some(self.ctx_size); // Otherwise we set the preset loader to use the ctx_size from server_config.
            }
            self.n_gpu_layers = 9999; // Since the model is guaranteed to be constrained to the vram size, we max n_gpu_layers.
        }
        if self.llm_loader.preset_loader.is_none() && self.llm_loader.gguf_loader.is_none() {
            self.llm_loader.preset_loader();
        }
        let model = self.llm_loader.load()?;
        if self.ctx_size > model.model_config_json.max_position_embeddings as u32 {
            eprintln!("Given value for ctx_size {} is greater than the model's max {}. Using the models max.", self.ctx_size, model.model_config_json.max_position_embeddings);
            self.ctx_size = model.model_config_json.max_position_embeddings as u32;
        };
        Ok(model)
    }

    pub async fn start_server<P: AsRef<Path>>(
        &mut self,
        local_model_path: P,
    ) -> Result<ServerStatus> {
        match self
            .connect_with_timeouts(
                &local_model_path,
                Duration::from_millis(STATUS_CHECK_TIME_MS),
                Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
            )
            .await?
        {
            ServerStatus::RunningRequested => return Ok(ServerStatus::RunningRequested),
            ServerStatus::Stopped => (),
            ServerStatus::Running => self.kill_server_process(),
        };

        self.server_process = Some(self.start_server_backend(&local_model_path));

        match self
            .connect_with_timeouts(
                &local_model_path,
                Duration::from_secs(START_UP_CHECK_TIME_S),
                Duration::from_secs(START_UP_RETRY_TIME_S),
            )
            .await?
        {
            ServerStatus::RunningRequested => Ok(ServerStatus::RunningRequested),
            ServerStatus::Stopped => {
                self.kill_server_process();
                tracing::info!("Failed to start server");
                panic!("Failed to start server")
            }
            ServerStatus::Running => {
                self.kill_server_process();
                tracing::info!("Failed to start server with correct model.");
                panic!("Failed to start server with correct model.")
            }
        }
    }

    fn start_server_backend<P: AsRef<Path>>(&self, local_model_path: P) -> std::process::Child {
        let requested_model = local_model_path.as_ref().to_string_lossy();
        let mut command = Command::new("./llama-server");

        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        println!("manifest_dir: {:?}", manifest_dir);
        let path = std::fs::canonicalize(manifest_dir.join(LLAMA_PATH))
            .expect("Failed to canonicalize path");

        command
            .current_dir(path)
            .arg("--n-gpu-layers")
            .arg(self.n_gpu_layers.to_string())
            .arg("--threads")
            .arg(self.threads.to_string())
            .arg("--model")
            .arg(requested_model.to_string())
            .arg("--ctx-size")
            .arg(self.ctx_size.to_string())
            .arg("--timeout")
            .arg("600")
            .arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(&self.port)
            .arg("--verbose")
            .arg("--log-disable");

        let process = command.spawn().expect("Failed to start server");
        println!("Starting server with process PID: {}", process.id());
        process
    }

    pub async fn connect<P: AsRef<Path>>(&self, local_model_path: P) -> Result<ServerStatus> {
        self.connect_with_timeouts(
            local_model_path,
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        )
        .await
    }

    async fn connect_with_timeouts<P: AsRef<Path>>(
        &self,
        local_model_path: P,
        test_duration: Duration,
        retry_timeout: Duration,
    ) -> Result<ServerStatus> {
        let requested_model = local_model_path.as_ref().to_string_lossy();
        if self.test_connection(test_duration, retry_timeout) == ServerStatus::Running {
            println!("Server is running.");
            tracing::info!("Server is running.");

            {
                if self
                    .check_server_config(&requested_model, 3, retry_timeout)
                    .await?
                    == ServerStatus::RunningRequested
                {
                    println!(
                        "Server is running with the correct model: {}",
                        &requested_model
                    );
                    tracing::info!(
                        "Server is running with the correct model: {}",
                        &requested_model
                    );
                    Ok(ServerStatus::RunningRequested)
                } else {
                    Ok(ServerStatus::Stopped)
                }
            }
        } else {
            Ok(ServerStatus::Stopped)
        }
    }

    fn test_connection(&self, test_time: Duration, retry_time: Duration) -> ServerStatus {
        let start_time = Instant::now();
        while Instant::now().duration_since(start_time) < test_time {
            match TcpStream::connect(format!("{}:{}", self.host, self.port)) {
                Ok(_) => {
                    return ServerStatus::Running;
                }
                Err(_) => thread::sleep(retry_time),
            };
        }
        ServerStatus::Stopped
    }

    async fn check_server_config(
        &self,
        requested_model: &str,
        conn_attempts: u8,
        retry_time: Duration,
    ) -> Result<ServerStatus> {
        let mut attempts: u8 = 0;
        while attempts < conn_attempts {
            let request = LlamaCompletionsRequestArgs::default()
                .prompt(vec![0u32])
                .cache_prompt(false)
                .n_predict(0u16)
                .build()?;
            match LlamaClient::new(&self.host, &self.port)
                .completions()
                .create(request)
                .await
            {
                Ok(res) => {
                    if requested_model == res.model {
                        return Ok(ServerStatus::RunningRequested);
                    } else {
                        println!("error in check_server_config:\n running model: {}\n requested_model: {requested_model}", res.model);
                        tracing::info!(
                       "error in check_server_config:\n running model: {}\n requested_model: {requested_model}", res.model
                        );
                        return Ok(ServerStatus::Running);
                    }
                }
                Err(e) => {
                    println!("error in check_server_config:\n{e}");
                    tracing::info!("error in check_server_config:\n{e}");
                    attempts += 1;
                    thread::sleep(retry_time);
                }
            }
        }
        Ok(ServerStatus::Stopped)
    }

    pub fn kill_server_process(&mut self) {
        if let Some(server_process) = &mut self.server_process {
            server_process
                .kill()
                .expect("Failed to kill server. This shouldn't ever panic.");
        }

        kill_all_servers();
        thread::sleep(Duration::from_secs(1));
    }
}

impl Drop for LlamaServerConfig {
    fn drop(&mut self) {
        self.kill_server_process();
    }
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

impl LlmPresetTrait for LlamaServerConfig {
    fn preset_loader(&mut self) -> &mut LlmPresetLoader {
        if self.llm_loader.preset_loader.is_none() {
            self.llm_loader.preset_loader = Some(LlmPresetLoader::new());
        }
        self.llm_loader.preset_loader.as_mut().unwrap()
    }
}

impl LlmGgufTrait for LlamaServerConfig {
    fn gguf_loader(&mut self) -> &mut LlmGgufLoader {
        if self.llm_loader.gguf_loader.is_none() {
            self.llm_loader.gguf_loader = Some(LlmGgufLoader::new());
        }
        self.llm_loader.gguf_loader.as_mut().unwrap()
    }
}

impl HfTokenTrait for LlamaServerConfig {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.llm_loader.hf_loader.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.llm_loader.hf_loader.hf_token_env_var
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_server() -> Result<()> {
        let mut config = LlamaServerConfig::default();
        let model = config.load_model()?;
        config.start_server(&model.local_model_path).await?;
        std::mem::drop(config);

        let mut new_config = LlamaServerConfig::default();
        if new_config.test_connection(
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        ) == ServerStatus::Running
        {
            panic!("Server should be stopped");
        }

        let model = new_config.load_model()?;
        new_config.start_server(&model.local_model_path).await?;

        new_config.kill_server_process();

        if new_config.test_connection(
            Duration::from_millis(STATUS_CHECK_TIME_MS),
            Duration::from_millis(STATUS_RETRY_TIMEOUT_MS),
        ) == ServerStatus::Running
        {
            panic!("Server should be stopped");
        }
        Ok(())
    }

    #[tokio::test]
    #[serial]
    async fn test_builder() -> Result<()> {
        let mut config = LlamaServerConfig::default()
            .ctx_size(2048)
            .threads(2)
            .n_gpu_layers(6)
            .llama3_8b_instruct();
        let model = config.load_model()?;
        config.start_server(model.local_model_path).await?;
        Ok(())
    }
}
