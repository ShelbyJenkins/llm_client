//! Downloads to Path: "/root/.cache/huggingface/hub/
use std::{
    path::PathBuf,
    sync::{LazyLock, OnceLock},
};

use dotenvy::dotenv;
use hf_hub::{
    Cache,
    api::sync::{Api, ApiBuilder, ApiError},
};

static HF_CACHE: LazyLock<Cache> = LazyLock::new(|| Cache::default());

const DEFAULT_ENV_VAR: &str = "HUGGING_FACE_TOKEN";

#[derive(Clone, Debug)]
pub struct HuggingFaceLoader {
    pub hf_token: Option<String>,
    pub hf_token_env_var: String,
    pub hf_api: OnceLock<Api>,
}

impl Default for HuggingFaceLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl HuggingFaceLoader {
    pub fn new() -> Self {
        Self {
            hf_token: None,
            hf_token_env_var: DEFAULT_ENV_VAR.to_string(),
            hf_api: OnceLock::new(),
        }
    }

    pub fn hf_api(&self) -> &Api {
        self.hf_api.get_or_init(|| {
            ApiBuilder::from_cache(HF_CACHE.clone())
                .with_progress(true)
                .with_token(self.load_hf_token())
                .build()
                .expect("Failed to build Hugging Face API")
        })
    }

    fn load_hf_token(&self) -> Option<String> {
        if let Some(hf_token) = &self.hf_token {
            crate::trace!("Using hf_token from parameter");
            return Some(hf_token.to_owned());
        }

        dotenv().ok();

        match dotenvy::var(&self.hf_token_env_var) {
            Ok(hf_token) => Some(hf_token),
            Err(_) => {
                crate::trace!(
                    "{} not found in dotenv, nor was it set manually",
                    self.hf_token_env_var
                );
                None
            }
        }
    }

    pub fn load_file<T: AsRef<str>, S: Into<String>>(
        &self,
        file_name: T,
        repo_id: S,
    ) -> Result<PathBuf, crate::Error> {
        let repo_id = repo_id.into();
        match self.hf_api().model(repo_id.clone()).get(file_name.as_ref()) {
            Ok(path) => Ok(path),
            Err(e) => {
                if let ApiError::LockAcquisition(lock_path) = &e {
                    // Check if lock file exists
                    if lock_path.exists() && !Self::is_file_in_use(lock_path) {
                        // Remove the stale lock file
                        std::fs::remove_file(&lock_path)?;

                        // Try downloading again
                        Ok(self.hf_api().model(repo_id).get(file_name.as_ref())?)
                    } else {
                        crate::bail!(e)
                    }
                } else {
                    crate::bail!(e)
                }
            }
        }
    }

    pub fn model_info<S: Into<String>>(
        &self,
        repo_id: S,
    ) -> Result<HuggingFaceRepoInfo, crate::Error> {
        let blobs_info: serde_json::Value = self
            .hf_api()
            .model(repo_id.into())
            .info_request()
            .query("blobs", "true")
            .call()
            .map_err(|e| crate::anyhow!(e))?
            .into_json()
            .map_err(|e| crate::anyhow!(e))?;
        let result: HuggingFaceRepoInfo = serde_json::from_value(blobs_info.clone())?;
        Ok(result)
    }

    pub fn load_model_safe_tensors<S: Into<String>>(
        &self,
        repo_id: S,
    ) -> Result<Vec<PathBuf>, crate::Error> {
        let repo_id = repo_id.into();
        let mut safe_tensor_filenames = vec![];
        let siblings = self.hf_api().model(repo_id.clone()).info()?.siblings;
        for sib in siblings {
            if sib.rfilename.ends_with(".safetensors") {
                safe_tensor_filenames.push(sib.rfilename);
            }
        }
        let mut safe_tensor_paths = vec![];
        for safe_tensor_filename in &safe_tensor_filenames {
            let safe_tensor_path = self.load_file(safe_tensor_filename, repo_id.clone())?;

            println!("Downloaded safe tensor: {:?}", safe_tensor_path);
            safe_tensor_paths.push(safe_tensor_path);
        }
        Ok(safe_tensor_paths)
    }

    pub fn parse_full_model_url(model_url: &str) -> (String, String, String) {
        if !model_url.starts_with("https://huggingface.co") {
            panic!(
                "URL does not start with https://huggingface.co\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf"
            );
        } else if !model_url.ends_with(".gguf") {
            panic!(
                "URL does not end with .gguf\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf"
            );
        } else {
            let parts: Vec<&str> = model_url.split('/').collect();
            if parts.len() < 5 {
                panic!(
                    "URL does not have enough parts\n Format should be like: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf"
                );
            }
            let model_id = parts[4].to_string();
            let repo_id = format!("{}/{}", parts[3], parts[4]);
            let gguf_model_filename = parts.last().unwrap_or(&"").to_string();
            (model_id, repo_id, gguf_model_filename)
        }
    }

    #[cfg(target_family = "unix")]
    fn is_file_in_use(lock_path: &std::path::Path) -> bool {
        // Try both lsof and flock for more reliable detection

        // First try flock
        if let Ok(output) = std::process::Command::new("flock")
            .arg("-n") // non-blocking
            .arg(lock_path.to_str().unwrap_or(""))
            .arg("-c")
            .arg("echo") // just try to get the lock and echo
            .output()
        {
            // If flock succeeds (exit code 0), the file is not locked
            return output.status.success();
        }

        // Fallback to lsof
        if let Ok(output) = std::process::Command::new("lsof")
            .arg(lock_path.to_str().unwrap_or(""))
            .output()
        {
            return output.status.success();
        }

        Self::is_lock_stale(lock_path)
    }

    #[cfg(target_family = "windows")]
    fn is_file_in_use(lock_path: &std::path::Path) -> bool {
        // On Windows, try to open the file with exclusive access
        // If we can't, it means someone else has it open
        if let Ok(file) = std::fs::OpenOptions::new()
            .write(true)
            .create(false)
            .open(lock_path)
        {
            drop(file);
            false
        } else {
            // If that fails, fall back to checking if lock is stale
            Self::is_lock_stale(lock_path)
        }
    }

    fn is_lock_stale(lock_path: &std::path::Path) -> bool {
        if let Ok(metadata) = std::fs::metadata(lock_path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = std::time::SystemTime::now().duration_since(modified) {
                    // Consider locks older than 2 hours as stale
                    return duration > std::time::Duration::from_secs(7200);
                }
            }
        }
        false
    }
}

impl HfTokenTrait for HuggingFaceLoader {
    fn hf_token_mut(&mut self) -> &mut Option<String> {
        &mut self.hf_token
    }

    fn hf_token_env_var_mut(&mut self) -> &mut String {
        &mut self.hf_token_env_var
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct HuggingFaceRepoInfo {
    pub siblings: Vec<HuggingFaceSibling>,
}

impl HuggingFaceRepoInfo {
    pub fn get_file(&self, filename: &str) -> Option<&HuggingFaceSibling> {
        self.siblings.iter().find(|sib| sib.rfilename == filename)
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct HuggingFaceSibling {
    #[serde(alias = "blobId")]
    pub blob_id: String,
    pub rfilename: String,
    pub size: u64,
}

pub struct HuggingFaceFileCacheStatus {
    pub available: bool,
    pub on_disk_file_size_bytes: Option<u64>,
}

impl HuggingFaceFileCacheStatus {
    pub fn new<T: AsRef<str>, S: Into<String>>(
        file_name: T,
        repo_id: S,
        total_file_size_bytes: u64,
    ) -> Result<HuggingFaceFileCacheStatus, crate::Error> {
        if let Some(pointer_path) = HF_CACHE.model(repo_id.into()).get(file_name.as_ref()) {
            let initial_size = std::fs::metadata(&pointer_path)
                .map_err(|e| crate::anyhow!(e))?
                .len();
            std::thread::sleep(std::time::Duration::from_millis(100));
            let final_size = std::fs::metadata(pointer_path)
                .map_err(|e| crate::anyhow!(e))?
                .len();

            if initial_size == final_size && final_size == total_file_size_bytes as u64 {
                Ok(HuggingFaceFileCacheStatus {
                    available: true,
                    on_disk_file_size_bytes: Some(final_size as u64),
                })
            } else {
                Ok(HuggingFaceFileCacheStatus {
                    available: false,
                    on_disk_file_size_bytes: Some(final_size as u64),
                })
            }
        } else {
            Ok(HuggingFaceFileCacheStatus {
                available: false,
                on_disk_file_size_bytes: None,
            })
        }
    }
}

pub trait HfTokenTrait {
    fn hf_token_mut(&mut self) -> &mut Option<String>;

    fn hf_token_env_var_mut(&mut self) -> &mut String;

    /// Set the API key for the client. Otherwise it will attempt to load it from the .env file.
    fn hf_token<S: Into<String>>(mut self, hf_token: S) -> Self
    where
        Self: Sized,
    {
        *self.hf_token_mut() = Some(hf_token.into());
        self
    }

    /// Set the environment variable name for the API key. Default is "HUGGING_FACE_TOKEN".
    fn hf_token_env_var<S: Into<String>>(mut self, hf_token_env_var: S) -> Self
    where
        Self: Sized,
    {
        *self.hf_token_env_var_mut() = hf_token_env_var.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_model_url_valid() {
        let (model_id, repo_id, filename) = HuggingFaceLoader::parse_full_model_url(
            "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q8_0.gguf",
        );
        assert_eq!(model_id, "zephyr-7B-alpha-GGUF");
        assert_eq!(repo_id, "TheBloke/zephyr-7B-alpha-GGUF");
        assert_eq!(filename, "zephyr-7b-alpha.Q8_0.gguf");
    }

    // #[cfg(target_family = "unix")]
    // #[test]
    // fn test_is_file_in_use_unix() {
    //     use std::io::Write;
    //     let temp_dir = std::env::temp_dir();
    //     let lock_path = temp_dir.join("test_in_use_unix.lock");

    //     // Create the file
    //     let mut file = std::fs::File::create(&lock_path).expect("Failed to create lock file");
    //     writeln!(file, "some lock content").unwrap();
    //     drop(file);

    //     // First verify file is not in use
    //     assert!(!HuggingFaceLoader::is_file_in_use(&lock_path));

    //     // Create a real system lock using flock in a separate process
    //     let mut child = std::process::Command::new("flock")
    //         .arg("-x") // exclusive lock
    //         .arg(lock_path.to_str().unwrap())
    //         .arg("-c")
    //         .arg("sleep 5") // hold lock for 5 seconds
    //         .spawn()
    //         .expect("Failed to spawn flock process");

    //     // Give the process time to acquire the lock
    //     std::thread::sleep(std::time::Duration::from_millis(500));

    //     // Now check if we detect it as in use
    //     assert!(HuggingFaceLoader::is_file_in_use(&lock_path));

    //     // Wait for child process to finish
    //     child.wait().expect("Lock process didn't finish");

    //     // Cleanup
    //     std::fs::remove_file(&lock_path).expect("Failed to remove test file");
    // }

    #[cfg(target_family = "windows")]
    #[test]
    fn test_is_file_in_use_windows() {
        use std::fs::OpenOptions;

        let temp_dir = std::env::temp_dir();
        let lock_path = temp_dir.join("test_in_use_windows.lock");

        // First verify file is not in use
        assert!(!HuggingFaceLoader::is_file_in_use(&lock_path));

        {
            // Create and hold the file open
            let _file = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&lock_path)
                .expect("Failed to create lock file");

            // Now it should be detected as in use
            assert!(HuggingFaceLoader::is_file_in_use(&lock_path));
        }

        // After dropping _file, should not be in use
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert!(!HuggingFaceLoader::is_file_in_use(&lock_path));

        // Cleanup
        std::fs::remove_file(&lock_path).expect("Failed to remove test file");
    }
}
