use std::path::PathBuf;

use dotenvy::dotenv;
use ggus::GGuf;
use hf_hub::{
    Cache, CacheRepo, Repo, RepoType,
    api::tokio::{Api, ApiBuilder, ApiError, ApiRepo},
};
use secrecy::ExposeSecret;

use crate::{
    hf::{config::HfConfig, id::HfRepoId},
    manifest::profile_from_hf::{HfProfileError, RawRepoInfo},
};

#[derive(Debug)]
pub struct HfClient {
    cache: Cache,
    api: Api,
}

impl Default for HfClient {
    fn default() -> Self {
        HfConfig::default().build()
    }
}

impl HfClient {
    pub fn new(config: HfConfig) -> Self {
        let cache = if let Some(storage_location) = config.storage_location {
            Cache::new(storage_location.as_path_buf())
        } else {
            Cache::default() // Downloads to Path: "/root/.cache/huggingface/hub/
        };

        let token = match config.token {
            Some(token) => Some(token.expose_secret().to_owned()),
            None => {
                dotenv().ok();
                match dotenvy::var(&config.token_env_var) {
                    Ok(token) => Some(token),
                    Err(_) => {
                        crate::trace!(
                            "{} not found in dotenv, nor was it set manually",
                            config.token_env_var
                        );
                        None
                    }
                }
            }
        };

        let mut builder = ApiBuilder::from_cache(cache.clone())
            .with_progress(config.progress)
            .with_token(token);

        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint.to_string());
        };
        // if let Some(max_retries) = config.max_retries {
        //     builder = builder.with_retries(max_retries);
        // }

        let api = builder.build().unwrap();
        HfClient { cache, api }
    }

    pub fn api_repo(&self, repo_id: &HfRepoId) -> ApiRepo {
        match repo_id.sha() {
            Some(sha) => self.api.repo(Repo::with_revision(
                repo_id.repo_id().to_owned(),
                RepoType::Model,
                sha.to_owned(),
            )),
            None => self
                .api
                .repo(Repo::new(repo_id.repo_id().to_owned(), RepoType::Model)),
        }
    }

    pub fn cache_repo(&self, repo_id: &HfRepoId) -> CacheRepo {
        match repo_id.sha() {
            Some(sha) => self.cache.repo(Repo::with_revision(
                repo_id.repo_id().to_owned(),
                RepoType::Model,
                sha.to_owned(),
            )),
            None => self
                .cache
                .repo(Repo::new(repo_id.repo_id().to_owned(), RepoType::Model)),
        }
    }

    pub async fn load_file<T: AsRef<str>>(
        &self,
        file_name: T,
        repo_id: &HfRepoId,
    ) -> Result<PathBuf, HfProfileError> {
        let api_repo = self.api_repo(repo_id);
        Ok(api_repo.get(file_name.as_ref()).await?)
    }

    pub async fn repo_info(&self, repo_id: &HfRepoId) -> Result<RawRepoInfo, HfProfileError> {
        let api_repo = self.api_repo(repo_id);
        api_repo
            .info_request()
            .query(&[("blobs", "true")])
            .send()
            .await
            .map_err(|e| HfProfileError::Api(ApiError::RequestError(e)))?
            .json()
            .await
            .map_err(|e| HfProfileError::Api(ApiError::RequestError(e)))
    }

    pub async fn load_gguf_header_buffer(
        &self,
        repo_id: &HfRepoId,
        file_name: &str,
    ) -> Result<Vec<u8>, HfProfileError> {
        assert!(file_name.ends_with(".gguf"));

        if let Some(cached) = self.cache_repo(repo_id).get(file_name) {
            match std::fs::read(cached) {
                Ok(buf) => match GGuf::new(&buf) {
                    Ok(_) => return Ok(buf),
                    Err(_) => (),
                },
                Err(_) => (),
            }
        }

        const INITIAL_CHUNK: usize = 512 * 1024; // 512 KiB
        const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

        let url = self.api_repo(repo_id).url(file_name);
        let client = self.api.client();

        let mut buf = Vec::with_capacity(INITIAL_CHUNK);
        let mut start = 0usize;
        let mut chunk = INITIAL_CHUNK;

        loop {
            let end = start + chunk - 1;
            let range_val = format!("bytes={start}-{end}");

            let resp = client
                .get(&url)
                .header("Range", range_val)
                .send()
                .await
                .map_err(|e| HfProfileError::Api(ApiError::RequestError(e)))?;

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| HfProfileError::Api(ApiError::RequestError(e)))?;

            buf.extend_from_slice(&bytes);
            match GGuf::new(&buf) {
                Ok(_) => {
                    return Ok(buf);
                }
                Err(e) => {
                    eprintln!("Error parsing GGuf: {}", e);
                    // Need more bytes; enlarge chunk, respect cap.
                    start = end + 1;
                    if chunk >= MAX_HEADER_BYTES {
                        return Err(HfProfileError::HeaderTooLarge {
                            needed: start,
                            limit: MAX_HEADER_BYTES,
                        });
                    }
                    chunk = (chunk * 2).min(MAX_HEADER_BYTES);
                }
            }
        }
    }
}
//    pub fn load_file<T: AsRef<str>>(
//         &self,
//         file_name: T,
//         repo_id: HfRepoId,
//     ) -> Result<PathBuf, crate::Error> {
//         let repo = self.api_repo(&repo_id);
//         match repo.get(file_name.as_ref()) {
//             Ok(path) => Ok(path),
//             Err(e) => {
//                 if let ApiError::LockAcquisition(lock_path) = &e {
//                     // Check if lock file exists
//                     if lock_path.exists() && !Self::is_file_in_use(lock_path) {
//                         // Remove the stale lock file
//                         std::fs::remove_file(&lock_path)?;

//                         // Try downloading again
//                         Ok(repo.get(file_name.as_ref())?)
//                     } else {
//                         crate::bail!(e)
//                     }
//                 } else {
//                     crate::bail!(e)
//                 }
//             }
//         }
//     }

//     #[cfg(target_family = "unix")]
//     fn is_file_in_use(lock_path: &std::path::Path) -> bool {
//         // Try both lsof and flock for more reliable detection

//         // First try flock
//         if let Ok(output) = std::process::Command::new("flock")
//             .arg("-n") // non-blocking
//             .arg(lock_path.to_str().unwrap_or(""))
//             .arg("-c")
//             .arg("echo") // just try to get the lock and echo
//             .output()
//         {
//             // If flock succeeds (exit code 0), the file is not locked
//             return output.status.success();
//         }

//         // Fallback to lsof
//         if let Ok(output) = std::process::Command::new("lsof")
//             .arg(lock_path.to_str().unwrap_or(""))
//             .output()
//         {
//             return output.status.success();
//         }

//         Self::is_lock_stale(lock_path)
//     }

//     #[cfg(target_family = "windows")]
//     fn is_file_in_use(lock_path: &std::path::Path) -> bool {
//         // On Windows, try to open the file with exclusive access
//         // If we can't, it means someone else has it open
//         if let Ok(file) = std::fs::OpenOptions::new()
//             .write(true)
//             .create(false)
//             .open(lock_path)
//         {
//             drop(file);
//             false
//         } else {
//             // If that fails, fall back to checking if lock is stale
//             Self::is_lock_stale(lock_path)
//         }
//     }

//     fn is_lock_stale(lock_path: &std::path::Path) -> bool {
//         if let Ok(metadata) = std::fs::metadata(lock_path) {
//             if let Ok(modified) = metadata.modified() {
//                 if let Ok(duration) = std::time::SystemTime::now().duration_since(modified) {
//                     // Consider locks older than 2 hours as stale
//                     return duration > std::time::Duration::from_secs(7200);
//                 }
//             }
//         }
//         false
//     }
#[cfg(test)]
mod tests {
    // use super::*;

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
