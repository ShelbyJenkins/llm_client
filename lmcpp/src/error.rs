// top-level error for the public API

#[derive(serde::Serialize, Debug, thiserror::Error)]
pub enum LmcppError {
    #[error(transparent)]
    Process(#[from] crate::server::process::error::ProcessError),

    #[error(transparent)]
    Client(#[from] crate::server::ipc::error::ClientError),

    #[error("invalid {field}: {reason}")]
    InvalidConfig { field: &'static str, reason: String },

    #[error("{what} is unavailable on {os}/{arch}: {reason}")]
    BackendUnavailable {
        /// Label such as `"CUDA"`, `"Metal"`, `"AVX‑512"`, `"ROCm"`.
        what: &'static str,
        /// `std::env::consts::OS`
        os: &'static str,
        /// `std::env::consts::ARCH`
        arch: &'static str,
        /// Extra context, e.g. `": NVML not found"` or `": no GPU detected"`.
        reason: String,
    },

    #[error("build failed: {0}")]
    BuildFailed(String),

    #[error("server launch failed: {0}")]
    ServerLaunch(String),

    #[error("download failed: {0}")]
    DownloadFailed(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("fingerprint mismatch: {reason}")]
    Fingerprint { reason: String },

    #[error("{operation} failed for '{path}'")]
    FileSystem {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        #[serde(serialize_with = "std_io_error_to_string")]
        source: std::io::Error,
    },
}

pub type LmcppResult<T> = std::result::Result<T, LmcppError>;

impl LmcppError {
    pub fn file_system(
        operation: &'static str,
        path: impl Into<std::path::PathBuf>,
        err: impl Into<std::io::Error>,
    ) -> Self {
        Self::FileSystem {
            operation,
            path: path.into(),
            source: err.into(),
        }
    }
}

pub(crate) fn std_io_error_to_string<S>(e: &impl std::fmt::Display, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_str(&e.to_string())
}
