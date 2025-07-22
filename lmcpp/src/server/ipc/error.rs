#[derive(serde::Serialize, Debug, thiserror::Error)]
pub enum ClientError {
    #[error("I/O error: {0}")]
    #[serde(serialize_with = "crate::error::std_io_error_to_string")]
    Io(#[from] std::io::Error),

    #[error("timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("serialization error: {0}")]
    #[serde(serialize_with = "crate::error::std_io_error_to_string")]
    Serde(#[from] serde_json::Error),

    #[error("remote error {code}: {message}")]
    Remote { code: u16, message: String },

    #[error("client setup error: {reason}")]
    Setup { reason: String },
}

pub type Result<T> = std::result::Result<T, ClientError>;
