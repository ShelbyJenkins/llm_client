#[derive(Debug, thiserror::Error)]
pub enum LlmModelsError {
    #[error("Invalid model profile: {0}")]
    MissingRequiredValue(String),
    #[error(transparent)]
    FileSystem(#[from] crate::fs::error::FileSystemError),
}

pub type LlmModelsResult<T> = std::result::Result<T, LlmModelsError>;
