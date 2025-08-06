#[derive(Debug, thiserror::Error, serde::Serialize)]
pub enum FileSystemError {
    #[error("Not a directory: '{path}'")]
    NotADirectory { path: std::path::PathBuf },

    #[error("File not found: '{path}'")]
    NotFound { path: std::path::PathBuf },

    #[error("File '{target}' not found in '{root_dir}'")]
    FileNotFoundInDir {
        root_dir: std::path::PathBuf,
        target: String,
    },

    #[error("Not a regular file: '{path}'")]
    NotRegularFile { path: std::path::PathBuf },

    #[error("Permission denied: '{path}': {message}")]
    PermissionDenied {
        path: std::path::PathBuf,
        message: String,
    },

    #[error("Incorrect file extension for '{path}': expected .{expected}, got {actual}")]
    IncorrectExtension {
        path: std::path::PathBuf,
        expected: &'static str,
        actual: String,
    },

    #[error("Environment variable '{var}' is not set")]
    EnvVarMissing { var: String },

    #[error("{operation} failed for '{path}': {message}")]
    IoError {
        operation: &'static str,
        path: std::path::PathBuf,
        message: String,
    },
}

impl FileSystemError {
    pub fn from_io_error(
        operation: &'static str,
        path: impl Into<std::path::PathBuf>,
        error: std::io::Error,
    ) -> Self {
        let path = path.into();
        match error.kind() {
            std::io::ErrorKind::NotFound => Self::NotFound { path },
            std::io::ErrorKind::PermissionDenied => Self::PermissionDenied {
                path,
                message: error.to_string(),
            },
            _ => Self::IoError {
                operation,
                path,
                message: error.to_string(),
            },
        }
    }

    pub fn incorrect_extension(
        path: impl Into<std::path::PathBuf>,
        expected: &'static str,
    ) -> Self {
        let path = path.into();
        let actual = match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => ext.to_owned(),
            None => "(none)".to_owned(),
        };

        Self::IncorrectExtension {
            path,
            expected,
            actual,
        }
    }
}

pub type FileSystemResult<T> = std::result::Result<T, FileSystemError>;
