#[derive(serde::Serialize, Debug, thiserror::Error)]
pub enum ProcessError {
    /// OS rejected a spawn, kill, wait or similar operation.
    #[error("failed to {action} process: {source}")]
    CommandFailed {
        action: &'static str,
        #[source]
        #[serde(serialize_with = "crate::error::std_io_error_to_string")]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// We tried an operation the current user is not allowed to perform.
    #[error("insufficient privilege to {action}: {source}")]
    PermissionDenied {
        action: &'static str,
        #[source]
        #[serde(serialize_with = "crate::error::std_io_error_to_string")]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// We looked for a PID / command-line pattern but found nothing.
    #[error("no matching server process found for {query}")]
    NoSuchProcess { query: String },

    /// Graceful termination window expired; these PIDs remain alive.
    #[error("{operation} exceeded {elapsed:?}; PIDs still running: {leftovers:?}")]
    TerminationTimeout {
        operation: &'static str,
        elapsed: std::time::Duration,
        leftovers: Vec<u32>,
    },
}

pub type Result<T> = std::result::Result<T, ProcessError>;
