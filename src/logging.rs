use serde::{Deserialize, Serialize};
use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
};
use tracing::subscriber::{set_default, DefaultGuard};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::fmt::{self};

#[derive(Serialize, Deserialize, Debug)]
pub struct LogEntry {
    event: String,
    severity: u8,
    description: String,
}

pub fn create_logger(name: &str) -> DefaultGuard {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let log_dir = manifest_dir.join("logs");

    if !Path::new(&log_dir).exists() {
        create_dir_all(&log_dir).expect("Failed to create log directory");
    }

    let file_appender = RollingFileAppender::builder()
        .rotation(Rotation::HOURLY)
        .filename_prefix(name)
        .filename_suffix("json")
        .build(log_dir)
        .unwrap();

    let subscriber = fmt::Subscriber::builder()
        .json()
        .flatten_event(true)
        .with_writer(file_appender)
        .finish();

    set_default(subscriber)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use serial_test::serial;
    #[tokio::test]
    #[serial]
    async fn test() {
        create_logger("test");

        let log_entry = LogEntry {
            event: "LoginAttempt".to_string(),
            severity: 5,
            description: "User attempted to log in.".to_string(),
        };
        tracing::info!(entry = ?log_entry, "Log Event");
        tracing::info!(entry = ?log_entry, "Log Event");
        tracing::info!(entry = ?log_entry, "Log Event");
    }
}
