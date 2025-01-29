use crate::get_target_directory;

use colorful::Colorful;
use indenter::indented;
use std::fmt::Write;
use std::path::PathBuf;
use std::{fs::create_dir_all, path::Path};
use tracing_subscriber::layer::SubscriberExt;

/// Configuration for the logging system.
///
/// Manages log levels, file output, and logger initialization.
#[derive(Clone, Debug)]
pub struct LoggingConfig {
    /// Log level threshold (ERROR, WARN, INFO, DEBUG, TRACE)
    pub level: tracing::Level,

    /// Whether logging is enabled
    pub logging_enabled: bool,

    /// Name used to identify this logger in output
    pub logger_name: String,

    /// Custom path for log files. If None, uses default path
    pub log_path: Option<PathBuf>,

    /// Guard for the tracing subscriber
    pub _tracing_guard: Option<std::sync::Arc<tracing::subscriber::DefaultGuard>>,

    /// Whether this is a build log
    pub build_log: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: tracing::Level::INFO,
            logging_enabled: true,
            logger_name: "llm_interface".to_string(),
            log_path: None,
            _tracing_guard: None,
            build_log: false,
        }
    }
}

impl LoggingConfig {
    /// Creates a new LoggingConfig with default settings.
    ///
    /// Defaults to:
    /// - INFO level
    /// - Logging enabled
    /// - "llm_interface" logger name
    /// - Default log path
    pub fn new() -> Self {
        Default::default()
    }

    /// Initializes and starts the logger with the current configuration.
    ///
    /// If logging is enabled, creates log files and sets up console output.
    /// Logs are rotated hourly and up to 6 files are kept.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Log directory creation fails
    /// - File appender creation fails
    pub fn load_logger(&mut self) -> crate::Result<()> {
        self._tracing_guard = if self.logging_enabled {
            Some(std::sync::Arc::new(self.create_logger()?))
        } else {
            None
        };

        println!(
            "{}",
            format!("Starting {} Logger", self.logger_name)
                .color(colorful::RGB::new(0, 139, 248))
                .bold()
        );

        Ok(())
    }

    fn create_logger(&mut self) -> crate::Result<tracing::subscriber::DefaultGuard> {
        let log_dir = if let Some(log_path) = &self.log_path {
            log_path.clone()
        } else {
            let target_dir = get_target_directory()?;
            if self.build_log {
                target_dir.join("llm_devices_build_logs")
            } else {
                target_dir
                    .parent()
                    .map(Path::to_path_buf)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))?
                    .join("llm_logs")
            }
        };

        if !Path::new(&log_dir).exists() {
            create_dir_all(&log_dir).expect("Failed to create log directory");
        }

        let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
            .rotation(tracing_appender::rolling::Rotation::HOURLY)
            .max_log_files(6)
            .filename_prefix(&self.logger_name)
            .filename_suffix("log")
            .build(log_dir)
            .unwrap();

        let filter = tracing_subscriber::EnvFilter::builder()
            .with_default_directive(self.level.into())
            .parse_lossy("");

        let file_layer = tracing_subscriber::fmt::layer()
            .pretty()
            .with_ansi(false) // Disable ANSI codes for file output
            .with_writer(file_appender);

        let terminal_layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_ansi(false) // Enable ANSI codes for terminal output
            .with_writer(std::io::stdout);

        let subscriber = tracing_subscriber::registry()
            .with(filter)
            .with(file_layer)
            .with(terminal_layer);

        Ok(tracing::subscriber::set_default(subscriber))
    }
}

/// Trait for configuring logging behavior.
///
/// Provides a fluent interface for configuring logging settings.
#[allow(dead_code)]
pub trait LoggingConfigTrait {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig;

    /// Enables or disables logging for the configuration.
    ///
    /// # Arguments
    ///
    /// * `enabled` - A boolean value where `true` enables logging and `false` disables it.
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow for method chaining.
    fn logging_enabled(mut self, enabled: bool) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().logging_enabled = enabled;
        self
    }

    /// Sets the name of the logger.
    ///
    /// This method allows you to specify a custom name for the logger, which can be useful
    /// for identifying the source of log messages in applications with multiple components
    /// or services.
    ///
    /// # Arguments
    ///
    /// * `logger_name` - A string-like value that can be converted into a `String`.
    ///   This will be used as the name for the logger.
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow for method chaining.
    fn logger_name<S: Into<String>>(mut self, logger_name: S) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().logger_name = logger_name.into();
        self
    }

    /// Sets the path where log files will be stored.
    ///
    /// # Arguments
    ///
    /// * `path` - A path-like object that represents the directory where log files should be stored.
    ///
    /// # Returns
    ///
    /// Returns `Self` to allow for method chaining.
    ///
    /// # Notes
    ///
    /// - If no path is set, the default path is `CARGO_MANIFEST_DIRECTORY/llm_logs`.
    fn log_path<P: AsRef<Path>>(mut self, path: P) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().log_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the log level to TRACE.
    ///
    /// Use TRACE for purely "I am here!" logs. They indicate the flow of execution
    /// without additional context.
    ///
    /// TRACE logs should not be used to log variables or decisions.
    fn log_level_trace(mut self) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().level = tracing::Level::TRACE;
        self
    }

    /// Sets the log level to DEBUG.
    ///
    /// Use DEBUG to log variables or decisions. This level is appropriate for information
    /// that is useful for debugging but not necessary for normal operation.
    ///
    /// # Examples
    ///
    /// DEBUG logs should focus on logging specific data points or choices made in the code.
    fn log_level_debug(mut self) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().level = tracing::Level::DEBUG;
        self
    }

    /// Sets the log level to INFO.
    ///
    /// Use INFO for important runtime events that don't prevent the application from working
    /// but are significant milestones or status updates.
    ///
    /// INFO logs should provide a high-level overview of the application's operation.
    fn log_level_info(mut self) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().level = tracing::Level::INFO;
        self
    }

    /// Sets the log level to WARN.
    ///
    /// Use WARN for errors that were recovered from or potential issues that don't prevent
    /// the application from working but might lead to problems if not addressed.
    ///
    /// WARN logs often indicate situations that should be monitored or addressed soon.
    fn log_level_warn(mut self) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().level = tracing::Level::WARN;
        self
    }

    /// Sets the log level to ERROR.
    ///
    /// Use ERROR to log errors within specific tasks that cause the task to fail
    /// but don't crash the entire application.
    ///
    /// ERROR logs indicate serious issues that need immediate attention but don't
    /// necessarily stop the application.
    fn log_level_error(mut self) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().level = tracing::Level::ERROR;
        self
    }
}

/// Writes an indented line without newline.
///
/// # Arguments
///
/// * `f` - The formatter to write to
/// * `arg` - The arguments to format and write
///
/// # Returns
///
/// std::fmt::Result indicating success or failure
pub fn i_ln(f: &mut std::fmt::Formatter<'_>, arg: std::fmt::Arguments<'_>) -> std::fmt::Result {
    write!(indented(f), "{}", arg)?;
    Ok(())
}

/// Writes an indented line with newline.
///
/// # Arguments
///
/// * `f` - The formatter to write to
/// * `arg` - The arguments to format and write
///
/// # Returns
///
/// std::fmt::Result indicating success or failure
pub fn i_nln(f: &mut std::fmt::Formatter<'_>, arg: std::fmt::Arguments<'_>) -> std::fmt::Result {
    writeln!(indented(f), "{}", arg)?;
    Ok(())
}

/// Writes multiple indented lines without newlines.
///
/// # Arguments
///
/// * `f` - The formatter to write to
/// * `args` - Array of arguments to format and write
///
/// # Returns
///
/// std::fmt::Result indicating success or failure
pub fn i_lns(
    f: &mut std::fmt::Formatter<'_>,
    args: &[std::fmt::Arguments<'_>],
) -> std::fmt::Result {
    for arg in args {
        write!(indented(f), "{}", arg)?;
    }
    Ok(())
}

/// Writes multiple indented lines with newlines.
///
/// # Arguments
///
/// * `f` - The formatter to write to
/// * `args` - Array of arguments to format and write
///
/// # Returns
///
/// std::fmt::Result indicating success or failure
pub fn i_nlns(
    f: &mut std::fmt::Formatter<'_>,
    args: &[std::fmt::Arguments<'_>],
) -> std::fmt::Result {
    for arg in args {
        writeln!(indented(f), "{}", arg)?;
    }
    Ok(())
}
