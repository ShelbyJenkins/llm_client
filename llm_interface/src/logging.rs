use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
};
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Debug)]
pub struct LoggingConfig {
    pub level: tracing::Level,
    pub logging_enabled: bool,
    pub logger_name: String,
    pub _tracing_guard: Option<std::sync::Arc<tracing::subscriber::DefaultGuard>>,
}

impl LoggingConfig {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: tracing::Level::TRACE,
            logging_enabled: true,
            logger_name: "llm_interface".to_string(),
            _tracing_guard: None,
        }
    }
}

impl LoggingConfig {
    pub(crate) fn load_logger(&mut self) -> crate::Result<()> {
        self._tracing_guard = if self.logging_enabled {
            Some(std::sync::Arc::new(self.create_logger()))
        } else {
            None
        };
        Ok(())
    }

    fn create_logger(&mut self) -> tracing::subscriber::DefaultGuard {
        let out_dir = std::path::PathBuf::from(env!("OUT_DIR"));
        let project_dir = out_dir
            .ancestors()
            .find(|path| {
                // Check if this path's directory name is 'target'
                if let Some(dir_name) = path.file_name() {
                    dir_name == "target"
                } else {
                    false
                }
            })
            .and_then(|target_dir| target_dir.parent());
        let workspace_dir = if let Some(project_dir) = project_dir {
            project_dir.to_owned()
        } else {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        };
        let log_dir = workspace_dir.join("llm_client_logs");

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
            .pretty()
            .with_ansi(true) // Enable ANSI codes for terminal output
            .with_writer(std::io::stdout);

        let subscriber = tracing_subscriber::registry()
            .with(filter)
            .with(file_layer)
            .with(terminal_layer);

        tracing::subscriber::set_default(subscriber)
    }
}

#[allow(dead_code)]
pub trait LoggingConfigTrait {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig;

    fn logging_enabled(mut self, enabled: bool) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().logging_enabled = enabled;
        self
    }

    fn logger_name<S: Into<String>>(mut self, logger_name: S) -> Self
    where
        Self: Sized,
    {
        self.logging_config_mut().logger_name = logger_name.into();
        self
    }

    /// Sets the log level to TRACE.
    ///
    /// Use TRACE for purely "I am here!" logs. They indicate the flow of execution
    /// without additional context.
    ///
    /// # Examples
    ///
    /// ```
    /// trace!("Entering function foo");
    /// trace!("Exiting function foo");
    /// ```
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
    /// ```
    /// debug!("User {} chose option {}", user_id, option);
    /// debug!("Using chunked sending for request");
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// info!("Server listening on port 80");
    /// info!("Logged into <API> as <USER>");
    /// info!("Completed daily database expiration task");
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// warn!("Connection attempt failed, retrying (attempt 2 of 3)");
    /// warn!("Disk usage above 90%, consider freeing up space");
    /// ```
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
    /// # Examples
    ///
    /// ```
    /// error!("Broken pipe responding to request");
    /// error!("Failed to write to database: {}", e);
    /// ```
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
