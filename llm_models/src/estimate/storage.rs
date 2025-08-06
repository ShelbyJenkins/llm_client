use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::fs::{
    dir::StorageLocation,
    error::{FileSystemError, FileSystemResult},
};

/// **User‑supplied specification** for turning configuration data (CLI flags,
/// config files, environment variables …) into a concrete [`StorageLocation`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageSpec {
    /// Use the per‑platform “project data directory”, e.g.
    ///
    /// * **macOS**: `~/Library/Application Support/<org>/<app>`
    /// * **Windows**: `%APPDATA%\<org>\<app>`
    /// * **Linux (XDG)**: `~/.local/share/<app>`  (‖ `org` is ignored by the
    ///   spec in practice)
    Project {
        /// Second‑level domain or organisation / vendor name.
        org: String,
        /// The application or binary name.
        app: String,
    },
    /// Use an **explicit absolute or relative path** supplied by the user.
    ///
    /// Ancestors are created automatically (`mkdir -p`) and the path is then
    /// canonicalised.
    CustomPath(PathBuf),
    /// Look up an **environment variable whose _value_ is a directory path**.
    ///
    /// The variable name is stored; resolution happens when constructing a
    /// [`StorageLocation`].
    PathEnvVar(String),
}

impl TryFrom<&StorageSpec> for StorageLocation {
    type Error = FileSystemError;

    /// Convert a borrowed [`StorageSpec`] into a concrete, verified
    /// `StorageLocation`.
    fn try_from(spec: &StorageSpec) -> FileSystemResult<Self> {
        match spec {
            StorageSpec::Project { org, app } => Self::from_project(org, app),
            StorageSpec::CustomPath(p) => Self::from_custom_path(p),
            StorageSpec::PathEnvVar(k) => Self::from_path_env_var(k),
        }
    }
}
