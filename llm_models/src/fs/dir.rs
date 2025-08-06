//! Directory Path Helpers
//! ===============
//!
//!
//! ## Invariants enforced
//!
//! * **Canonical & absolute path** – all `.` / `..` components removed and
//!   symlinks resolved (the canonical form is platform‑appropriate; on Windows
//!   this includes case‑folding where applicable).
//! * **Entry‑type correctness**
//!   * [`ExistingDir`] is *always* an existing directory.
//!   * `ValidFile` (see sibling module) is *always* a regular file.
//!
//! These invariants are established once in the respective constructors and
//! thereafter hold for the lifetime of the value.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::fs::error::{FileSystemError, FileSystemResult};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct StorageLocation(
    /// The underlying directory that satisfied all invariants at construction
    /// time.  Public access goes through [`StorageLocation::as_path`].
    pub ExistingDir,
);

impl StorageLocation {
    /// Resolve a platform‑specific “project data” directory for the given
    /// **organisation**/**application** pair and wrap it in a `StorageLocation`.
    pub fn from_project(org: &str, app: &str) -> FileSystemResult<Self> {
        let pd = directories::ProjectDirs::from("", org, app).ok_or_else(|| {
            FileSystemError::IoError {
                operation: "get project directory",
                path: PathBuf::from(format!("{}/{}", org, app)),
                message: "unsupported platform".into(),
            }
        })?;
        let dir = ExistingDir::new(pd.data_dir())?;
        Self::ensure_perms(dir)
    }

    /// Construct a `StorageLocation` from a **custom user path**.  Missing
    /// directories are created automatically.
    pub fn from_custom_path(custom_path: &Path) -> FileSystemResult<Self> {
        let dir = ExistingDir::new(custom_path)?;
        Self::ensure_perms(dir)
    }

    /// Construct from the **value of an environment variable** expected to
    /// contain a directory path.
    ///
    /// # Errors
    /// * `FileSystemError::EnvVarMissing` if the variable is unset.
    /// * Any error emitted by [`ExistingDir::new`] or permission checks.
    pub fn from_path_env_var(env_var: &str) -> FileSystemResult<Self> {
        let raw = std::env::var_os(env_var).ok_or_else(|| FileSystemError::EnvVarMissing {
            var: env_var.into(),
        })?;
        let dir = ExistingDir::new(raw)?;
        Self::ensure_perms(dir)
    }

    /// **Empty and recreate** the directory (delegates to [`ExistingDir::reset`]).
    pub fn reset(&self) -> FileSystemResult<()> {
        self.0.reset()
    }

    /// **Permanently delete** the directory and all its contents
    /// (delegates to [`ExistingDir::remove`]).
    pub fn remove(&self) -> FileSystemResult<()> {
        self.0.remove()
    }

    /// Returns the underlying path as a `&Path`.
    #[inline]
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    #[inline]
    pub fn as_path_buf(self) -> PathBuf {
        self.0.as_path_buf()
    }

    /// Enforce Unix‑style “not world‑writable” policy (`mode & 0o022 == 0`).
    ///
    /// On non‑Unix targets this is a no‑op.  Returns a fully‑wrapped
    /// `StorageLocation` on success.
    fn ensure_perms(dir: ExistingDir) -> FileSystemResult<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&dir.0)
                .map_err(|e| FileSystemError::from_io_error("metadata", &dir.0, e))?
                .permissions()
                .mode()
                & 0o777;
            if mode & 0o022 != 0 {
                return Err(FileSystemError::PermissionDenied {
                    path: dir.0,
                    message: format!("directory is world‑writable (mode {:o})", mode),
                });
            }
        }
        Ok(Self(dir))
    }
}

/// A **canonical, existing directory**.
///
/// Creating an `ExistingDir` performs the following steps:
///
/// 1. *Canonicalise* the path (`std::fs::canonicalize`), resolving symlinks and
///    removing `.` / `..`.
/// 2. If the path does not exist, **create it recursively** (`mkdir -p`).
/// 3. Verify the result is in fact a **directory** (not a file, fifo, …).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ExistingDir(
    /// The underlying canonicalised [`PathBuf`].  Direct access is possible
    /// via [`std::ops::Deref`] or [`ExistingDir::as_path`].
    pub PathBuf,
);

impl ExistingDir {
    /// Create a new `ExistingDir`, auto‑creating any missing components.
    ///
    /// # Errors
    /// * Any I/O failure during creation or canonicalisation.
    /// * [`FileSystemError::NotADirectory`] if the canonical target is *not*
    ///   a directory.
    pub fn new<P: AsRef<Path>>(p: P) -> FileSystemResult<Self> {
        let path = p.as_ref();

        let canonical = match path.canonicalize() {
            Ok(abs) => abs,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                std::fs::create_dir_all(path)
                    .map_err(|e| FileSystemError::from_io_error("create directory", path, e))?;
                path.canonicalize()
                    .map_err(|e| FileSystemError::from_io_error("canonicalize path", path, e))?
            }
            Err(e) => Err(FileSystemError::from_io_error("canonicalize path", path, e))?,
        };

        if !canonical.is_dir() {
            return Err(FileSystemError::NotADirectory { path: canonical });
        }

        Ok(Self(canonical))
    }

    /// **Nuke & pave** – delete the directory tree and recreate it empty.
    ///
    /// Extremely useful for tests that need a predictable scratch directory.
    pub fn reset(&self) -> FileSystemResult<()> {
        self.remove()?;
        std::fs::create_dir_all(&self.0)
            .map_err(|e| FileSystemError::from_io_error("create directory", &self.0, e))?;
        Ok(())
    }

    /// Permanently delete the directory (recursively).  *Does not* recreate it.
    pub fn remove(&self) -> FileSystemResult<()> {
        match std::fs::remove_dir_all(self) {
            Ok(_) => (),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (), // already gone – fine
            Err(e) => {
                return Err(FileSystemError::from_io_error(
                    "remove directory",
                    &self.0,
                    e,
                ));
            }
        }

        debug_assert!(!self.exists());

        Ok(())
    }

    /// Returns the underlying path as a `&Path`.
    #[inline]
    pub fn as_path(&self) -> &Path {
        &self.0
    }

    #[inline]
    pub fn as_path_buf(self) -> PathBuf {
        self.0
    }
}

impl std::ops::Deref for ExistingDir {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for ExistingDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl TryFrom<PathBuf> for ExistingDir {
    type Error = FileSystemError;
    fn try_from(value: PathBuf) -> FileSystemResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a Path> for ExistingDir {
    type Error = FileSystemError;
    fn try_from(value: &'a Path) -> FileSystemResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a str> for ExistingDir {
    type Error = FileSystemError;
    fn try_from(value: &'a str) -> FileSystemResult<Self> {
        Self::new(value)
    }
}

impl From<ExistingDir> for PathBuf {
    fn from(value: ExistingDir) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn existing_dir_new_creation_scenarios() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();

        // Two scenarios: directory already exists vs. needs to be auto‑created.
        for auto_create in [false, true] {
            let target = if auto_create {
                base.join("child")
            } else {
                base.clone()
            };

            if auto_create {
                assert!(!target.exists(), "setup failure: dir should be absent");
            }

            let dir = super::ExistingDir::new(&target)?;
            assert!(dir.exists());
            assert!(dir.is_absolute());
        }
        Ok(())
    }

    #[test]
    fn existing_dir_rejects_file() -> FileSystemResult<()> {
        let tmp_file = tempfile::NamedTempFile::new().unwrap();

        assert!(matches!(
            super::ExistingDir::new(tmp_file.path()).unwrap_err(),
            FileSystemError::NotADirectory { .. }
        ));

        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn existing_dir_new_symlink_handling() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();

        // Prepare: a real directory and a real file.
        let dir_target = tmp.path().join("actual_dir");
        std::fs::create_dir(&dir_target).unwrap();
        let file_target = tmp.path().join("some_file");
        std::fs::File::create(&file_target).unwrap();

        // Create two symlinks: → dir (should succeed) and → file (should fail).
        let dir_link = tmp.path().join("dir_link");
        let file_link = tmp.path().join("file_link");
        std::os::unix::fs::symlink(&dir_target, &dir_link).unwrap();
        std::os::unix::fs::symlink(&file_target, &file_link).unwrap();

        for (link, expect_ok) in [(&dir_link, true), (&file_link, false)] {
            let res = super::ExistingDir::new(link);
            assert_eq!(res.is_ok(), expect_ok, "link: {}", link.display());

            if expect_ok {
                let dir = res?;
                assert_eq!(dir, super::ExistingDir::new(&dir_target)?);
            }
        }
        Ok(())
    }

    #[test]
    fn existing_dir_reset_behaviour() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dir = super::ExistingDir::new(tmp.path())?;

        // Test 1: Reset with existing directory containing files
        let trash = dir.join("trash.txt");
        std::fs::write(&trash, b"junk").unwrap();
        assert!(trash.exists());

        dir.reset()?;
        assert!(dir.exists());
        assert!(!trash.exists(), "file should be gone after reset");
        assert!(
            std::fs::read_dir(&*dir).unwrap().next().is_none(),
            "directory should be empty after reset"
        );

        // Test 2: Reset after directory was manually removed
        std::fs::remove_dir_all(&*dir).unwrap();
        assert!(!dir.exists());

        dir.reset()?; // Should succeed - remove() doesn't error on NotFound
        assert!(dir.exists());
        assert!(
            std::fs::read_dir(&*dir).unwrap().next().is_none(),
            "directory should be empty after reset"
        );

        Ok(())
    }

    #[test]
    fn existing_dir_remove_idempotent() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dir = super::ExistingDir::new(tmp.path())?;

        // First removal
        dir.remove()?;
        assert!(!dir.exists());

        // Second removal should also succeed (idempotent)
        dir.remove()?; // No error because NotFound is handled
        assert!(!dir.exists());

        Ok(())
    }

    #[test]
    fn existing_dir_remove_with_contents() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dir = super::ExistingDir::new(tmp.path())?;

        // Create nested structure
        let subdir = dir.join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("file.txt"), b"content").unwrap();
        std::fs::write(dir.join("root.txt"), b"root").unwrap();

        // Remove should delete everything
        dir.remove()?;
        assert!(!dir.exists());
        assert!(!subdir.exists());

        Ok(())
    }
}
