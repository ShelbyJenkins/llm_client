//! File Path Types
//! ===============
//!
//! This module provides type-safe wrappers for file paths that guarantee certain
//! filesystem properties at construction time.
//!
//! [`ExistingFile`] ensures that a path:
//! * Points to an existing regular file (not a directory, device, or socket)
//! * Is represented as an absolute path
//!
//! The module also provides extension-specific wrappers (e.g., [`JsonPath`], [`GgufPath`])
//! that additionally verify the file has the expected extension.
//!
//! By enforcing these invariants at construction, the rest of your code can safely
//! assume these properties hold without repeated filesystem checks.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fs::{
    dir::ExistingDir,
    error::{FileSystemError, FileSystemResult},
};

macro_rules! existing_file_wrappers {
    (
        $(
            $(#[$docs:meta])*
            $name:ident => $ext:literal
        ),* $(,)?
    ) => {
        $(
            $(#[$docs])*
            #[derive(
                Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash,
                serde::Serialize, serde::Deserialize
            )]
            #[serde(transparent)]
            #[repr(transparent)]
            pub struct $name(pub ExistingFile);

            impl $name {
                /// Expected file‑name suffix (without the “.”).
                pub const EXT: &'static str = $ext;

                /// Creates the wrapper after verifying the file exists and has
                /// the correct extension ([`Self::EXT`]).
                pub fn new<P: AsRef<std::path::Path>>(p: P) -> FileSystemResult<Self> {
                    let path = p.as_ref();
                    if path.extension().and_then(std::ffi::OsStr::to_str) != Some(Self::EXT) {
                        return Err(FileSystemError::incorrect_extension(path, Self::EXT));
                    }
                    Ok(Self(ExistingFile::try_new(path)?))
                }

                /// Recursively finds a file with the expected extension (delegates to [`ExistingFile::find_specific_file`]).
                pub fn find_specific_file(root_dir: &ExistingDir, target: &str) -> FileSystemResult<Self> {
                    ExistingFile::find_specific_file(root_dir, target)
                        .and_then(Self::from_existing_file)
                }

                /// Returns the underlying path as a `&Path`.
                #[inline]
                pub fn as_path(&self) -> &std::path::Path {
                    self.0.as_path()
                }

                /// Makes the file executable (delegates to [`ExistingFile::make_executable`]).
                pub fn make_executable(&self) -> FileSystemResult<()> {
                    self.0.make_executable()
                }

                /// Converts an `ExistingFile` to this type after validating the extension.
                fn from_existing_file(file: ExistingFile) -> FileSystemResult<Self> {
                    if file.extension().and_then(std::ffi::OsStr::to_str) != Some(Self::EXT) {
                        return Err(FileSystemError::incorrect_extension(&*file, Self::EXT));
                    }
                    Ok(Self(file))
                }
            }

            impl std::ops::Deref for $name {
                type Target = Path;
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }

            impl AsRef<Path> for $name {
                fn as_ref(&self) -> &Path {
                    &self.0
                }
            }

            impl TryFrom<PathBuf> for $name {
                type Error = FileSystemError;
                fn try_from(value: PathBuf) -> FileSystemResult<Self> {
                    Self::new(value)
                }
            }

            impl<'a> TryFrom<&'a Path> for $name {
                type Error = FileSystemError;
                fn try_from(value: &'a Path) -> FileSystemResult<Self> {
                    Self::new(value)
                }
            }

            impl<'a> TryFrom<&'a str> for $name {
                type Error = FileSystemError;
                fn try_from(value: &'a str) -> FileSystemResult<Self> {
                    Self::new(value)
                }
            }

            impl From<$name> for PathBuf {
                fn from(value: $name) -> Self {
                    value.0.0
                }
            }

        )*
    };
}

existing_file_wrappers! {
    /// A path to a JSON file (must have .json extension).
    JsonPath => "json",

    /// A path to a GGUF model file (must have .gguf extension).
    GgufPath => "gguf",

    /// A path to a ZIP archive file (must have .zip extension).
    ZipPath => "zip",
}

/// A path that is guaranteed to point to an existing regular file.
///
/// ### Guarantees
/// 1. The file exists on the filesystem at the time of creation
/// 2. It is a regular file (not a directory, device, socket, etc.)
/// 3. The path is absolute
///
/// Note: This type validates the path and filesystem state, not the file contents.
/// A `JsonPath` guarantees a `.json` extension exists, but not that the file
/// contains valid JSON.
/// `FileSystemError` is used to report any issues encountered during validation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ExistingFile(pub PathBuf);

impl ExistingFile {
    /// Creates an `ExistingFile` from a path, verifying it points to a regular file.
    ///
    /// The path will be made absolute if it isn't already. Symlinks are allowed
    /// but must ultimately point to a regular file.
    ///
    /// # Errors
    /// * `NotFound` - The file doesn't exist
    /// * `NotRegularFile` - The path exists but isn't a regular file
    /// * `PermissionDenied` - Insufficient permissions to access the file
    /// * `IoError` - Other I/O errors during validation
    pub fn try_new<P: AsRef<Path>>(p: P) -> FileSystemResult<Self> {
        let path = Self::make_absolute(p.as_ref())?;
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| FileSystemError::from_io_error("fetch symlink metadata", &path, e))?;

        Self::new_with_metadata(path, meta)
    }

    /// Recursively searches for a file with the given name within a directory tree.
    ///
    /// Performs a depth-first search, returning the first matching file found.
    ///
    /// # Arguments
    /// * `root_dir` - The directory to search within
    /// * `target` - The filename to search for (e.g., "config.json")
    ///
    /// # Errors
    /// * `FileNotFoundInDir` - The target file wasn't found in the directory tree
    /// * `IoError` - I/O errors during directory traversal
    /// * Other errors from `ExistingFile::new` if the found file fails validation
    pub fn find_specific_file(
        root_dir: &ExistingDir,
        target: &str,
    ) -> FileSystemResult<ExistingFile> {
        fn walk(dir: &Path, target: &str) -> Option<FileSystemResult<ExistingFile>> {
            let entries = match std::fs::read_dir(dir) {
                Ok(entries) => entries,
                Err(e) => {
                    return Some(Err(FileSystemError::from_io_error(
                        "read directory",
                        dir,
                        e,
                    )));
                }
            };

            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(e) => {
                        return Some(Err(FileSystemError::from_io_error(
                            "read directory entry",
                            dir,
                            e,
                        )));
                    }
                };

                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(e) => {
                        return Some(Err(FileSystemError::from_io_error("get file type", dir, e)));
                    }
                };

                // Skip non-files early
                if !file_type.is_file() && !file_type.is_symlink() {
                    if file_type.is_dir() {
                        // Recurse into directories
                        if let Some(result) = walk(&entry.path(), target) {
                            return Some(result);
                        }
                    }
                    continue;
                }

                // Check filename
                if entry.file_name().to_str() == Some(target) {
                    let path = entry.path();
                    let meta = match entry.metadata() {
                        Ok(m) => m,
                        Err(e) => {
                            return Some(Err(FileSystemError::from_io_error(
                                "read file metadata",
                                &path,
                                e,
                            )));
                        }
                    };

                    let abs_path = match ExistingFile::make_absolute(&path) {
                        Ok(p) => p,
                        Err(e) => return Some(Err(e)),
                    };

                    return Some(ExistingFile::new_with_metadata(abs_path, meta));
                }
            }

            None
        }

        walk(root_dir.as_ref(), target).unwrap_or_else(|| {
            Err(FileSystemError::FileNotFoundInDir {
                root_dir: root_dir.as_ref().to_path_buf(),
                target: target.to_string(),
            })
        })
    }

    /// Returns the underlying path as a `&Path`.
    #[inline]
    pub fn as_path(&self) -> &std::path::Path {
        &self.0
    }

    /// Makes the file executable on Unix platforms (chmod 755).
    ///
    /// This is a no-op on non-Unix platforms to maintain portability.
    ///
    /// # Errors
    /// * `PermissionDenied` - Insufficient permissions to modify the file
    /// * `IoError` - Other I/O errors
    pub fn make_executable(&self) -> FileSystemResult<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self, std::fs::Permissions::from_mode(0o755)).map_err(
                |e| FileSystemError::PermissionDenied {
                    path: self.0.clone(),
                    message: e.to_string(),
                },
            )?;
        }
        Ok(())
    }

    /// Creates an `ExistingFile` from a path and pre-fetched metadata.
    ///
    /// This avoids an additional stat call when metadata is already available.
    fn new_with_metadata(path: PathBuf, meta: std::fs::Metadata) -> FileSystemResult<Self> {
        Self::validate_metadata(&path, &meta)?;
        Ok(Self(path))
    }

    /// Converts a relative path to absolute without performing validation.
    ///
    /// # Errors
    /// * `IoError` - If the current directory cannot be determined
    fn make_absolute(path: &Path) -> FileSystemResult<PathBuf> {
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            Ok(std::env::current_dir()
                .map_err(|e| FileSystemError::from_io_error("get current dir", path, e))?
                .join(path))
        }
    }

    /// Validates that metadata represents a regular file.
    ///
    /// Also checks that symlinks (if present) point to regular files.
    fn validate_metadata(path: &Path, meta: &std::fs::Metadata) -> FileSystemResult<()> {
        if meta.file_type().is_symlink() {
            // For symlinks, check what they point to
            let target_meta = std::fs::metadata(path)
                .map_err(|e| FileSystemError::from_io_error("read symlink target", path, e))?;
            if !target_meta.is_file() {
                return Err(FileSystemError::NotRegularFile {
                    path: path.to_path_buf(),
                });
            }
        } else if !meta.is_file() {
            // For non-symlinks, must be a regular file
            return Err(FileSystemError::NotRegularFile {
                path: path.to_path_buf(),
            });
        }

        Ok(())
    }
}

impl std::ops::Deref for ExistingFile {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for ExistingFile {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl TryFrom<PathBuf> for ExistingFile {
    type Error = FileSystemError;
    fn try_from(value: PathBuf) -> FileSystemResult<Self> {
        Self::try_new(value)
    }
}

impl<'a> TryFrom<&'a Path> for ExistingFile {
    type Error = FileSystemError;
    fn try_from(value: &'a Path) -> FileSystemResult<Self> {
        Self::try_new(value)
    }
}

impl<'a> TryFrom<&'a str> for ExistingFile {
    type Error = FileSystemError;
    fn try_from(value: &'a str) -> FileSystemResult<Self> {
        Self::try_new(value)
    }
}

impl From<ExistingFile> for PathBuf {
    fn from(value: ExistingFile) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    /// Re‑export the crate’s result alias so every test can simply return it.
    use super::*;

    #[test]
    fn valid_file_new_absolute_and_relative() -> FileSystemResult<()> {
        // Absolute path
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let file = ExistingFile::try_new(tmp_file.path())?;
        assert!(file.exists());
        assert!(file.is_absolute());

        // Relative path
        let cwd = std::env::current_dir().unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        std::env::set_current_dir(tmp_dir.path()).unwrap();

        let rel_path = std::path::Path::new("rel_file.txt");
        std::fs::File::create(rel_path).unwrap();
        let rel_file = ExistingFile::try_new(rel_path)?;
        assert!(rel_file.is_absolute());

        std::env::set_current_dir(cwd).unwrap();
        Ok(())
    }

    #[test]
    fn valid_file_new_errors() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        // Non‑existent file
        let no_exist = tmp.path().join("nope");
        assert!(matches!(
            ExistingFile::try_new(&no_exist).unwrap_err(),
            FileSystemError::NotFound { .. }
        ));

        // Directory supplied
        assert!(matches!(
            ExistingFile::try_new(tmp.path()).unwrap_err(),
            FileSystemError::NotRegularFile { .. }
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_file_new_symlink_cases() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();

        // symlink → file (accepted)
        let tgt_file = tmp.path().join("target");
        std::fs::File::create(&tgt_file).unwrap();
        let ln_file = tmp.path().join("ln_file");
        std::os::unix::fs::symlink(&tgt_file, &ln_file).unwrap();
        ExistingFile::try_new(&ln_file).unwrap(); // must succeed

        // symlink → dir (rejected)
        let tgt_dir = tmp.path().join("dir");
        std::fs::create_dir(&tgt_dir).unwrap();
        let ln_dir = tmp.path().join("ln_dir");
        std::os::unix::fs::symlink(&tgt_dir, &ln_dir).unwrap();

        assert!(matches!(
            ExistingFile::try_new(&ln_dir).unwrap_err(),
            FileSystemError::NotRegularFile { .. }
        ));

        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_file_make_executable() -> FileSystemResult<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let vf = ExistingFile::try_new(tmp_file.path())?;

        vf.make_executable()?;
        let mode = std::fs::metadata(&*vf).unwrap().permissions().mode();

        // Expect exactly rwxr‑xr‑x irrespective of prior permissions.
        assert_eq!(mode & 0o777, 0o755);
        Ok(())
    }

    #[test]
    fn find_specific_file_root_and_nested() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let root_dir = ExistingDir::new(tmp.path())?;

        // Root‑level hit
        let root_target = root_dir.join("hit.txt");
        std::fs::File::create(&root_target).unwrap();
        let found = ExistingFile::find_specific_file(&root_dir, "hit.txt").unwrap();
        assert_eq!(found.as_ref(), &root_target);

        // Nested hit (depth‑first)
        let sub = root_dir.join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();
        let nested_target = sub.join("deep.txt");
        std::fs::File::create(&nested_target).unwrap();

        let found_nested = ExistingFile::find_specific_file(&root_dir, "deep.txt")?;
        assert_eq!(found_nested.as_ref(), &nested_target);
        Ok(())
    }

    #[test]
    fn find_specific_file_not_found() -> FileSystemResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let root_dir = ExistingDir::new(tmp.path())?;

        assert!(matches!(
            ExistingFile::find_specific_file(&root_dir, "ghost").unwrap_err(),
            FileSystemError::FileNotFoundInDir { target, .. } if target == "ghost"
        ));

        Ok(())
    }

    #[test]
    fn try_from_round_trip_variants() -> FileSystemResult<()> {
        // Directories
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir_path = tmp_dir.path();
        let dir_str = dir_path.to_string_lossy();

        let dir_variants: Vec<ExistingDir> = vec![
            std::convert::TryFrom::try_from(dir_str.as_ref())?,
            std::convert::TryFrom::try_from(dir_path)?,
            std::convert::TryFrom::try_from(dir_path.to_path_buf())?,
        ];

        for a in &dir_variants {
            for b in &dir_variants {
                assert_eq!(a, b);
            }
        }

        // Files
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let file_path = tmp_file.path();
        let file_str = file_path.to_string_lossy();

        let file_variants: Vec<ExistingFile> = vec![
            std::convert::TryFrom::try_from(file_str.as_ref())?,
            std::convert::TryFrom::try_from(file_path)?,
            std::convert::TryFrom::try_from(file_path.to_path_buf())?,
        ];

        for a in &file_variants {
            for b in &file_variants {
                assert_eq!(a, b);
            }
        }
        Ok(())
    }
}
