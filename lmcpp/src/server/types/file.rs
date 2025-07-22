//! Lmcpp – Valid‑Path Utilities
//! ===============================
//!
//! This module provides two tiny, zero‑cost new‑types—[`ValidDir`] and
//! [`ValidFile`]—that statically guarantee *at construction time* that a given
//! path is:
//!
//! * **Absolute & canonical** (`..` removed, symlinks resolved, platform‑
//!   specific case‑folding applied).
//! * **Type‑correct** (`ValidDir` → directory, `ValidFile` → regular file).
//!
//! By enforcing these invariants up front, the rest of your code can freely
//! assume that every `&Path` obtained from these wrappers is safe to pass to
//! the standard library without extra `std::fs::metadata` checks.
//!
//! ## Feature highlights
//!
//! * **Auto‑creation** – `ValidDir::new` transparently `mkdir -p`s missing
//!   ancestors, eliminating boiler‑plate.
//! * **Safety nets** – Attempting to wrap the *wrong* entry type (e.g. a
//!   symlink‑to‑dir inside `ValidFile::new`) yields an immediate `anyhow::Error`.
//! * **Utility helpers** – Methods such as [`ValidDir::reset`] and
//!   [`ValidFile::make_executable`] cover common filesystem chores.  

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::error::{LmcppError, LmcppResult};
/// A *canonical* path guaranteed to reference an **existing directory**.
///
/// ### Invariants
/// 1. Absolute, canonicalised (`Path::canonicalize`) representation.  
/// 2. `metadata().is_dir()` is `true`.  
///
/// Because the inner value satisfies these constraints *forever*, every borrow
/// obtained via [`Deref`] or [`AsRef<Path>`] is safe to hand to file‑system
/// APIs without further checks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ValidDir(pub PathBuf);

impl ValidDir {
    /// Construct a `ValidDir`, creating the directory tree if it does not yet
    /// exist (`mkdir ‑p` semantics).
    ///
    /// # Errors
    /// * I/O failures during canonicalisation or directory creation.  
    /// * The path exists but is **not** a directory.
    pub fn new<P: AsRef<Path>>(p: P) -> LmcppResult<Self> {
        let path = p.as_ref();

        let canonical = match path.canonicalize() {
            Ok(abs) => abs,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                std::fs::create_dir_all(path)
                    .map_err(|e| LmcppError::file_system("create dir", path, e))?;
                path.canonicalize()
                    .map_err(|e| LmcppError::file_system("canonicalise dir", path, e))?
            }
            Err(e) => return Err(LmcppError::file_system("canonicalise dir", path, e)),
        };

        if !canonical.is_dir() {
            return Err(LmcppError::file_system(
                "ValidDir is_dir failed",
                path,
                std::io::Error::from(ErrorKind::NotADirectory),
            ));
        }

        Ok(Self(canonical))
    }

    /// **Delete** the directory (recursively) *and recreate it empty*.
    ///
    /// Handy for “scratch” folders that must start from a clean slate at
    /// program start‑up or before a test case.
    pub fn reset(&self) -> LmcppResult<()> {
        match std::fs::remove_dir_all(&self.0) {
            Ok(_) => (),                                              // removed
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (), // already gone – fine
            Err(e) => {
                return Err(LmcppError::file_system("reset dir", &self.0, e));
            }
        }
        std::fs::create_dir_all(&self.0)
            .map_err(|e| LmcppError::file_system("recreate dir", &self.0, e))?;
        Ok(())
    }

    /// Permanently remove the directory and *do not* recreate it.
    pub fn remove(&self) -> LmcppResult<()> {
        std::fs::remove_dir_all(self)
            .map_err(|e| LmcppError::file_system("ValidDir remove dir", &self.0, e))?;

        debug_assert!(!self.exists());

        Ok(())
    }
}

impl std::ops::Deref for ValidDir {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for ValidDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl TryFrom<PathBuf> for ValidDir {
    type Error = LmcppError;
    fn try_from(value: PathBuf) -> LmcppResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a Path> for ValidDir {
    type Error = LmcppError;
    fn try_from(value: &'a Path) -> LmcppResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a str> for ValidDir {
    type Error = LmcppError;
    fn try_from(value: &'a str) -> LmcppResult<Self> {
        Self::new(value)
    }
}

/// A *canonical* path guaranteed to reference an **existing regular file**
/// (never a directory and never a symlink whose ultimate target is a
/// directory).
///
/// ### Invariants
/// 1. Absolute path **without** automatic symlink resolution—so a symlink
///    remains a symlink if that is what was supplied.
/// 2. The referenced filesystem entry is a *regular* file.  
///
/// All standard library APIs that accept `&Path` can therefore be used on a
/// `ValidFile` safely and without extra sanity checks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ValidFile(pub PathBuf);

impl ValidFile {
    pub fn new<P: AsRef<Path>>(p: P) -> LmcppResult<Self> {
        // 1. Make the path absolute *without* following symlinks
        let mut path = p.as_ref().to_path_buf();
        if !path.is_absolute() {
            path = std::env::current_dir()
                .map_err(|e| LmcppError::file_system("get current dir", &path, e))?
                .join(path);
        }

        // 2. Reject directories (whether they are real dirs or symlinks to dirs)
        let meta = std::fs::symlink_metadata(&path)
            .map_err(|e| LmcppError::file_system("fetch symlink metadata failed", &path, e))?;

        if meta.file_type().is_dir() {
            return Err(LmcppError::file_system(
                "ValidFile is_dir failed",
                &path,
                std::io::Error::from(ErrorKind::NotADirectory),
            ));
        }

        if meta.file_type().is_symlink() {
            let target_meta = std::fs::metadata(&path)
                .map_err(|e| LmcppError::file_system("fetch path metadata", &path, e))?;
            if target_meta.file_type().is_dir() {
                return Err(LmcppError::file_system(
                    "ValidFile is_symlink failed",
                    &path,
                    std::io::Error::from(ErrorKind::InvalidInput),
                ));
            }
        }

        Ok(Self(path))
    }

    /// On Unix platforms, change the mode bits to `755` (`rwxr‑xr‑x`).
    ///
    /// **No‑op on non‑Unix** targets so builds remain portable.
    pub fn make_executable(&self) -> LmcppResult<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self, std::fs::Permissions::from_mode(0o755))
                .map_err(|e| LmcppError::file_system("make executable", &self.0, e))?;
        }
        Ok(())
    }

    /// Recursively search `root_dir` for a file named `target` and return it
    /// wrapped as a `ValidFile`.
    ///
    /// Depth‑first traversal is used; the **first** match wins.
    ///
    /// # Errors
    /// * Propagates I/O errors encountered while traversing the tree.  
    /// * Returns a `bail!` message if `target` cannot be found.
    pub fn find_specific_file(root_dir: &ValidDir, target: &str) -> LmcppResult<ValidFile> {
        // Helper that works with plain `&Path` so recursion is inexpensive.
        fn walk(dir: &Path, target: &str) -> LmcppResult<ValidFile> {
            for entry in
                std::fs::read_dir(dir).map_err(|e| LmcppError::file_system("read dir", dir, e))?
            {
                let path = entry
                    .map_err(|e| LmcppError::file_system("read dir entry", dir, e))?
                    .path();

                if path.is_file()
                    && path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map_or(false, |n| n == target)
                {
                    // Validate once, then bubble the `ValidFile` upward.
                    return ValidFile::new(path);
                } else if path.is_dir() {
                    // Depth-first search: propagate success upward, ignore “not found” errors.
                    if let Ok(found) = walk(&path, target) {
                        return Ok(found);
                    }
                }
            }
            Err(LmcppError::file_system(
                "ValidDir find_specific_file failed",
                dir,
                std::io::Error::new(
                    ErrorKind::NotFound,
                    format!("Could not find `{}` in `{}`", target, dir.display()),
                ),
            ))
        }

        walk(root_dir.as_ref(), target)
    }
}

impl std::ops::Deref for ValidFile {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for ValidFile {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl TryFrom<PathBuf> for ValidFile {
    type Error = LmcppError;
    fn try_from(value: PathBuf) -> LmcppResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a Path> for ValidFile {
    type Error = LmcppError;
    fn try_from(value: &'a Path) -> LmcppResult<Self> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a str> for ValidFile {
    type Error = LmcppError;
    fn try_from(value: &'a str) -> LmcppResult<Self> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    /// Re‑export the crate’s result alias so every test can simply return it.
    use super::LmcppResult;
    /* ----------------------------------------------------------------- *
     *                    ValidDir::new – creation paths                  *
     * ----------------------------------------------------------------- */

    #[test]
    fn valid_dir_new_creation_scenarios() -> LmcppResult<()> {
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

            let dir = super::ValidDir::new(&target)?;
            assert!(dir.exists());
            assert!(dir.is_absolute());
        }
        Ok(())
    }

    #[test]
    fn valid_dir_rejects_file() -> LmcppResult<()> {
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let err = super::ValidDir::new(tmp_file.path()).unwrap_err();
        assert!(
            err.to_string().contains("ValidDir is_dir failed"),
            "err should contain 'ValidDir is_dir failed', but got: {}",
            err
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_dir_new_symlink_handling() -> LmcppResult<()> {
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
            let res = super::ValidDir::new(link);
            assert_eq!(res.is_ok(), expect_ok, "link: {}", link.display());

            if expect_ok {
                let dir = res?;
                assert_eq!(dir, super::ValidDir::new(&dir_target)?);
            }
        }
        Ok(())
    }

    /* ----------------------------------------------------------------- *
     *                     ValidDir::reset / remove                       *
     * ----------------------------------------------------------------- */

    #[test]
    fn valid_dir_reset_behaviour() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().to_path_buf();

        for remove_before_reset in [false, true] {
            let dir = super::ValidDir::new(&base)?;

            // Add junk so we can observe the clear‑out.
            let trash = dir.join("trash.txt");
            std::fs::write(&trash, b"junk").unwrap();
            assert!(trash.exists());

            if remove_before_reset {
                std::fs::remove_dir_all(&*dir).unwrap();
                assert!(!dir.exists(), "directory should be gone before reset()");
            }

            dir.reset()?;
            assert!(dir.exists());
            assert!(
                std::fs::read_dir(&*dir).unwrap().next().is_none(),
                "directory not empty after reset()"
            );
        }
        Ok(())
    }

    #[test]
    fn valid_dir_remove_success_and_not_found() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dir = super::ValidDir::new(tmp.path())?;

        // First removal succeeds.
        dir.remove()?;
        assert!(!dir.exists());

        // Second removal must error.
        let err = dir.remove().unwrap_err();
        assert!(err.to_string().contains("ValidDir remove dir"));
        Ok(())
    }

    /* ----------------------------------------------------------------- *
     *                           ValidFile::new                           *
     * ----------------------------------------------------------------- */

    #[test]
    fn valid_file_new_absolute_and_relative() -> LmcppResult<()> {
        // Absolute path
        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let file = super::ValidFile::new(tmp_file.path())?;
        assert!(file.exists());
        assert!(file.is_absolute());

        // Relative path
        let cwd = std::env::current_dir().unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        std::env::set_current_dir(tmp_dir.path()).unwrap();

        let rel_path = std::path::Path::new("rel_file.txt");
        std::fs::File::create(rel_path).unwrap();
        let rel_file = super::ValidFile::new(rel_path)?;
        assert!(rel_file.is_absolute());

        std::env::set_current_dir(cwd).unwrap();
        Ok(())
    }

    #[test]
    fn valid_file_new_errors() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();

        // Non‑existent file
        let no_exist = tmp.path().join("nope");
        assert!(super::ValidFile::new(&no_exist).is_err());

        // Directory supplied
        let err = super::ValidFile::new(tmp.path()).unwrap_err();
        assert!(
            err.to_string().contains("ValidFile is_dir failed"),
            "err should contain 'ValidFile is_dir failed', but got: {}",
            err
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_file_new_symlink_cases() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();

        // symlink → file (accepted)
        let tgt_file = tmp.path().join("target");
        std::fs::File::create(&tgt_file).unwrap();
        let ln_file = tmp.path().join("ln_file");
        std::os::unix::fs::symlink(&tgt_file, &ln_file).unwrap();
        super::ValidFile::new(&ln_file).unwrap(); // must succeed

        // symlink → dir (rejected)
        let tgt_dir = tmp.path().join("dir");
        std::fs::create_dir(&tgt_dir).unwrap();
        let ln_dir = tmp.path().join("ln_dir");
        std::os::unix::fs::symlink(&tgt_dir, &ln_dir).unwrap();
        let err = super::ValidFile::new(&ln_dir).unwrap_err();
        assert!(
            err.to_string().contains("ValidFile is_symlink failed"),
            "err should contain 'ValidFile is_symlink failed', but got: {}",
            err
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_file_make_executable() -> LmcppResult<()> {
        use std::os::unix::fs::PermissionsExt;

        let tmp_file = tempfile::NamedTempFile::new().unwrap();
        let vf = super::ValidFile::new(tmp_file.path())?;

        vf.make_executable()?;
        let mode = std::fs::metadata(&*vf).unwrap().permissions().mode();

        // Expect exactly rwxr‑xr‑x irrespective of prior permissions.
        assert_eq!(mode & 0o777, 0o755);
        Ok(())
    }

    /* ----------------------------------------------------------------- *
     *                  ValidFile::find_specific_file                     *
     * ----------------------------------------------------------------- */

    #[test]
    fn find_specific_file_root_and_nested() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let root_dir = super::ValidDir::new(tmp.path())?;

        // Root‑level hit
        let root_target = root_dir.join("hit.txt");
        std::fs::File::create(&root_target).unwrap();
        let found = super::ValidFile::find_specific_file(&root_dir, "hit.txt").unwrap();
        assert_eq!(found.as_ref(), &root_target);

        // Nested hit (depth‑first)
        let sub = root_dir.join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();
        let nested_target = sub.join("deep.txt");
        std::fs::File::create(&nested_target).unwrap();

        let found_nested = super::ValidFile::find_specific_file(&root_dir, "deep.txt")?;
        assert_eq!(found_nested.as_ref(), &nested_target);
        Ok(())
    }

    #[test]
    fn find_specific_file_not_found() -> LmcppResult<()> {
        let tmp = tempfile::tempdir().unwrap();
        let root_dir = super::ValidDir::new(tmp.path())?;
        let err = super::ValidFile::find_specific_file(&root_dir, "ghost").unwrap_err();
        assert!(
            err.to_string()
                .contains("ValidDir find_specific_file failed"),
            "err should contain 'ValidDir find_specific_file failed', but got: {}",
            err
        );
        Ok(())
    }

    /* ----------------------------------------------------------------- *
     *                          TryFrom round‑trip                       *
     * ----------------------------------------------------------------- */

    #[test]
    fn try_from_round_trip_variants() -> LmcppResult<()> {
        // Directories
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir_path = tmp_dir.path();
        let dir_str = dir_path.to_string_lossy();

        let dir_variants: Vec<super::ValidDir> = vec![
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

        let file_variants: Vec<super::ValidFile> = vec![
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
