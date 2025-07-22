use std::{
    path::Path,
    process::{Command, Stdio},
};

use crate::error::{LmcppError, LmcppResult};

pub fn cmake_is_available() -> LmcppResult<()> {
    //   Locate the executable
    let out = std::process::Command::new("cmake")
        .arg("--version")
        .output()
        .map_err(|e| LmcppError::FileSystem {
            operation: "`cmake --version`",
            path: std::path::PathBuf::from("cmake"),
            source: e,
        })?;

    if !out.status.success() {
        return Err(LmcppError::BuildFailed(
            "`cmake --version` returned non-zero status".into(),
        ));
    }
    // Parse the version string (first line looks like `cmake version 3.25.4`)
    let stdout_content = String::from_utf8_lossy(&out.stdout);
    let version_line = stdout_content.lines().next().unwrap_or_default();
    let ver = version_line.split_whitespace().nth(2).unwrap_or("0.0.0");
    let mut parts = ver.split('.');
    let major = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    let minor = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);

    crate::trace!("CMake detected: {}", version_line);

    // Enforce the minimum required feature level (≥ 3.15)
    const MIN_MAJOR: u32 = 3;
    const MIN_MINOR: u32 = 15;
    if (major, minor) < (MIN_MAJOR, MIN_MINOR) {
        Err(LmcppError::InvalidConfig {
            field: "CMake",
            reason: "requires ≥ 3.15".into(),
        })
    } else {
        Ok(())
    }
}

/// Verifies that a usable `curl` binary is on the `PATH`.
/// Returns `Ok(())` on success or a descriptive error otherwise.
pub fn curl_is_available() -> LmcppResult<()> {
    let out = std::process::Command::new("curl")
        .arg("--version")
        .output()
        .map_err(|e| LmcppError::FileSystem {
            operation: "spawn curl --version",
            path: std::path::PathBuf::from("curl"),
            source: e,
        })?;

    if out.status.success() {
        Ok(())
    } else {
        Err(LmcppError::BuildFailed("`curl --version` returned non-zero status".into()).into())
    }
}

pub fn cmake_project_buildsystem(working_dir: &Path, build_args: &[&str]) -> LmcppResult<()> {
    assert!(
        working_dir.is_dir(),
        "cmake_project_buildsystem: {} is not a directory",
        working_dir.display()
    );
    let mut unique_build_args: Vec<&str> = build_args.to_vec();
    unique_build_args.sort();
    unique_build_args.dedup();
    assert_eq!(
        unique_build_args.len(),
        build_args.len(),
        "build_args contains duplicate entries"
    );

    let mut cmd = Command::new("cmake");
    cmd.arg("-B").arg("build");
    if !build_args.is_empty() {
        cmd.args(build_args);
    }

    cmd.current_dir(working_dir);
    crate::trace!("Running cmake command: {:?}", cmd);
    let out = cmd.output().map_err(|e| LmcppError::FileSystem {
        operation: "spawn cmake -B build",
        path: working_dir.to_path_buf(),
        source: e,
    })?;
    if out.status.success() {
        Ok(())
    } else {
        Err(LmcppError::BuildFailed(format!(
            "CMake failed:\nCommand: {:?}\nExit code {:?}\nstderr: {}",
            cmd,
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )))
    }
}

pub fn cmake_build_project(working_dir: &Path) -> LmcppResult<()> {
    assert!(
        working_dir.is_dir(),
        "cmake_build_project: {} is not a directory",
        working_dir.display()
    );

    let num_jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut cmd = Command::new("cmake");
    cmd.arg("--build")
        .arg("build")
        .arg("--config")
        .arg("Release")
        .arg("-j")
        .arg(num_jobs.to_string())
        .arg("-t")
        .arg("llama-server")
        .current_dir(working_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    crate::trace!("Running cmake command: {:?}", cmd);

    // 3. Run it once and stream output live
    let status = cmd.status().map_err(|e| LmcppError::FileSystem {
        operation: "spawn cmake --build",
        path: working_dir.to_path_buf(),
        source: e,
    })?;

    if !status.success() {
        if let Some(code) = status.code() {
            return Err(LmcppError::BuildFailed(format!(
                "cmake --build failed (exit code {code}). \
                   Scroll up for the first compiler/linker error."
            )));
        } else {
            return Err(LmcppError::BuildFailed(format!(
                "cmake --build was terminated by a signal"
            )));
        }
    }

    crate::trace!("cmake completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit‑tests for the helpers in `cmake.rs`.
    //!
    //! They work by fabricating a tiny “cmake” wrapper that prints the desired
    //! version string and putting the directory that contains that wrapper at
    //! the front of `PATH`.  
    //! A global mutex guarantees only one test manipulates `PATH` at a time.

    use std::sync::{Mutex, OnceLock};

    use serial_test::serial;

    /// Global lock so `PATH` tweaks never race.
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    /// Create a fake **cmake** executable that reports the supplied `ver`.
    ///
    /// The stub is written into `dir` and returned.
    #[cfg(unix)]
    fn stub_cmake(dir: &std::path::Path, ver: &str) -> std::path::PathBuf {
        let path = dir.join("cmake");
        std::fs::write(&path, format!("#!/bin/sh\necho \"cmake version {ver}\"\n")).unwrap();
        // make it executable
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    #[cfg(windows)]
    fn stub_cmake(dir: &std::path::Path, ver: &str) -> std::path::PathBuf {
        use std::process::Command;

        let src = dir.join("stub.rs");
        std::fs::write(
            &src,
            format!(r#"fn main() {{ println!("cmake version {ver}"); }}"#),
        )
        .unwrap();

        let exe = dir.join("cmake.exe");
        Command::new("rustc")
            .args([src.to_str().unwrap(), "-O", "-o", exe.to_str().unwrap()])
            .status()
            .unwrap();

        exe
    }

    /// Helper: prepend `new_dir` to `PATH`, returning the *previous* value so
    /// that it can later be restored.
    fn prepend_path(new_dir: &std::path::Path) -> Option<std::ffi::OsString> {
        let old = std::env::var_os("PATH");
        let sep = if cfg!(windows) { ";" } else { ":" };
        let new_path = match &old {
            Some(val) => format!("{}{}{}", new_dir.display(), sep, val.to_string_lossy()),
            None => new_dir.display().to_string(),
        };
        unsafe {
            std::env::set_var("PATH", &new_path);
        }
        old
    }

    // ───────────────────────── cmake_is_available ──────────────────────────

    #[test]
    #[serial]
    fn cmake_is_available_parses_valid_version() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stub_cmake(tmp.path(), "3.27.1");
        let old = prepend_path(tmp.path());

        let result = super::cmake_is_available();

        // restore PATH
        if let Some(v) = old {
            unsafe { std::env::set_var("PATH", v) }
        } else {
            // SAFETY: Removing PATH is safe in this test context as we're only
            // temporarily modifying it and restoring it after the test completes
            unsafe { std::env::remove_var("PATH") }
        }

        assert!(
            result.is_ok(),
            "Expected Ok(()) for version ≥ 3.15, got {result:?}"
        );
    }

    #[test]
    #[serial]
    fn cmake_is_available_fails_old_version() {
        let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stub_cmake(tmp.path(), "3.10.0");
        let old = prepend_path(tmp.path());

        let result = super::cmake_is_available();

        if let Some(v) = old {
            unsafe { std::env::set_var("PATH", v) }
        } else {
            unsafe { std::env::remove_var("PATH") }
        }

        assert!(
            result.is_err(),
            "Expected Err(..) for version < 3.15, got {result:?}"
        );
    }

    // ─────────────────── cmake_project_buildsystem assertions ──────────────

    #[test]
    #[serial]
    fn cmake_project_buildsystem_detects_duplicates() {
        let repo_dir = tempfile::tempdir().unwrap();
        let outcome = std::panic::catch_unwind(|| {
            super::cmake_project_buildsystem(
                repo_dir.path(),
                &["-DFOO=ON", "-DFOO=ON"], /* duplicates */
            )
            .unwrap();
        });

        assert!(
            outcome.is_err(),
            "Passing duplicate flags must trigger the internal `assert!`"
        );
    }

    #[test]
    #[serial]
    fn cmake_project_buildsystem_rejects_non_dir_path() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let outcome = std::panic::catch_unwind(|| {
            super::cmake_project_buildsystem(temp_file.path(), &[]).unwrap();
        });

        assert!(
            outcome.is_err(),
            "Providing a non‑directory path must panic via the pre‑condition"
        );
    }
}
