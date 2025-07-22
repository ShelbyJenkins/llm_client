//! Server Process - PID
//! =================================
//!
//! A small collection of utilities for **PID‑file management** and **process
//! discovery**.
//!
//! ```text
//!  ┌─ disk • pidfile helpers ───────────────────────┐
//!  │ pidfile_path()      → deterministic file name  │
//!  │ create_pidfile()    → exclusive create (+chmod)│
//!  │ pid_from_pidfile()  → read & parse u32         │
//!  │ discover_pidfiles() → scan directory           │
//!  └────────────────────────────────────────────────┘
//!
//!  ┌─ memory • process scanning ────────────────────┐
//!  │ get_all_server_pids()     → by executable name │
//!  │ get_server_pid_by_cmd_args() → by argv slices  │
//!  └────────────────────────────────────────────────┘
//!
//!  ┌─ kernel • liveness probe ──────────────────────┐
//!  │ pid_alive() → cross‑platform “is this pid up?” │
//!  └────────────────────────────────────────────────┘
//! ```
//!
//! These helpers are **synchronous, side‑effect free** (except
//! `create_pidfile`) and meant for CLI start‑up / shutdown
//! housekeeping rather than the performance‑critical path.

use std::{
    ffi::OsString,
    fs::OpenOptions,
    path::{Path, PathBuf},
};

use sysinfo::{ProcessRefreshKind, RefreshKind, UpdateKind};

use super::error::*;

pub fn create_pidfile(pidfile_path: &Path) -> Result<std::fs::File> {
    if pidfile_path.exists() {
        return Err(ProcessError::CommandFailed {
            action: "check pidfile existence",
            source: format!("pidfile {pidfile_path:?} already exists").into(),
        });
    }

    match pidfile_path.parent() {
        Some(parent) => {
            if !parent.exists() {
                match std::fs::create_dir_all(parent) {
                    Ok(_) => (),
                    Err(e) => {
                        return Err(ProcessError::CommandFailed {
                            action: "create pidfile parent directory",
                            source: e.into(),
                        });
                    }
                }
            }
        }
        None => {
            return Err(ProcessError::CommandFailed {
                action: "determine pidfile parent directory",
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "pidfile has no parent directory",
                )
                .into(),
            });
        }
    }
    let mut options = OpenOptions::new();

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o644); // unix-only: sane perms
    }

    options
        .write(true)
        .create_new(true) // fail if already exists
        .open(pidfile_path)
        .map_err(|e| ProcessError::CommandFailed {
            action: "create pidfile",
            source: e.into(),
        })
}

pub fn pid_from_pidfile(pidfile_path: &Path) -> Result<u32> {
    let content =
        std::fs::read_to_string(pidfile_path).map_err(|e| ProcessError::CommandFailed {
            action: "read pidfile",
            source: e.into(),
        })?;
    content
        .trim()
        .parse::<u32>()
        .map_err(|e| ProcessError::CommandFailed {
            action: "parse pid from pidfile",
            source: e.into(),
        })
}

pub fn discover_pidfiles(bin_dir: &Path, executable_name: &str) -> Vec<(String, u32, PathBuf)> {
    let prefix = format!("{executable_name}_");
    let suffix = ".pid";

    let mut out = Vec::new();

    // `read_dir` may fail – return an empty vec in that case
    let entries = match std::fs::read_dir(bin_dir) {
        Ok(rd) => rd,
        Err(_) => return out,
    };

    for entry_res in entries {
        // skip individual IO errors
        let entry = match entry_res {
            Ok(e) => e,
            Err(_) => continue,
        };

        // ── FIX: make the lossy conversion live long enough ──────────────
        let name_owned: String = entry.file_name().to_string_lossy().into_owned(); // now we own it
        let file_str = name_owned.as_str(); // &str that lives

        if !(file_str.starts_with(&prefix) && file_str.ends_with(suffix)) {
            continue;
        }

        let host = file_str[prefix.len()..file_str.len() - suffix.len()].to_owned();

        let pid = std::fs::read_to_string(entry.path())
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());

        if let Some(pid) = pid {
            out.push((host, pid, entry.path()));
        }
    }

    out
}

pub fn get_all_server_pids(executable_name: &str) -> Vec<u32> {
    let sys = sysinfo::System::new_with_specifics(
        RefreshKind::nothing()
            .with_processes(sysinfo::ProcessRefreshKind::nothing().with_exe(UpdateKind::Always)),
    );

    let want_name_prefix = &executable_name.as_bytes()[..15.min(executable_name.len())];

    sys.processes()
        .values()
        .filter_map(|p| {
            // Fallback: compare first 15 bytes of the name
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::ffi::OsStrExt;

                if p.name().as_bytes() == want_name_prefix {
                    return Some(p.pid().as_u32());
                }
            }
            #[cfg(any(windows, target_os = "macos"))]
            if p.name().as_encoded_bytes() == want_name_prefix {
                return Some(p.pid().as_u32());
            }

            None
        })
        .collect()
}

pub fn get_server_pid_by_cmd_args<'a>(patterns: &[&[&'a str]]) -> Option<u32> {
    fn argv_contains_sequence(cmd: &[OsString], seq: &[&str]) -> bool {
        // ── flatten argv ────────────────────────────────────────────────────
        let flat: Vec<String> = cmd
            .iter()
            .flat_map(|arg| {
                arg.to_string_lossy()
                    .split_whitespace() // ⇒ "/NOBREAK", "&REM", "--test-flag=value"
                    .flat_map(|tok| {
                        if let Some((flag, val)) = tok.split_once('=') {
                            vec![flag.to_owned(), val.to_owned()] // "--test-flag", "value"
                        } else {
                            vec![tok.to_owned()] // unchanged token
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // ── sliding‑window search ───────────────────────────────────────────
        flat.windows(seq.len())
            .any(|w| seq.iter().zip(w).all(|(pat, tok)| pat == tok))
    }

    // ── gather process list ────────────────────────────────────────────────

    let sys = sysinfo::System::new_with_specifics(
        RefreshKind::nothing().with_processes(
            ProcessRefreshKind::nothing()
                .with_cmd(UpdateKind::Always)
                .with_exe(UpdateKind::Always),
        ),
    );

    // ── scan processes (no owner filtering) ────────────────────────────────
    for p in sys.processes().values() {
        if patterns
            .iter()
            .any(|pat| argv_contains_sequence(p.cmd(), pat))
        {
            return Some(p.pid().as_u32());
        }
    }

    None
}

#[cfg(unix)]
pub fn pid_alive(pid: u32) -> Result<bool> {
    use nix::{errno::Errno, sys::signal::kill, unistd::Pid};

    // ① Does the kernel still know this PID?
    match kill(Pid::from_raw(pid as i32), None) {
        Err(Errno::ESRCH) => return Ok(false), // completely gone
        Err(Errno::EPERM) => {
            return Err(ProcessError::PermissionDenied {
                action: "probe process (signal 0)",
                source: "operation not permitted".into(),
            });
        }
        Err(e) => {
            return Err(ProcessError::CommandFailed {
                action: "probe process (signal 0)",
                source: e.into(),
            });
        }
        Ok(_) => (), // alive, or at least not dead
    }

    // ② On Linux: zombie == not alive
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string(format!("/proc/{pid}/status")) {
            if status
                .lines()
                .find(|l| l.starts_with("State:"))
                .and_then(|l| l.split_whitespace().nth(1))
                == Some("Z")
            {
                return Ok(false); // defunct, will never run again
            }
        }
    }

    // ③ Anything else (running, sleeping, unidentifiable) is considered alive
    Ok(true)
}

#[cfg(windows)]
pub fn pid_alive(pid: u32) -> Result<bool> {
    use windows::{
        Win32::{
            Foundation::{CloseHandle, ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER, WAIT_TIMEOUT},
            System::Threading::{
                OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
                WaitForSingleObject,
            },
        },
        core::HRESULT,
    };

    unsafe {
        // Try to open the process with minimal rights
        match OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
            false,
            pid,
        ) {
            Ok(handle) => {
                // Successfully opened the process
                let status = WaitForSingleObject(handle, 0);
                let _ = CloseHandle(handle);

                // WAIT_TIMEOUT means the process is still running
                Ok(status == WAIT_TIMEOUT)
            }
            Err(e) => match e.code() {
                hr if hr == HRESULT::from(ERROR_INVALID_PARAMETER) => Ok(false),
                hr if hr == HRESULT::from(ERROR_ACCESS_DENIED) => {
                    Err(ProcessError::PermissionDenied {
                        action: "probe process (OpenProcess)",
                        source: Box::new(e),
                    })
                }
                _ => Err(ProcessError::CommandFailed {
                    action: "probe process (OpenProcess)",
                    source: Box::new(e),
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{path::Path, thread, time::Duration};

    use super::*;
    use crate::server::process::tests_helpers::*;

    #[test]
    fn get_all_server_pids_name_fallback() {
        let mut child = long_cmd().spawn().unwrap();
        let pid = child.id();

        #[cfg(unix)]
        let exe_name = "sleep";
        #[cfg(windows)]
        let exe_name = "timeout.exe";

        let pids = super::get_all_server_pids(exe_name);
        assert!(pids.contains(&pid));

        let _ = child.kill();
    }

    #[test]
    fn get_all_server_pids_canonical_path_match() {
        #[cfg(unix)]
        {
            let abs = Path::new("/bin/sleep");
            if abs.exists() {
                let mut child = std::process::Command::new(abs).arg("5").spawn().unwrap();
                let pid = child.id();
                assert!(super::get_all_server_pids("sleep").contains(&pid));
                let _ = child.kill();
            }
        }

        #[cfg(windows)]
        {
            let root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".into());
            let abs = Path::new(&root).join("System32").join("timeout.exe");
            if abs.exists() {
                let mut child = std::process::Command::new(&abs)
                    .args(["/T", "5", "/NOBREAK"])
                    .spawn()
                    .unwrap();
                let pid = child.id();
                assert!(super::get_all_server_pids("timeout.exe").contains(&pid));
                let _ = child.kill();
            }
        }
    }

    #[test]
    fn test_get_server_pid_by_cmd_args_cases() {
        // Helper process with `--foo bar`
        #[cfg(target_os = "linux")]
        let child1 = {
            let mut c = std::process::Command::new("sh");
            c.args(["-c", "sleep 5"]).arg("--foo").arg("bar").spawn()
        };
        #[cfg(target_os = "macos")]
        let child1 = {
            let mut c = std::process::Command::new("sh");
            c.args([
                "-c",
                // A cheap infinite loop so the process survives long enough for the test.
                "while true; do sleep 60; done",
                "--", // <-- end‑of‑options marker
                "--foo",
                "bar", // <-- the two tokens you want to match
            ])
            .spawn()
        };
        #[cfg(windows)]
        let child1 = std::process::Command::new("cmd")
            .args(["/C", "timeout /T 5 /NOBREAK &REM --foo bar"])
            .spawn();
        let mut child1 = child1.expect("spawn child1");
        let pid1 = child1.id();
        {
            let sys = sysinfo::System::new_with_specifics(
                RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
            );
            if let Some(p) = sys.process(sysinfo::Pid::from_u32(pid1)) {
                eprintln!("DEBUG argv = {:?}", p.cmd());
            }
        }

        thread::sleep(Duration::from_millis(200));

        let cases: &[(&[&[&str]], Option<u32>)] = &[
            (&[&["--unlikely-flag", "foo"]], None), // no match
            (&[&["--foo", "bar"]], Some(pid1)),     // correct order
            (&[&["bar", "--foo"]], None),           // wrong order
        ];
        for (patterns, expect) in cases {
            let found = super::get_server_pid_by_cmd_args(patterns);
            match expect {
                Some(p) => assert_eq!(
                    found,
                    Some(*p),
                    "Failed to find PID for patterns: {:?}",
                    patterns
                ),
                None => assert!(found.is_none()),
            }
        }
        let _ = child1.kill();

        // Helper process for "--test-flag" (equals vs space)
        #[cfg(target_os = "linux")]
        let child2 = {
            let mut c = std::process::Command::new("sh");
            c.args(["-c", "sleep 5"]).arg("--test-flag=value").spawn()
        };
        #[cfg(target_os = "macos")]
        let child2 = {
            let mut c = std::process::Command::new("sh");
            c.args([
                "-c",
                // A cheap infinite loop so the process survives long enough for the test.
                "while true; do sleep 60; done",
                "--", // <-- end‑of‑options marker
                "--test-flag",
                "value", // <-- the two tokens you want to match
            ])
            .spawn()
        };
        #[cfg(windows)]
        let child2 = {
            let mut c = std::process::Command::new("cmd");
            c.args(["/C", "timeout /T 5 /NOBREAK &REM --test-flag=value"])
                .spawn()
        };
        let mut child2 = child2.expect("spawn child2");
        let pid2 = child2.id();
        thread::sleep(Duration::from_millis(200));

        let patterns: &[&[&str]] = &[&["--test-flag", "value"], &["--test-flag=value"]];
        assert_eq!(super::get_server_pid_by_cmd_args(patterns), Some(pid2));

        let _ = child2.kill();
    }

    #[test]
    fn test_discover_pidfiles_cases() {
        let exe = "svc";
        let bogus = Path::new("/tmp/definitely-missing-dir-xyz");
        assert!(super::discover_pidfiles(bogus, exe).is_empty());

        let td = tempfile::tempdir().unwrap();

        let good = td.path().join(format!("{exe}_good.pid"));
        std::fs::write(&good, "12345").unwrap(); // valid

        let bad = td.path().join(format!("{exe}_bad.pid"));
        std::fs::write(&bad, "NA").unwrap();
        std::fs::write(td.path().join("foo-host.pid"), "3").unwrap();
        std::fs::write(td.path().join("svc-host.txt"), "4").unwrap();

        let results = super::discover_pidfiles(td.path(), exe);

        // always expect the valid “good” entry
        assert!(results.iter().any(|t| t.1 == 12345));
    }

    #[test]
    fn test_pid_alive_states() -> Result<()> {
        let mut child = short_cmd().spawn().unwrap();
        let pid = child.id();
        assert!(pid_alive(pid)?);
        let _ = child.kill();
        let _ = child.wait();
        assert!(!pid_alive(pid)?);

        #[cfg(target_os = "linux")]
        {
            let ch = std::process::Command::new("sh")
                .args(["-c", "exit 0"])
                .spawn()
                .unwrap();
            let zpid = ch.id();
            drop(ch);
            thread::sleep(Duration::from_millis(100)); // allow /proc to update
            assert!(!pid_alive(zpid).unwrap());
        }
        Ok(())
    }
}
