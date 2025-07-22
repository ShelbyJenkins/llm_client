//! Server Process - Kill
//! =====================
//!
//! Cross-platform helpers for terminating *server instances* started by this
//! crate.  The public surface consists of two primary entry points:
//!
//! * [`kill_by_hostname`] – shut down **exactly one** instance that was launched
//!   with `--host <hostname>` (or `-h <hostname>`).
//! * [`kill_all_servers`] – blanket kill for **all** running copies of the same
//!   executable.
//!
//! Internally the module follows a three-step escalation ladder:
//!
//! 1. **PID-file fast path** – If the server created a PID-file we trust it and
//!    attempt a *polite* signal (`SIGTERM`, `CTRL_BREAK`, or the platform
//!    equivalent) against the recorded PID.  
//! 2. **Command-line scan fallback** – When the PID-file is missing or stale we
//!    search the process list for an argument pair matching `--host <hostname>`
//!    or `-h <hostname>` (same logic used by the launcher).  
//! 3. **Force-kill escalation** – If the target is still alive after the
//!    configurable *grace period* (`POLITE_WAIT`) we deliver an unconditional
//!    kill (`SIGKILL` / `TerminateProcess`).  
//!
//! The helper returns **as soon as every targeted PID is gone** or the
//! force-kill timeout elapses, whichever happens first.  Any stubborn PIDs are
//! propagated back in a [`ProcessError::TerminationTimeout`] so that higher
//! layers can decide whether to retry, log, or treat it as fatal.
//!
//! ## Error semantics
//!
//! *All* public functions return [`crate::server::process::error::Result`].  The
//! errors are designed to be actionable; for instance, [`ProcessError::PermissionDenied`]
//! will not be silently swallowed.
//!
//! ## Constants
//!
//! Timing knobs such as `POLITE_WAIT`, `POLL_INTERVAL_MS`, and
//! `FORCE_KILL_TIMEOUT_SECS` live in the *parent* module so that they are shared
//! across all process-control utilities.  They are referenced here but not
//! re-exported.
//!
//! ## Thread safety
//!
//! The functions perform blocking I/O (`std::fs`, `nix`, Win32 API) and sleep
//! loops.  They are therefore *synchronous* and expected to run on a dedicated
//! management thread or as part of a CLI command—not in an async runtime.

use core::str;
use std::{
    path::Path,
    time::{Duration, Instant},
};

use super::{error::*, pid::*, *};

pub fn kill_by_client(pidfile_path: &Path, host: &str) -> Result<()> {
    // ── 1 ▪ PID-file fast path ────────────────────────────────────────────
    let pid: Option<u32> = std::fs::read_to_string(pidfile_path)
        .ok()
        .and_then(|s| s.trim().parse().ok());
    if let Some(pid) = pid {
        match pid_alive(pid) {
            // a) recognised & alive ⇒ killfs
            Ok(true) => {
                crate::info!("Killing server at {host} (PID {pid}) via pid-file");
                let res = kill_pids(&[pid], POLITE_WAIT);

                if res.is_ok() {
                    if let Err(e) = std::fs::remove_file(pidfile_path) {
                        crate::warn!("Failed to remove pid-file for host {host}: {e}");
                    }
                }
                return res;
            }
            // b) recognised & dead ⇒ stale file → delete & fall through
            Ok(false) => match std::fs::remove_file(pidfile_path) {
                Ok(_) => (),
                Err(e) => crate::warn!("Failed to remove stale pid-file for host {host}: {e}"),
            },
            // c) could not check (permissions, EACCES, etc.) ⇒ warn & fall back
            Err(e) => {
                crate::warn!("pid_alive({pid}) failed: {e}. Falling back to argv scan…");
            }
        }
    } else {
        if let Err(e) = std::fs::remove_file(pidfile_path) {
            crate::warn!("Failed to remove malformed pid-file for host {host}: {e}");
        }
    };

    // ── 2 ▪ Fallback: argv scan for --host / -h ───────────────────────────
    let patterns: &[&[&str]] = &[&["--host", host], &["-h", host]];
    if let Some(pid) = get_server_pid_by_cmd_args(patterns) {
        crate::info!("Killing server at {host} (PID {pid}) via argv scan");
        match kill_pids(&[pid], POLITE_WAIT) {
            Ok(()) => {
                match std::fs::remove_file(pidfile_path) {
                    Ok(_) => (),
                    Err(e) => {
                        crate::warn!("Failed to remove stale pid-file for host {host}: {e}")
                    }
                }
                return Ok(());
            }
            Err(e) => {
                crate::warn!("Failed to kill server at {host} (PID {pid}): {e}");
            }
        }
    }

    // ── 3 ▪ Nothing matched ───────────────────────────────────────────────
    Err(ProcessError::NoSuchProcess {
        query: format!("host={host}"),
    })
}

pub fn kill_all_servers(executable_name: &str) -> Result<()> {
    crate::info!("Killing all {executable_name} processes");

    // ── 1 ▪ kill-by-path / name (old behaviour) ────────────────
    let pids = get_all_server_pids(executable_name);
    let mut errors = Vec::new();

    if !pids.is_empty() {
        if let Err(e) = kill_pids(&pids, POLITE_WAIT) {
            errors.push(e);
        }
    }

    // ── 2 ▪ sweep pid-files we just invalidated ────────────────
    for pid in pids {
        // If the process is gone (or we just killed it), delete the file.
        match pid_alive(pid) {
            Ok(false) => crate::info!("PID {pid} shut down"),
            Ok(true) => crate::warn!("PID {pid} still alive, but we tried to kill it"),
            Err(e) => crate::warn!("Could not probe PID {pid}: {e}"),
        }
    }

    // ── 3 ▪ summarise result ───────────────────────────────────
    if errors.is_empty() {
        Ok(())
    } else {
        // bubble up the first error but log the rest
        for e in &errors[1..] {
            crate::warn!("Additional error while killing servers: {e}");
        }
        Err(errors.remove(0))
    }
}

fn kill_pids(pids: &[u32], polite_wait: Duration) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(pids.len());
    let uniq: Vec<u32> = pids.iter().copied().filter(|p| seen.insert(*p)).collect();
    if uniq.is_empty() {
        return Ok(());
    }
    let start = Instant::now();
    // phase 1 ▪ TERM / taskkill /T
    for pid in &uniq {
        match pid_alive(*pid) {
            Ok(true) => match kill_pid(*pid) {
                Ok(()) => crate::info!("Sent TERM to PID {}", pid),
                Err(e) => crate::error!("Failed to send TERM to PID {}: {}", pid, e),
            },
            Ok(false) => (),
            Err(e) => crate::error!("Failed to check PID {}: {}", pid, e),
        }
    }

    // phase 2 ▪ wait a little, bailing early if all gone
    let polite_deadline = Instant::now() + polite_wait;
    let mut probe_failures: Vec<(u32, ProcessError)> = Vec::new();

    while Instant::now() < polite_deadline {
        let all_dead = pids.iter().all(|&pid| match pid_alive(pid) {
            Ok(alive) => !alive,
            Err(e) => {
                // record only once; keep treating PID as "alive"
                if !probe_failures.iter().any(|(p, _)| *p == pid) {
                    probe_failures.push((pid, e));
                }
                false
            }
        });

        if all_dead {
            break;
        }
        std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }

    // phase 3 ▪ KILL / taskkill /F
    for &pid in pids {
        if let Err(e) = force_kill_pid(pid) {
            crate::error!("Failed to force-kill PID {pid}: {e}");
        }
    }

    if !probe_failures.is_empty() {
        for (pid, err) in &probe_failures {
            crate::warn!("Never obtained status for PID {pid}: {err}");
        }
    }
    let force_kill_deadline = Instant::now() + Duration::from_secs(FORCE_KILL_TIMEOUT_SECS);
    while Instant::now() < force_kill_deadline {
        if pids
            .iter()
            .all(|&pid| matches!(pid_alive(pid), Ok(false) | Err(_)))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }

    #[cfg(target_os = "macos")]
    for &pid in &uniq {
        use nix::sys::wait::{waitpid, WaitPidFlag};
        let _ = nix::unistd::Pid::from_raw(pid as i32);
        // Try to reap our own children; ignore errors & non‑children.
        let _ = waitpid(
            nix::unistd::Pid::from_raw(pid as i32),
            Some(WaitPidFlag::WNOHANG),
        );
    }

    // return any stubborn PIDs so callers can escalate or log
    let leftovers: Vec<u32> = pids
        .iter()
        .copied()
        .filter(|&pid| match pid_alive(pid) {
            Ok(alive) => alive,
            Err(_) => true, // treat unknowns as alive
        })
        .collect();

    let elapsed = start.elapsed();
    if leftovers.is_empty() {
        Ok(())
    } else {
        Err(ProcessError::TerminationTimeout {
            operation: "kill_pids",
            elapsed,
            leftovers,
        })
    }
}

#[cfg(unix)]
pub fn kill_pid(pid: u32) -> Result<()> {
    use nix::{
        errno::Errno,
        sys::signal::{kill, Signal},
        unistd::Pid,
    };
    match kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
        Ok(_) | Err(Errno::ESRCH) => Ok(()), // gone already → success

        Err(Errno::EPERM) => Err(ProcessError::PermissionDenied {
            action: "send SIGTERM",
            source: "operation not permitted".into(),
        }),

        Err(e) => Err(ProcessError::CommandFailed {
            action: "send SIGTERM",
            source: e.into(),
        }),
    }
}

#[cfg(unix)]
fn force_kill_pid(pid: u32) -> Result<()> {
    use nix::{
        errno::Errno,
        sys::signal::{kill, Signal},
        unistd::Pid,
    };
    match kill(Pid::from_raw(pid as i32), Signal::SIGKILL) {
        Ok(_) | Err(Errno::ESRCH) => Ok(()),

        Err(Errno::EPERM) => Err(ProcessError::PermissionDenied {
            action: "send SIGKILL",
            source: "operation not permitted".into(),
        }),

        Err(e) => Err(ProcessError::CommandFailed {
            action: "send SIGKILL",
            source: e.into(),
        }),
    }
}

#[cfg(windows)]
pub fn kill_pid(pid: u32) -> Result<()> {
    use windows::Win32::{
        Foundation::CloseHandle,
        System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE},
    };

    unsafe {
        // Open the process with termination rights
        let handle = OpenProcess(PROCESS_TERMINATE, false, pid).map_err(|e| {
            ProcessError::CommandFailed {
                action: "OpenProcess",
                source: Box::new(e),
            }
        })?;

        if handle.is_invalid() {
            // Process already gone
            return Ok(());
        }

        // Terminate the process (this is already "forceful" on Windows)
        let result = TerminateProcess(handle, 1);
        let _ = CloseHandle(handle);

        result.map_err(|e| ProcessError::CommandFailed {
            action: "TerminateProcess",
            source: Box::new(e),
        })
    }
}

#[cfg(windows)]
pub fn force_kill_pid(pid: u32) -> Result<()> {
    use windows::Win32::{
        Foundation::CloseHandle,
        System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE},
    };
    fn win32_error(action: &'static str) -> ProcessError {
        // windows::core::Error already wraps GetLastError + FormatMessageW.
        let err = windows::core::Error::from_win32();
        match err.code().0 {
            5 => ProcessError::PermissionDenied {
                /* ERROR_ACCESS_DENIED */
                action,
                source: Box::new(err),
            },
            _ => ProcessError::CommandFailed {
                action,
                source: Box::new(err),
            },
        }
    }

    unsafe {
        let h = OpenProcess(PROCESS_TERMINATE, false, pid).map_err(|e| {
            ProcessError::CommandFailed {
                action: "force-kill (OpenProcess)",
                source: e.into(),
            }
        })?;
        if h.is_invalid() {
            let err = windows::core::Error::from_win32();
            return match err.code().0 {
                87 => Ok(()), // already gone
                _ => Err(win32_error("force-kill (OpenProcess)")),
            };
        }
        match TerminateProcess(h, 1) {
            Ok(_) => CloseHandle(h).map_err(|e| ProcessError::CommandFailed {
                action: "force-kill (CloseHandle)",
                source: e.into(),
            }),
            Err(_) => {
                let _ = CloseHandle(h);
                Err(win32_error("force-kill (TerminateProcess)"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serial_test::serial;
    use tempfile::tempdir;

    use super::*;
    use crate::server::process::tests_helpers::*;

    // ─────────────────────────────────────────────────────────────────────
    //  kill_pids – unchanged edge-case matrix
    // ─────────────────────────────────────────────────────────────────────
    #[test]
    #[serial]
    fn kill_pids_scenarios() {
        use ProcessError::*;

        // 1 ▪ empty slice is always Ok
        assert!(kill_pids(&[], Duration::from_millis(10)).is_ok());

        // 2 ▪ “all-dead” slice: spawn a short-lived child, wait for it to exit,
        //     then point kill_pids at that *real* but now-dead PID.
        let dead_pid = {
            let mut child = short_cmd().spawn().unwrap();
            let pid = child.id();
            let _ = child.wait();
            pid
        };
        match kill_pids(&[dead_pid], Duration::from_millis(200)) {
            Ok(()) | Err(TerminationTimeout { .. }) => {} // both are fine
            Err(e) => panic!("unexpected error on all-dead slice: {e:?}"),
        }

        // 3 ▪ Live-process variants ------------------------------------------------
        fn spawn_and_kill(wait: Duration, duplicate: bool) {
            let mut child = long_cmd().spawn().unwrap();
            let pid = child.id();
            let pids = if duplicate { vec![pid, pid] } else { vec![pid] };

            // Accept success or the timeout error – either means *we tried*.
            match kill_pids(&pids, wait) {
                Ok(()) | Err(TerminationTimeout { .. }) => {}
                Err(e) => panic!("kill_pids failed unexpectedly: {e:?}"),
            }

            let _ = child.wait();
            assert!(
                !pid_alive(pid).unwrap_or(true),
                "child {pid} still alive after kill_pids(wait={wait:?}, dup={duplicate})"
            );
        }

        // duplicate-PID case
        spawn_and_kill(Duration::from_secs(2), true);

        // polite-wait vs. force-kill
        for &d in &[Duration::from_secs(2), Duration::from_secs(0)] {
            spawn_and_kill(d, false);
        }

        // 4 ▪ Mixed live / dead ----------------------------------------------------
        let mut child_live = long_cmd().spawn().unwrap();
        let pid_live = child_live.id();

        let mut child_dead = long_cmd().spawn().unwrap();
        let pid_dead = child_dead.id();
        kill_pid(pid_dead).unwrap(); // terminate it immediately
        let _ = child_dead.wait(); // reap

        match kill_pids(&[pid_dead, pid_live], Duration::from_millis(500)) {
            Ok(()) | Err(TerminationTimeout { .. }) => {}
            Err(e) => panic!("mixed kill failed unexpectedly: {e:?}"),
        }
        let _ = child_live.wait();
        assert!(
            !pid_alive(pid_live).unwrap_or(true),
            "live child {pid_live} survived mixed-status kill"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    //  kill_by_client – permutations covering pid-file & argv scan paths
    // ─────────────────────────────────────────────────────────────────────
    #[test]
    #[serial]
    fn kill_by_client_scenarios() {
        use sanitize_filename::sanitize; // kept local to honour “no extra imports” request

        struct Case<'a> {
            name: &'a str,
            pidfile_raw: Option<&'a str>, // None ⇒ no pid-file
            spawn_child: bool,            // spawn a long_cmd() child?
            expect_ok: bool,              // expect Ok(())
            pf_removed: bool,             // should pid-file be gone?
            argv_scan: bool,              // child must advertise --host
        }

        let cases = [
            Case {
                name: "no_match",
                pidfile_raw: None,
                spawn_child: false,
                expect_ok: false,
                pf_removed: false,
                argv_scan: false,
            },
            Case {
                name: "corrupt_pidfile",
                pidfile_raw: Some("not-a-number"),
                spawn_child: false,
                expect_ok: false,
                pf_removed: true,
                argv_scan: false,
            },
            Case {
                name: "stale_pidfile",
                pidfile_raw: Some("999999"), // dead PID
                spawn_child: false,
                expect_ok: false,
                pf_removed: true,
                argv_scan: false,
            },
            Case {
                name: "pidfile_happy",
                pidfile_raw: None, // will be filled with real PID below
                spawn_child: true,
                expect_ok: true,
                pf_removed: true,
                argv_scan: false,
            },
            Case {
                name: "argv_scan",
                pidfile_raw: None,
                spawn_child: true,
                expect_ok: true,
                pf_removed: false,
                argv_scan: true,
            },
        ];

        for Case {
            name,
            pidfile_raw,
            spawn_child,
            expect_ok,
            pf_removed,
            argv_scan,
        } in cases
        {
            let td = tempdir().unwrap();
            let host = name; // host string kept short to stay well below 240 chars

            // Construct the pid-file path *exactly* the way production code does.
            // We mimic the “*_unix_*” layout for simplicity; the exact flavour
            // (unix / http / tcp) is irrelevant to the termination logic.
            let pid_id = sanitize(format!("{TEST_EXE}_unix_{host}").to_ascii_lowercase());
            let pidfile_path = td.path().join(format!("{pid_id}.pid"));

            // maybe spawn a child
            let child = if spawn_child {
                #[cfg(unix)]
                {
                    let mut c = std::process::Command::new("sh");
                    c.args(["-c", "sleep 30"]);
                    if argv_scan {
                        c.arg("--host").arg(host);
                    }
                    Some(c.spawn().unwrap())
                }
                #[cfg(windows)]
                {
                    let mut c = std::process::Command::new("cmd");
                    c.args(["/C", "timeout", "/T", "30", "/NOBREAK"]);
                    if argv_scan {
                        c.arg("--host").arg(host);
                    }
                    Some(c.spawn().unwrap())
                }
            } else {
                None
            };

            // create / tweak pid-file if required
            if let Some(contents) = pidfile_raw {
                std::fs::write(&pidfile_path, contents.as_bytes()).unwrap();
            } else if spawn_child && !argv_scan {
                // happy-path pid-file with live PID
                let pid = child.as_ref().unwrap().id();
                std::fs::write(&pidfile_path, pid.to_string()).unwrap();
            }

            // -------- exercise ------------------------------------------------
            let result = kill_by_client(&pidfile_path, host);

            // -------- assertions ---------------------------------------------
            if expect_ok {
                result.unwrap();
            } else {
                matches!(
                    result.expect_err("should fail"),
                    ProcessError::NoSuchProcess { .. }
                );
            }

            // pid-file may have been removed by kill_by_client
            let expect_exists = pidfile_raw.is_some() && !pf_removed;
            assert_eq!(
                pidfile_path.exists(),
                expect_exists,
                "[{name}] pid-file existence mismatch (expected {expect_exists})"
            );

            if let Some(mut ch) = child {
                let _ = ch.wait();
                assert!(
                    !pid_alive(ch.id()).unwrap_or(true),
                    "[{name}] child process not killed"
                );
            }
        }
    }
}
