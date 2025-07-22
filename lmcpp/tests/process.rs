//! Process-control integration tests
//! =================================
//!
//! These end-to-end checks spawn *real* OS processes and exercise the
//! public helpers in `crate::server::process` to make sure that
//! launching, crashing, and shutting down work exactly as specified.
//
//  No `use` statements — every path is fully-qualified on purpose.

mod common;
use common::*;
use lmcpp::*;
use serial_test::serial;

#[test]
#[serial]
fn server_launch_writes_pidfile() -> lmcpp::LmcppResult<()> {
    let srv = make_server()?;
    let pid = srv.pid();
    assert!(srv.pidfile_path.exists(), "pid‑file missing after launch");
    let contents = std::fs::read_to_string(&srv.pidfile_path).unwrap();
    assert_eq!(contents.trim(), pid.to_string(), "pid‑file content wrong");
    assert!(
        server::process::pid::pid_alive(pid).unwrap(),
        "pid_alive returned false for live process"
    );
    let _ = srv.stop();
    Ok(())
}

#[test]
#[serial]
fn guard_stop_graceful() -> lmcpp::LmcppResult<()> {
    let srv = make_server()?;
    let pid = srv.pid();
    srv.stop()?; // polite path
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !server::process::pid::pid_alive(pid).expect("pid_alive failed"),
        "PID {pid} still alive after stop()"
    );
    assert!(!srv.pidfile_path.exists(), "pid‑file not deleted by stop()");
    Ok(())
}

#[test]
#[serial]
fn guard_drop_cleans_up() -> LmcppResult<()> {
    let server = make_server()?;
    let pid = server.pid();
    assert!(
        server::process::pid::pid_alive(pid).unwrap(),
        "child should be alive right after launch"
    );
    let pidfile_path = server.pidfile_path.clone();

    drop(server);
    std::thread::sleep(std::time::Duration::from_millis(300));

    assert!(
        !server::process::pid::pid_alive(pid).unwrap(),
        "child PID {pid} survived guard Drop"
    );
    assert!(!pidfile_path.exists(), "pid-file should be removed on Drop");
    Ok(())
}

#[test]
#[serial]
fn guard_handles_external_crash() -> LmcppResult<()> {
    let server = make_server()?;
    let pid = server.pid();
    let _ = server::process::kill::kill_pid(pid);
    server.stop().expect("stop() after hard-kill failed");
    Ok(())
}

#[test]
#[serial]
fn kill_by_client_end_to_end() {
    let server = make_server().expect("make_server failed");
    let pid = server.pid();
    let host = server.client.host();
    let pidfile_path = server.pidfile_path.clone();

    let _ = server::process::kill::kill_by_client(&pidfile_path, &host);
    std::thread::sleep(std::time::Duration::from_secs(1));
    assert!(
        !server::process::pid::pid_alive(pid).expect("pid_alive failed"),
        "child PID {pid} should be dead after kill_by_client"
    );
    assert!(
        !pidfile_path.exists(),
        "pid-file should be removed after kill_by_client"
    );
}

#[test]
#[serial]
fn kill_by_client_fast_path() -> lmcpp::LmcppResult<()> {
    let srv = make_server()?;
    let pid = srv.pid();
    let host = srv.client.host();
    let pf = srv.pidfile_path.clone();
    server::process::kill::kill_by_client(&pf, &host).expect("kill_by_client failed");
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !server::process::pid::pid_alive(pid).unwrap(),
        "PID {pid} survived kill_by_client"
    );
    assert!(!pf.exists(), "pid‑file not removed by kill_by_client");
    Ok(())
}

#[test]
#[serial]
fn kill_by_client_argv_fallback() -> lmcpp::LmcppResult<()> {
    let srv = make_server()?;
    let pid = srv.pid();
    let host = srv.client.host();
    let pf = srv.pidfile_path.clone();
    // Emulate “missing pid‑file” so helper must fall back to argv scan
    std::fs::remove_file(&pf).unwrap();
    server::process::kill::kill_by_client(&pf, &host).expect("kill_by_client (argv path) failed");
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !server::process::pid::pid_alive(pid).unwrap(),
        "PID {pid} still alive after argv‑scan kill"
    );
    assert!(
        !pf.exists(),
        "unexpected pid‑file re‑created during argv‑scan path"
    );
    Ok(())
}

#[test]
#[serial]
fn pid_alive_and_relaunch_flow() -> lmcpp::LmcppResult<()> {
    let srv = make_server()?;
    let pid = srv.pid();
    assert!(
        server::process::pid::pid_alive(pid).unwrap(),
        "pid_alive false for fresh server"
    );
    let _ = server::process::kill::kill_pid(pid); // hard‑kill
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !server::process::pid::pid_alive(pid).unwrap(),
        "pid still alive after explicit kill"
    );
    drop(srv); // guard cleans pid‑file

    // Re‑launch should succeed because cleanup removed obstruction
    let srv2 = make_server()?; // new unique host
    let _ = srv2.stop();
    Ok(())
}
