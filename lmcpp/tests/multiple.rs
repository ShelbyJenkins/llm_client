use lmcpp::*;
use serial_test::serial;

mod common;
use common::*;

#[test]
#[serial]
fn multiple_servers_at_once() -> LmcppResult<()> {
    let srv1 = make_server()?;
    let srv2 = make_server()?;
    let srv3 = make_server()?;
    let pid1 = srv1.pid();
    let pid2 = srv2.pid();
    let pid3 = srv3.pid();
    assert!(server::process::pid::pid_alive(pid1).unwrap());
    assert!(server::process::pid::pid_alive(pid2).unwrap());
    assert!(server::process::pid::pid_alive(pid3).unwrap());
    let exe_name = current_server_exe_name(pid1);
    let pids = server::process::pid::get_all_server_pids(&exe_name);
    assert!(
        pids.contains(&pid1) && pids.contains(&pid2) && pids.contains(&pid3),
        "Expected pids {pid1} & {pid2} & {pid3} in list, got {pids:?}"
    );
    let host1 = srv1.client.host().to_owned();
    let host2 = srv2.client.host().to_owned();
    let host3 = srv3.client.host().to_owned();
    // All hosts must be unique
    assert_ne!(host1, host2);
    assert_ne!(host1, host3);
    assert_ne!(host2, host3);

    std::thread::scope(|s| {
        s.spawn(|| common::endpoints::exercise_all(&srv1).expect("srv1 endpoints failed"));
        s.spawn(|| common::endpoints::exercise_all(&srv2).expect("srv2 endpoints failed"));
        s.spawn(|| common::endpoints::exercise_all(&srv3).expect("srv3 endpoints failed"));
    });

    srv1.stop()?;
    srv2.stop()?;
    srv3.stop()?;
    assert!(
        !server::process::pid::pid_alive(pid1).unwrap()
            && !server::process::pid::pid_alive(pid2).unwrap()
            && !server::process::pid::pid_alive(pid3).unwrap(),
        "One or more PIDs survived stop()"
    );
    Ok(())
}

#[test]
#[serial]
fn discover_pidfiles_lists_every_host() -> lmcpp::LmcppResult<()> {
    let srv1 = make_server()?;
    let srv2 = make_server()?;
    let pid1 = srv1.pid();
    let pid2 = srv2.pid();
    let dir1 = srv1.pidfile_path.parent().unwrap();
    let dir2 = srv2.pidfile_path.parent().unwrap();
    assert!(dir1 == dir2, "Servers must use the same pidfile directory");
    let exe_name = current_server_exe_name(srv1.pid());
    let list = server::process::pid::discover_pidfiles(dir1, &exe_name);

    let mut pids: std::collections::HashSet<_> = list.into_iter().map(|t| t.1).collect();
    assert!(
        pids.remove(&pid1) && pids.remove(&pid2),
        "discover_pidfiles missing one of the PIDs; got {pids:?}"
    );
    Ok(())
}

#[test]
#[serial]
fn kill_all_servers_multiple_instances() -> lmcpp::LmcppResult<()> {
    let srv1 = make_server()?;
    let srv2 = make_server()?;
    let pid1 = srv1.pid();
    let pid2 = srv2.pid();
    let exe_name = current_server_exe_name(pid1);

    server::process::kill::kill_all_servers(&exe_name).expect("kill_all_servers returned error");

    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(
        !server::process::pid::pid_alive(pid1).unwrap()
            && !server::process::pid::pid_alive(pid2).unwrap(),
        "One or more PIDs survived stop()"
    );
    drop((srv1, srv2));
    Ok(())
}
