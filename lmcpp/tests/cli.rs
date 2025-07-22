use serial_test::serial;

/// Smoke‑test that `--help` prints and exits 0.
///
#[test]
fn help_smoke() -> anyhow::Result<()> {
    assert_cmd::Command::cargo_bin("lmcpp-server-cli")?
        .arg("--help")
        .assert()
        .success() // exit status == 0
        .stdout(predicates::str::contains("Usage")); // fragment of clap help
    Ok(())
}

/// `pids` sub‑command when no servers are running.
/// You don’t need a real llama.cpp here; just assert the UX.
#[test]
#[serial]
fn pids_when_none() -> anyhow::Result<()> {
    assert_cmd::Command::cargo_bin("lmcpp-server-cli")?
        .arg("pids")
        .assert()
        .success()
        .stdout(predicates::str::contains("No running"));
    Ok(())
}

/// `kill-all` must succeed (exit‑0) even when there is nothing to kill.
#[test]
#[serial]
fn killall_is_idempotent() -> anyhow::Result<()> {
    assert_cmd::Command::cargo_bin("lmcpp-server-cli")?
        .arg("kill-all")
        .assert()
        .success()
        .stdout(predicates::str::contains("Sent termination"));
    Ok(())
}



/// Smoke‑test that `--help` prints and exits 0.
/// (No shared state ⟶ runs safely in parallel.)
#[test]
fn help_shows_usage() -> anyhow::Result<()> {
    assert_cmd::Command::cargo_bin("lmcpp-toolchain-cli")?
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("Usage: lmcpp-toolchain-cli"));
    Ok(())
}

/// Conflicting flags (e.g. duplicate backend) trip clap before main().
#[test]
fn clap_argument_errors_reported() -> anyhow::Result<()> {
    assert_cmd::Command::cargo_bin("lmcpp-toolchain-cli")?
        .args([
            "install",
            "--backend",
            "cpu",
            "--backend",
            "cuda", // duplicates on purpose
        ])
        .assert()
        .failure() // clap returns code 2
        .code(2)
        .stderr(predicates::str::contains("cannot be used multiple times"));
    Ok(())
}
