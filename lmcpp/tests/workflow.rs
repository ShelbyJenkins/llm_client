//! Full‑stack workflow matrix – Windows / Linux / macOS
//! ====================================================
//!
//! One **integration‑test** binary, one **global lock**, and an **N × M**
//! cartesian product of runtime variants:
//!
//! * **Compute back‑end**  
//!   * Windows → CPU / CUDA (when available)  
//!   * Linux   → CPU / CUDA (when present)  
//!   * macOS   → CPU / Metal  
//! * **Build‑install mode** – `InstallOnly` or `BuildOnly`
//!
//! For every `(backend, mode)` pair we launch the server under **two** transport
//! permutations: default UDS, and `--http`.
//! Each instance is then validated by invoking **every public
//! inference endpoint** through `common::endpoints`.

use lmcpp::*;
use serial_test::serial;

mod common;

#[test]
#[serial]
fn workflow_variants_windows() -> LmcppResult<()> {
    let variants = common::runtime_variants();
    for (case_idx, (backend, mode)) in variants.into_iter().enumerate() {
        println!("🛠️  case {case_idx}  -  {backend:?} / {mode:?}");

        /* ── Build or install tool-chain ─────────────────────────── */
        let tmp_root = tempfile::TempDir::new().expect("cannot create tmp dir");

        let toolchain = LmcppToolChain::builder()
            .compute_backend(backend)
            .build_install_mode(mode)
            .override_root(tmp_root.path())
            .unwrap()
            .build()
            .unwrap();

        let outcome = toolchain.run().expect("tool-chain workflow failed");
        let expected_status = match mode {
            LmcppBuildInstallMode::InstallOnly => LmcppBuildInstallStatus::Installed,
            LmcppBuildInstallMode::BuildOnly => LmcppBuildInstallStatus::Built,
            LmcppBuildInstallMode::BuildOrInstall => {
                unimplemented!("BuildOrInstall not covered by integration tests")
            }
        };
        assert_eq!(
            outcome.status, expected_status,
            "unexpected tool-chain status"
        );

        /* ── Prepare launcher permutations ───────────────────────── */

        let launchers = vec![
            // Default UDS
            LmcppServerLauncher::builder()
                .toolchain(toolchain.clone())
                .build(),
            // HTTP via explicit flag
            LmcppServerLauncher::builder()
                .toolchain(toolchain.clone())
                .http(true)
                .build(),
        ];

        /* ── Launch & exercise endpoints ─────────────────────────── */
        for (launch_idx, launcher) in launchers.into_iter().enumerate() {
            let server = launcher
                .load()
                .expect(&format!("launcher {launch_idx} failed in case {case_idx}"));

            common::endpoints::exercise_all(&server).expect(&format!(
                "endpoints failed (launcher {launch_idx}, case {case_idx})"
            ));
            let pid = pid_from_pidfile(&server.pidfile_path).expect("failed to read pid");
            server.stop()?;
            assert!(!pid_alive(pid).expect("failed to check pid liveness"));
        }
    }
    Ok(())
}
