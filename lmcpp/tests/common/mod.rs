//! Shared helpers for the full‑stack integration‑test suite.
//
//  The helpers below avoid any `use` statements so they honour the project
//  guideline “do not show imports unless requested”.
#![allow(dead_code)]

pub mod endpoints;

use lmcpp::*;

/// Detect at run‑time whether a CUDA‑capable NVIDIA GPU is available.
/// Returns `true` iff NVML initialises and reports ≥ 1 device.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn cuda_available() -> bool {
    nvml_wrapper::Nvml::init()
        .ok()
        .and_then(|nvml| nvml.device_count().ok())
        .map_or(false, |n| n > 0)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
#[allow(dead_code)]
pub fn cuda_available() -> bool {
    false
}

/// Cartesian product of every legal `(backend, mode)` pair for the current host.
///
/// *   CPU is always available.<br>
/// *   CUDA only when `cuda_available` is `true` on Linux/Windows.<br>
/// *   Metal only on macOS.
pub fn runtime_variants() -> Vec<(ComputeBackendConfig, LmcppBuildInstallMode)> {
    let mut cases = Vec::new();
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        if cuda_available() {
            cases.push((ComputeBackendConfig::Cuda, LmcppBuildInstallMode::BuildOnly));
            cases.push((
                ComputeBackendConfig::Cuda,
                LmcppBuildInstallMode::InstallOnly,
            ));
        }
    }

    #[cfg(target_os = "macos")]
    {
        cases.push((
            ComputeBackendConfig::Metal,
            LmcppBuildInstallMode::BuildOnly,
        ));
        cases.push((
            ComputeBackendConfig::Metal,
            LmcppBuildInstallMode::InstallOnly,
        ));
    }

    cases.push((ComputeBackendConfig::Cpu, LmcppBuildInstallMode::BuildOnly));
    cases.push((
        ComputeBackendConfig::Cpu,
        LmcppBuildInstallMode::InstallOnly,
    ));
    cases
}

pub fn current_server_exe_name(pid: u32) -> String {
    // Query the process list once – cheap enough for test code.
    let sys = sysinfo::System::new_with_specifics(sysinfo::RefreshKind::nothing().with_processes(
        sysinfo::ProcessRefreshKind::nothing().with_exe(sysinfo::UpdateKind::Always),
    ));

    let proc = sys
        .process(sysinfo::Pid::from_u32(pid))
        .expect("launched server PID not present in /proc list");

    // `Process::name()` is already truncated to the kernel’s 15‑char limit
    // on Linux and never exceeds that length on macOS / Windows either.
    proc.name().to_string_lossy().into_owned()
}

pub fn make_server() -> LmcppResult<LmcppServer> {
    let server = LmcppServerLauncher::builder()
        .toolchain(
            LmcppToolChain::builder()
                .compute_backend(ComputeBackendConfig::Cpu)
                .build_install_mode(LmcppBuildInstallMode::BuildOnly)
                .build()?,
        )
        .build()
        .load()?;
    Ok(server)
}
