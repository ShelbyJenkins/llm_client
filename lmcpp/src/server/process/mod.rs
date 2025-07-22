pub mod error;
pub mod guard;
pub mod kill;
pub mod pid;

pub use error::*;
pub use kill::*;
pub use pid::*;

const POLITE_WAIT: std::time::Duration = std::time::Duration::from_secs(2);
const POLL_INTERVAL_MS: u64 = 100;
const FORCE_KILL_TIMEOUT_SECS: u64 = 1;

#[cfg(test)]
// Shared test utilities for process management
mod tests_helpers {

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub const TEST_EXE: &str = "llama_cpp_test";
    #[cfg(target_os = "windows")]
    pub const TEST_EXE: &str = "llama_cpp_test";

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub fn long_cmd() -> std::process::Command {
        let mut c = std::process::Command::new("sleep");
        c.arg("30");
        c
    }
    #[cfg(target_os = "windows")]
    pub fn long_cmd() -> std::process::Command {
        let mut c = std::process::Command::new("timeout.exe");
        c.args(["/T", "30", "/NOBREAK"]);
        c
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub fn short_cmd() -> std::process::Command {
        let mut c = std::process::Command::new("sleep");
        c.arg("1");
        c
    }

    #[cfg(target_os = "windows")]
    pub fn short_cmd() -> std::process::Command {
        let mut c = std::process::Command::new("timeout.exe");
        c.args(["/T", "5", "/NOBREAK"]);
        c
    }
}
