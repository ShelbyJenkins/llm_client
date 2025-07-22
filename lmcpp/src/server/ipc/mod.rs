pub mod error;
pub mod http;
pub mod uds;

pub use error::*;

pub trait ServerClient: std::fmt::Display + std::fmt::Debug + Send + Sync {
    fn get_raw(&self, path: &str) -> Result<Vec<u8>>;
    fn post_raw(&self, path: &str, body: &[u8]) -> Result<Vec<u8>>;
    fn stop(&self) -> Result<()>;
    /// This represents what the host server arg is set to.
    /// It is not necessarily the same as the host used in the base URL.
    fn host(&self) -> String;
    /// A unique identifier for the server client. Used to differentiate
    /// and create the file that stores the PID of the server process.
    fn pid_id(&self) -> String;
}

pub trait ServerClientExt: ServerClient {
    fn get<R: serde::de::DeserializeOwned>(&self, path: &str) -> Result<R> {
        let bytes = self.get_raw(path)?;
        serde_json::from_slice(&bytes).map_err(|e| e.into())
    }

    fn post<B: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let body_bytes = serde_json::to_vec(body)?;
        let response_bytes = self.post_raw(path, &body_bytes)?;
        serde_json::from_slice(&response_bytes).map_err(|e| e.into())
    }
}

impl<T: ServerClient + ?Sized> ServerClientExt for T {}
