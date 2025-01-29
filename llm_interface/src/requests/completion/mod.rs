// Internal modules
mod error;
mod request;
mod response;

// Public exports
pub use error::CompletionError;
pub use request::CompletionRequest;
pub use response::{CompletionFinishReason, CompletionResponse};
