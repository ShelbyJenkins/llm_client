//! Types used in OpenAI API requests and responses.
//! These types are created from component schemas in the [OpenAPI spec](https://github.com/openai/openai-openapi)
mod impls;
mod types;
use derive_builder::UninitializedFieldError;
pub use types::*;

use super::error::LlamaApiError;

impl From<UninitializedFieldError> for LlamaApiError {
    fn from(value: UninitializedFieldError) -> Self {
        LlamaApiError::InvalidArgument(value.to_string())
    }
}
