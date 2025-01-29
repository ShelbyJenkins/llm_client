use crate::llms::{
    api::{ApiClient, ClientError},
    local::llama_cpp::LlamaCppConfig,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(super) enum HealthStatus {
    Alive,
    // Loading,
    ErrorOrOffline(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HealthErrorResponse {
    pub error: HealthErrorDetail,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HealthErrorDetail {
    pub code: u16,
    pub message: String,
    pub r#type: String,
}

#[derive(Debug, Error)]
pub(super) enum HealthError {
    // #[error("Service unavailable: {message}")]
    // ServiceUnavailable { message: String },
    #[error("Unexpected error: {0}")]
    UnexpectedError(#[from] ClientError),
}

pub(super) async fn health_request(client: &ApiClient<LlamaCppConfig>) -> HealthStatus {
    match client.get::<HealthResponse>("/health").await {
        Ok(response) => {
            crate::trace!("health_request: {:?}", response);
            HealthStatus::Alive
        }
        Err(e) => match e {
            // ClientError::ServiceUnavailable { message, .. } => {
            //     crate::trace!("health_request: {:?}", message);
            //     HealthStatus::Loading
            // }
            other => HealthStatus::ErrorOrOffline(format!("{:?}", other)),
        },
    }
}
