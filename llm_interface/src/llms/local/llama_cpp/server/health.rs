use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::llms::{
    api::{client::ApiClient, error::ClientError},
    local::llama_cpp::LlamaCppConfig,
};

pub enum HealthStatus {
    Alive,
    Loading,
    ErrorOrOffline(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthErrorResponse {
    pub error: HealthErrorDetail,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthErrorDetail {
    pub code: u16,
    pub message: String,
    pub r#type: String,
}

#[derive(Debug, Error)]
pub enum HealthError {
    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },
    #[error("Unexpected error: {0}")]
    UnexpectedError(#[from] ClientError),
}

pub(crate) async fn health_request(client: &ApiClient<LlamaCppConfig>) -> HealthStatus {
    match client.get::<HealthResponse>("/health").await {
        Ok(response) => {
            crate::trace!("health_request: {:?}", response);
            HealthStatus::Alive
        }
        Err(e) => match e {
            ClientError::ServiceUnavailable { message, .. } => {
                crate::trace!("health_request: {:?}", message);
                HealthStatus::Loading
            }
            other => HealthStatus::ErrorOrOffline(format!("{:?}", other)),
        },
    }
}
