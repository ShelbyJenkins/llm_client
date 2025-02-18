use crate::llms::{
    api::{ApiClient, ClientError},
    local::llama_cpp::LlamaCppConfig,
};
use serde::{Deserialize, Serialize};

pub(super) enum ModelStatus {
    LoadedModel(String),
    _LoadedModels(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PropsResponse {
    pub model_path: String,
}

pub(crate) async fn model_request(
    client: &ApiClient<LlamaCppConfig>,
) -> crate::Result<ModelStatus, ClientError> {
    match client.get::<PropsResponse>("/props").await {
        Err(e) => Err(e),
        Ok(model_response) => Ok(ModelStatus::LoadedModel(model_response.model_path)),
    }
}
