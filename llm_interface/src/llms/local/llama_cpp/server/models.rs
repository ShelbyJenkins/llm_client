use crate::llms::{
    api::{ApiClient, ClientError},
    local::llama_cpp::LlamaCppConfig,
};
use serde::{Deserialize, Serialize};

pub(super) enum ModelStatus {
    LoadedModel(String),
    LoadedModels(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

pub(crate) async fn model_request(
    client: &ApiClient<LlamaCppConfig>,
) -> crate::Result<ModelStatus, ClientError> {
    match client.get::<ModelsResponse>("/v1/models").await {
        Err(e) => Err(e),
        Ok(model_response) => match model_response.data.as_slice() {
            [] => Err(ClientError::GenericError {
                message: "No models found".to_string(),
            }),
            [model] => Ok(ModelStatus::LoadedModel(model.id.clone())),
            models => Ok(ModelStatus::LoadedModels(
                models.iter().map(|m| m.id.clone()).collect(),
            )),
        },
    }
}
