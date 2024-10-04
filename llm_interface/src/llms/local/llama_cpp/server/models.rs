use serde::{Deserialize, Serialize};

use crate::llms::{
    api::{client::ApiClient, error::ClientError},
    local::llama_cpp::LlamaCppConfig,
};

pub enum ModelStatus {
    LoadedModel(String),
    LoadedModels(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
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
