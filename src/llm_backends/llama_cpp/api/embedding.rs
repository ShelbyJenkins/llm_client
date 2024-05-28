use super::{
    client::LlamaClient,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateEmbeddingRequest, LlamaCreateEmbeddingResponse},
};

pub struct Embedding<'c, C: Config> {
    client: &'c LlamaClient<C>,
}

impl<'c, C: Config> Embedding<'c, C> {
    pub fn new(client: &'c LlamaClient<C>) -> Self {
        Self { client }
    }
    pub async fn create(
        &self,
        request: LlamaCreateEmbeddingRequest,
    ) -> Result<LlamaCreateEmbeddingResponse, LlamaApiError> {
        self.client.post("/embedding", request).await
    }
}
