use super::{
    client::Client,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateEmbeddingRequest, LlamaCreateEmbeddingResponse},
};

pub struct Embedding<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Embedding<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }
    pub async fn create(
        &self,
        request: LlamaCreateEmbeddingRequest,
    ) -> Result<LlamaCreateEmbeddingResponse, LlamaApiError> {
        self.client.post("/embedding", request).await
    }
}
