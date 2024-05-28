use super::{
    client::LlamaClient,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateDetokenizeRequest, LlamaCreateDetokenizeResponse},
};

pub struct Detokenize<'c, C: Config> {
    client: &'c LlamaClient<C>,
}

impl<'c, C: Config> Detokenize<'c, C> {
    pub fn new(client: &'c LlamaClient<C>) -> Self {
        Self { client }
    }

    /// Creates a completion for the provided prompt and parameters
    pub async fn create(
        &self,
        request: LlamaCreateDetokenizeRequest,
    ) -> Result<LlamaCreateDetokenizeResponse, LlamaApiError> {
        self.client.post("/detokenize", request).await
    }
}
