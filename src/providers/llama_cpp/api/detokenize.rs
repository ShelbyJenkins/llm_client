use super::{
    client::Client,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateDetokenizeRequest, LlamaCreateDetokenizeResponse},
};

pub struct Detokenize<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Detokenize<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
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
