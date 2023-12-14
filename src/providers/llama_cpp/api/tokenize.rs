use super::{
    client::Client,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateTokenizeRequest, LlamaCreateTokenizeResponse},
};

pub struct Tokenize<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Tokenize<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates a completion for the provided prompt and parameters
    pub async fn create(
        &self,
        request: LlamaCreateTokenizeRequest,
    ) -> Result<LlamaCreateTokenizeResponse, LlamaApiError> {
        self.client.post("/tokenize", request).await
    }
}
