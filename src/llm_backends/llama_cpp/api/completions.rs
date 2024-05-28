use super::{
    client::LlamaClient,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCompletionResponse, LlamaCompletionsRequest},
};

pub struct Completions<'c, C: Config> {
    client: &'c LlamaClient<C>,
}

impl<'c, C: Config> Completions<'c, C> {
    pub fn new(client: &'c LlamaClient<C>) -> Self {
        Self { client }
    }

    pub async fn create(
        &self,
        request: LlamaCompletionsRequest,
    ) -> Result<LlamaCompletionResponse, LlamaApiError> {
        if request.stream.is_some() && request.stream.unwrap() {
            return Err(LlamaApiError::InvalidArgument(
                "When stream is true, use Completion::create_stream".into(),
            ));
        }
        self.client.post("/completion", request).await
    }
}
