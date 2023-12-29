use super::{
    client::Client,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateCompletionResponse, LlamaCreateCompletionsRequest},
};

pub struct Completions<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Completions<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    pub async fn create(
        &self,
        request: LlamaCreateCompletionsRequest,
    ) -> Result<LlamaCreateCompletionResponse, LlamaApiError> {
        if request.stream.is_some() && request.stream.unwrap() {
            return Err(LlamaApiError::InvalidArgument(
                "When stream is true, use Completion::create_stream".into(),
            ));
        }
        self.client.post("/completion", request).await
    }
}
