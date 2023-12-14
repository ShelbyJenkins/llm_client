use super::{
    client::Client,
    config::Config,
    error::LlamaApiError,
    types::{LlamaCreateCompletionResponse, LlamaCreateCompletionsRequest},
};

/// Given a prompt, the model will return one or more predicted completions,
/// and can also return the probabilities of alternative tokens at each position.
/// We recommend most users use our Chat completions API.
/// [Learn more](https://platform.openai.com/docs/deprecations/2023-07-06-gpt-and-embeddings)
///
/// Related guide: [Legacy Completions](https://platform.openai.com/docs/guides/gpt/completions-api)
pub struct Completions<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Completions<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates a completion for the provided prompt and parameters
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
