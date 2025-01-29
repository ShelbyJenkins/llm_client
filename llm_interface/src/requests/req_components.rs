// Internal imports
use llm_prompt::{check_and_get_max_tokens, MaxTokenState, RequestTokenLimitError};

#[derive(Clone)]
pub struct RequestConfig {
    /// Total token limit for input and output combined.
    ///
    /// This value represents the maximum number of tokens that can be used for both
    /// the input prompt and the model's output combined. It's set once when the
    /// RequestConfig is created and is used to calculate the available token budget
    /// for each request.
    ///
    /// This limit applies to all LLM types, including both local and API-based models.
    pub(crate) model_ctx_size: u64,
    /// Maximum token limit for model output.
    ///
    /// This value represents the maximum number of tokens the model can generate
    /// as output. It's set once when the RequestConfig is created and is used to
    /// ensure the model's response doesn't exceed this limit.
    ///
    /// Note: This limit is primarily used by API-based LLMs. For local LLMs,
    /// [RequestConfig::inference_ctx_size] should use the same value as '[RequestConfig::model_ctx_size].
    pub(crate) inference_ctx_size: u64,
    /// Requested maximum number of tokens for the model's output.
    ///
    /// This value specifies the upper limit of tokens the model should generate in its response.
    ///
    /// The system uses this value, along with the input prompt length, to ensure the entire
    /// request (input + output) stays within the model's token limits.
    ///
    /// - For OpenAI API-compatible LLMs, this corresponds to the 'max_tokens' parameter.
    /// - For local LLMs, this is equivalent to the 'n_predict' parameter.
    ///
    /// If `None`, the system will use a default or calculated value based on [RequestConfig::model_ctx_size] or [RequestConfig::inference_ctx_size].
    pub requested_response_tokens: Option<u64>,
    /// A small safety margin to prevent exceeding model limits.
    ///
    /// This is a count of tokens subtracted from the total available tokens to help ensure
    /// that the model doesn't unexpectedly exceed its token limit.
    /// This prevents issues that might arise from slight discrepancies in token counting or unexpected model behavior.
    ///
    /// Defaults to 10 tokens.
    pub safety_tokens: u64,
    /// Final adjusted token count for model output.
    ///
    /// This value represents the actual number of tokens requested for the model's output
    /// after all adjustments and calculations have been made. It's derived from
    /// [RequestConfig::requested_response_tokens] but may be different to ensure the request stays
    /// within the model's limits.
    pub(crate) actual_request_tokens: Option<u64>,
    /// Controls the randomness of the model's output.
    ///
    /// The temperature parameter adjusts the randomness in token selection for the model's
    /// response. It accepts values between 0.0 and 2.0:
    /// - Higher values (e.g., 0.8) increase randomness, leading to more diverse and creative outputs.
    /// - Lower values (e.g., 0.2) decrease randomness, resulting in more focused and deterministic responses.
    ///
    /// Note: It's generally recommended to adjust either this parameter or `top_p`, but not both simultaneously.
    ///
    /// Special considerations:
    /// - For Anthropic models: This value is automatically scaled to the range 0.0 to 1.0 to match
    ///   the requirements of [crate::llms::api::anthropic::completion::AnthropicCompletionRequest::temperature].
    ///
    /// Supported by all LLM backends.
    ///
    /// Defaults to `1.0`.
    pub temperature: f32,
    /// Adjusts token selection based on their frequency in the generated text.
    ///
    /// The frequency penalty influences how the model selects tokens based on their existing
    /// frequency in the output. It accepts values between -2.0 and 2.0:
    /// - Positive values decrease the likelihood of repeating tokens, reducing verbatim repetition.
    /// - Negative values increase the likelihood of repeating tokens, potentially leading to more repetitive text.
    /// - A value of 0.0 (or `None`) applies no frequency-based adjustments.
    ///
    /// This can be particularly useful for:
    /// - Encouraging more diverse vocabulary usage (with positive values)
    /// - Maintaining consistent terminology (with negative values)
    ///
    /// Supported LLMs: openai, llama_cpp
    ///
    /// Defaults to `None` (no frequency penalty applied).
    pub frequency_penalty: Option<f32>,
    /// Adjusts token selection based on their presence in the generated text.
    ///
    /// The presence penalty influences how the model selects tokens based on whether they've
    /// appeared at all in the output, regardless of frequency. It accepts values between -2.0 and 2.0:
    /// - Positive values decrease the likelihood of using tokens that have appeared at all,
    ///   encouraging the model to introduce new concepts and topics.
    /// - Negative values increase the likelihood of reusing tokens that have appeared,
    ///   potentially leading to more focused or repetitive text.
    /// - A value of 0.0 applies no presence-based adjustments.
    ///
    /// This differs from `frequency_penalty` in that it considers only whether a token has
    /// appeared, not how often.
    ///
    /// Use cases:
    /// - Encouraging the model to cover more topics (with positive values)
    /// - Maintaining focus on specific themes (with negative values)
    ///
    /// Supported LLMs: openai, llama_cpp
    ///
    /// Defaults to `0.0` (no presence penalty applied).
    pub presence_penalty: f32,
    /// Controls diversity via nucleus sampling.
    ///
    /// Top-p sampling (also called nucleus sampling) is an alternative to temperature-based sampling.
    /// It selects from the smallest possible set of tokens whose cumulative probability exceeds
    /// the probability `p`. The value should be between 0.0 and 1.0:
    /// - A value of 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// - Lower values lead to more focused and deterministic outputs.
    /// - Higher values allow for more diverse outputs.
    ///
    /// Key points:
    /// - It's generally recommended to adjust either this or `temperature`, but not both simultaneously.
    /// - This method is considered more advanced than `temperature` and is recommended for
    ///   users who need fine-grained control over output diversity.
    ///
    /// Supported LLMs: All
    ///
    /// Defaults to `None` (not used, falling back to temperature-based sampling).
    pub top_p: Option<f32>,
    /// Maximum number of retry attempts after a request failure.
    ///
    /// Specifies how many times the system should attempt to retry a failed request before giving up.
    /// This can help handle transient errors or temporary service unavailability.
    ///
    /// Supported LLMs: All
    ///
    /// Defaults to `3`.
    pub retry_after_fail_n_times: u8,
    /// Automatically increase token limit on request failure.
    ///
    /// When set to `true`, if a request fails due to token limit constraints or other errors,
    /// the system will attempt to increase the token limit using [`RequestConfig::increase_token_limit`]
    /// before retrying the request.
    ///
    /// Supported LLMs: All
    ///
    /// Defaults to `false`.
    pub increase_limit_on_fail: bool,
    /// Enable prompt caching for subsequent requests.
    ///
    /// When set to `true`, the system will cache the prompt and reuse it for the next request.
    /// This can potentially improve performance for repeated or similar queries.
    ///
    /// Supported LLMs: llama_cpp
    ///
    /// Defaults to `false`.
    pub cache_prompt: bool,
}

impl RequestConfig {
    pub fn new(model_ctx_size: u64, inference_ctx_size: u64) -> Self {
        Self {
            model_ctx_size,
            inference_ctx_size,
            requested_response_tokens: None,
            actual_request_tokens: None,
            frequency_penalty: None,
            presence_penalty: 0.0,
            temperature: 1.0,
            top_p: None,
            safety_tokens: 10,
            retry_after_fail_n_times: 3,
            increase_limit_on_fail: false,
            cache_prompt: false,
        }
    }

    pub fn set_max_tokens_for_request(
        &mut self,
        total_prompt_tokens: u64,
    ) -> crate::Result<(), RequestTokenLimitError> {
        let actual_request_tokens = check_and_get_max_tokens(
            self.model_ctx_size,
            Some(self.inference_ctx_size),
            total_prompt_tokens,
            Some(self.safety_tokens),
            self.requested_response_tokens,
        )?;
        self.actual_request_tokens = Some(actual_request_tokens);
        if self.requested_response_tokens.is_none() {
            self.requested_response_tokens = Some(actual_request_tokens);
        }
        Ok(())
    }

    pub const DEFAULT_INCREASE_FACTOR: f32 = 1.33;
    pub fn increase_token_limit(
        &mut self,
        total_prompt_tokens: u64,
        token_increase_factor: Option<f32>,
    ) -> crate::Result<(), RequestTokenLimitError> {
        let token_increase_factor = token_increase_factor.unwrap_or(Self::DEFAULT_INCREASE_FACTOR);
        crate::info!("Attempting to increase requested_response_tokens by {token_increase_factor} before retrying.");

        if self.actual_request_tokens.is_none() || self.requested_response_tokens.is_none() {
            self.set_max_tokens_for_request(total_prompt_tokens)?; // To ensure both token sets are set
        }

        let initial_state = MaxTokenState {
            actual_request: self
                .actual_request_tokens
                .expect("requested_response_tokens"),
            requested_response: self
                .requested_response_tokens
                .expect("requested_response_tokens"),
        };

        self.requested_response_tokens =
            Some((initial_state.requested_response as f32 * token_increase_factor) as u64);

        let new_state = MaxTokenState {
            actual_request: self
                .actual_request_tokens
                .expect("requested_response_tokens"),
            requested_response: self
                .requested_response_tokens
                .expect("requested_response_tokens"),
        };

        crate::info!(
            "Token counts changed: actual_request ({} -> {}), requested_response ({} -> {})",
            initial_state.actual_request,
            new_state.actual_request,
            initial_state.requested_response,
            new_state.requested_response
        );

        self.set_max_tokens_for_request(total_prompt_tokens)?;

        if new_state.actual_request <= initial_state.actual_request {
            crate::error!("Increase limit failed.");
            Err(RequestTokenLimitError::TokenLimitIncreaseError {
                initial_state,
                new_state,
            })
        } else {
            crate::info!("Increase limit succeeded. Retrying.");
            Ok(())
        }
    }
}

pub trait RequestConfigTrait {
    fn config(&mut self) -> &mut RequestConfig;

    fn reset_request(&mut self);

    /// Sets the value of [RequestConfig::requested_response_tokens].
    fn max_tokens(&mut self, max_tokens: u64) -> &mut Self {
        self.config().requested_response_tokens = Some(max_tokens);
        self
    }

    /// Sets the value of [RequestConfig::frequency_penalty].
    fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.config().frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Sets the value of [RequestConfig::presence_penalty].
    fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        match presence_penalty {
            value if (-2.0..=2.0).contains(&value) => self.config().presence_penalty = value,
            _ => self.config().presence_penalty = 0.0,
        };
        self
    }

    /// Sets the value of [RequestConfig::temperature].
    fn temperature(&mut self, temperature: f32) -> &mut Self {
        match temperature {
            value if (0.0..=2.0).contains(&value) => self.config().temperature = value,
            _ => self.config().temperature = 1.0,
        };
        self
    }

    /// Sets the value of [RequestConfig::top_p].
    fn top_p(&mut self, top_p: f32) -> &mut Self {
        self.config().top_p = Some(top_p);
        self
    }

    /// Sets the value of [RequestConfig::retry_after_fail_n_times].
    fn retry_after_fail_n_times(&mut self, retry_after_fail_n_times: u8) -> &mut Self {
        self.config().retry_after_fail_n_times = retry_after_fail_n_times;
        self
    }

    /// Sets the value of [RequestConfig::increase_limit_on_fail].
    fn increase_limit_on_fail(&mut self, increase_limit_on_fail: bool) -> &mut Self {
        self.config().increase_limit_on_fail = increase_limit_on_fail;
        self
    }

    /// Sets the value of [RequestConfig::cache_prompt].
    fn cache_prompt(&mut self, cache_prompt: bool) -> &mut Self {
        self.config().cache_prompt = cache_prompt;
        self
    }
}

impl std::fmt::Display for RequestConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    model_ctx_size: {}", self.model_ctx_size)?;
        writeln!(f, "    inference_ctx_size: {}", self.inference_ctx_size)?;
        writeln!(
            f,
            "    requested_response_tokens: {:?}",
            self.requested_response_tokens
        )?;
        writeln!(
            f,
            "    actual_request_tokens: {:?}",
            self.actual_request_tokens
        )?;
        writeln!(f, "    frequency_penalty: {:?}", self.frequency_penalty)?;
        writeln!(f, "    presence_penalty: {:?}", self.presence_penalty)?;
        writeln!(f, "    temperature: {:?}", self.temperature)?;
        writeln!(f, "    top_p: {:?}", self.top_p)?;
        writeln!(
            f,
            "    retry_after_fail_n_times: {:?}",
            self.retry_after_fail_n_times
        )?;
        writeln!(
            f,
            "    increase_limit_on_fail: {:?}",
            self.increase_limit_on_fail
        )?;
        writeln!(f, "    cache_prompt: {:?}", self.cache_prompt)
    }
}
