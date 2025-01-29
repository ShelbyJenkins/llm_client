// Internal imports
use super::CompletionRequest;
use crate::llms::api::{
    anthropic::completion::res::AnthropicCompletionResponse,
    openai::completion::res::OpenAiCompletionResponse,
};

// Internal feature-specific imports
#[cfg(feature = "llama_cpp_backend")]
use crate::llms::local::llama_cpp::completion::res::LlamaCppCompletionResponse;
#[cfg(feature = "mistral_rs_backend")]
use mistralrs::CompletionResponse as MistralCompletionResponse;

/// The log probability of the completion.
#[derive(Debug)]
pub struct InferenceProbabilities {
    /// The token selected by the model.
    pub content: Option<String>,
    /// An array of length n_probs.
    pub top_probs: Vec<TopProbabilities>,
}

#[derive(Debug)]
pub struct TopProbabilities {
    /// The token.
    pub token: String,
    /// The log probability of this token.
    pub prob: f32,
}

/// The settings used to generate the completion.
pub struct GenerationSettings {
    /// The model used
    pub model: String,
    // pub prompt: String, // Need to think how to handle tokens vs. text
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    /// The number of choices to generate.
    pub n_choices: u8,
    /// The number of tokens to predict same as max_tokens.
    pub n_predict: Option<i32>,
    /// The maxium context size of the model or server setting.
    pub n_ctx: u64,
    pub logit_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub grammar: Option<String>,
    pub stop_sequences: Vec<String>, // change toi vec of stop sequences
}

impl GenerationSettings {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn new_from_llama(res: &LlamaCppCompletionResponse) -> Self {
        Self {
            model: res.model.to_owned(),
            frequency_penalty: Some(res.generation_settings.frequency_penalty),
            presence_penalty: res.generation_settings.presence_penalty,
            temperature: res.generation_settings.temperature,
            top_p: Some(res.generation_settings.top_p),
            n_choices: 1,
            n_predict: Some(res.generation_settings.n_predict as i32),
            n_ctx: res.generation_settings.n_ctx as u64,
            logit_bias: res.generation_settings.logit_bias.clone(),
            grammar: Some(res.generation_settings.grammar.to_owned()),
            stop_sequences: res.generation_settings.stop.clone(),
        }
    }

    #[cfg(feature = "mistral_rs_backend")]
    pub fn new_from_mistral(req: &CompletionRequest, res: &MistralCompletionResponse) -> Self {
        Self {
            model: res.model.to_string(),
            frequency_penalty: req.config.frequency_penalty,
            presence_penalty: req.config.presence_penalty,
            temperature: req.config.temperature,
            top_p: req.config.top_p,
            n_choices: 1,
            n_predict: req.config.actual_request_tokens.map(|x| x as i32),
            n_ctx: req.config.inference_ctx_size,
            logit_bias: vec![].into(),
            grammar: None,
            stop_sequences: req
                .stop_sequences
                .sequences
                .iter()
                .map(|x| x.as_str().to_owned())
                .collect(),
        }
    }

    pub fn new_from_openai(req: &CompletionRequest, res: &OpenAiCompletionResponse) -> Self {
        Self {
            model: res.model.to_owned(),
            frequency_penalty: req.config.frequency_penalty,
            presence_penalty: req.config.presence_penalty,
            temperature: req.config.temperature,
            top_p: req.config.top_p,
            n_choices: 1,
            n_predict: req.config.actual_request_tokens.map(|x| x as i32),
            n_ctx: req.config.inference_ctx_size,
            logit_bias: None,
            grammar: None,
            stop_sequences: req
                .stop_sequences
                .sequences
                .iter()
                .map(|x| x.as_str().to_owned())
                .collect(),
        }
    }

    pub fn new_from_anthropic(req: &CompletionRequest, res: &AnthropicCompletionResponse) -> Self {
        Self {
            model: res.model.to_string(),
            frequency_penalty: req.config.frequency_penalty,
            presence_penalty: req.config.presence_penalty,
            temperature: req.config.temperature,
            top_p: req.config.top_p,
            n_choices: 1,
            n_predict: req.config.actual_request_tokens.map(|x| x as i32),
            n_ctx: req.config.inference_ctx_size,
            logit_bias: None,
            grammar: None,
            stop_sequences: req
                .stop_sequences
                .sequences
                .iter()
                .map(|x| x.as_str().to_owned())
                .collect(),
        }
    }
}

impl std::fmt::Display for GenerationSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    model: {:?}", self.model)?;
        writeln!(f, "    frequency_penalty: {:?}", self.frequency_penalty)?;
        writeln!(f, "    presence_penalty: {:?}", self.presence_penalty)?;
        writeln!(f, "    temperature: {:?}", self.temperature)?;
        writeln!(f, "    top_p: {:?}", self.top_p)?;
        writeln!(f, "    n_choices: {:?}", self.n_choices)?;
        writeln!(f, "    n_predict: {:?}", self.n_predict)?;
        writeln!(f, "    n_ctx: {:?}", self.n_ctx)?;
        writeln!(f, "    logit_bias: {:?}", self.logit_bias)?;
        writeln!(f, "    grammar: {:?}", self.grammar)?;
        writeln!(f, "    stop_sequences: {:?}", self.stop_sequences)
    }
}

/// Timing statistics for the completion request.
pub struct TimingUsage {
    /// Timestamp of when the request was created.
    pub start_time: std::time::Instant,
    /// Timestamp of when the request was completed.
    pub end_time: std::time::Instant,
    /// Total time taken to process the request.
    pub total_time: std::time::Duration,
    /// Time taken to process the prompt.
    pub prompt_processing_t: Option<std::time::Duration>,
    /// Time taken to generate the completion.
    pub generation_t: Option<std::time::Duration>,
    /// Number of prompt tokens processed per millisecond.
    pub prompt_tok_per_ms: Option<f32>,
    /// Number of prompt tokens processed per second.
    pub prompt_tok_per_sec: Option<f32>,
    /// Number of tokens generated per millisecond.
    pub generation_tok_per_ms: Option<f32>,
    /// Number of tokens generated per second.
    pub generation_tok_per_sec: Option<f32>,
}

impl TimingUsage {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn new_from_llama(
        res: &LlamaCppCompletionResponse,
        start_time: std::time::Instant,
    ) -> Self {
        Self {
            total_time: start_time.elapsed(),
            start_time,
            end_time: std::time::Instant::now(),
            prompt_processing_t: Some(std::time::Duration::from_millis(
                res.timings.prompt_ms.round() as u64,
            )),
            generation_t: Some(std::time::Duration::from_millis(
                res.timings.predicted_ms.round() as u64,
            )),
            prompt_tok_per_ms: Some(res.timings.prompt_per_token_ms),
            prompt_tok_per_sec: Some(res.timings.prompt_per_second),
            generation_tok_per_ms: Some(res.timings.predicted_per_token_ms),
            generation_tok_per_sec: Some(res.timings.predicted_per_second),
        }
    }

    #[cfg(feature = "mistral_rs_backend")]
    pub fn new_from_mistral(
        res: &MistralCompletionResponse,
        start_time: std::time::Instant,
    ) -> Self {
        Self {
            total_time: start_time.elapsed(),
            start_time,
            end_time: std::time::Instant::now(),
            prompt_processing_t: Some(std::time::Duration::from_secs_f32(
                res.usage.total_prompt_time_sec,
            )),
            generation_t: Some(std::time::Duration::from_secs_f32(
                res.usage.total_completion_time_sec,
            )),
            prompt_tok_per_ms: Some(res.usage.avg_prompt_tok_per_sec / 1000.0),
            prompt_tok_per_sec: Some(res.usage.avg_prompt_tok_per_sec),
            generation_tok_per_ms: Some(res.usage.avg_compl_tok_per_sec / 1000.0),
            generation_tok_per_sec: Some(res.usage.avg_compl_tok_per_sec),
        }
    }

    pub fn new_from_generic(start_time: std::time::Instant) -> Self {
        Self {
            total_time: start_time.elapsed(),
            start_time,
            end_time: std::time::Instant::now(),
            prompt_processing_t: None,
            generation_t: None,
            prompt_tok_per_ms: None,
            prompt_tok_per_sec: None,
            generation_tok_per_ms: None,
            generation_tok_per_sec: None,
        }
    }
}

impl std::fmt::Display for TimingUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    total_time: {:?}", self.total_time)?;
        writeln!(f, "    prompt_processing_t: {:?}", self.prompt_processing_t)?;
        writeln!(f, "    generation_t: {:?}", self.generation_t)?;
        writeln!(f, "    prompt_tok_per_ms: {:?}", self.prompt_tok_per_ms)?;
        writeln!(f, "    prompt_tok_per_sec: {:?}", self.prompt_tok_per_sec)?;
        writeln!(
            f,
            "    generation_tok_per_ms: {:?}",
            self.generation_tok_per_ms
        )?;
        writeln!(
            f,
            "    generation_tok_per_sec: {:?}",
            self.generation_tok_per_sec
        )
    }
}

/// Token statistics for the completion request.
pub struct TokenUsage {
    /// Number of tokens from the prompt which could be re-used from previous completion (n_past)
    pub tokens_cached: Option<u32>,
    /// Number of tokens evaluated in total from the prompt. Same as tokens_evaluated.
    pub prompt_tokens: u32,
    /// Number of tokens in the generated completion. Same as predicted_n.
    pub completion_tokens: u32,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u32,
    /// Dollar cost of the request.
    pub dollar_cost: Option<f32>,
    /// Cents cost of the request.
    pub cents_cost: Option<f32>,
}

impl TokenUsage {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn new_from_llama(res: &LlamaCppCompletionResponse) -> Self {
        Self {
            tokens_cached: Some(res.tokens_cached as u32),
            prompt_tokens: res.tokens_evaluated as u32,
            completion_tokens: res.timings.predicted_n as u32,
            total_tokens: res.tokens_evaluated as u32 + res.timings.predicted_n as u32,
            dollar_cost: None,
            cents_cost: None,
        }
    }
    #[cfg(feature = "mistral_rs_backend")]
    pub fn new_from_mistral(res: &MistralCompletionResponse) -> Self {
        Self {
            tokens_cached: None,
            prompt_tokens: res.usage.prompt_tokens as u32,
            completion_tokens: res.usage.completion_tokens as u32,
            total_tokens: res.usage.prompt_tokens as u32 + res.usage.completion_tokens as u32,
            dollar_cost: None,
            cents_cost: None,
        }
    }

    pub fn new_from_generic(res: &OpenAiCompletionResponse) -> Self {
        if let Some(usage) = &res.usage {
            Self {
                tokens_cached: None,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                dollar_cost: None,
                cents_cost: None,
            }
        } else {
            Self {
                tokens_cached: None,
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                dollar_cost: None,
                cents_cost: None,
            }
        }
    }

    pub fn new_from_anthropic(res: &AnthropicCompletionResponse) -> Self {
        Self {
            tokens_cached: None,
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.input_tokens + res.usage.output_tokens,
            dollar_cost: None,
            cents_cost: None,
        }
    }
}

impl std::fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    tokens_cached: {:?}", self.tokens_cached)?;
        writeln!(f, "    prompt_tokens: {:?}", self.prompt_tokens)?;
        writeln!(f, "    completion_tokens: {:?}", self.completion_tokens)?;
        writeln!(f, "    total_tokens: {:?}", self.total_tokens)?;
        writeln!(f, "    dollar_cost: {:?}", self.dollar_cost)?;
        writeln!(f, "    cents_cost: {:?}", self.cents_cost)
    }
}
