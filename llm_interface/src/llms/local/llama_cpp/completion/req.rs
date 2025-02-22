use crate::requests::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct LlamaCppCompletionRequest {
    pub prompt: Vec<usize>,
    #[serde(skip)]
    pub prompt_string: Option<String>,
    /// A formatted "Grammar" as a string.
    /// See: https://github.com/richardanaya/gbnf/blob/main/gbnf/src/lib.rs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// Re-use previously cached prompt from the last request if possible. This may prevent re-caching the prompt from scratch. Default: false
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_prompt: Option<bool>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    ///
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
    /// Mathematically, the bias is added to the logits generated by the model prior to sampling.
    /// The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection;
    /// values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Vec<Vec<serde_json::Value>>>, // default: null
    /// The maximum number of [tokens](https://platform.openai.com/tokenizer) to generate in the chat completion.
    ///
    /// The total length of input tokens and generated tokens is limited by the model's context length. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) for counting tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<usize>,
    /// stop: Specify a JSON array of stopping strings.
    /// These words will not be included in the completion,
    /// so make sure to add them to the prompt for the next iteration (default: []).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Not currently used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// min: 0.0, max: 2.0, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// min: -2.0, max: 2.0, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// min: -2.0, max: 2.0, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// min: 0.0, max: 1.0, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

impl LlamaCppCompletionRequest {
    pub fn new(req: &CompletionRequest) -> crate::Result<Self, CompletionError> {
        let cache_prompt = if req.config.cache_prompt {
            Some(true)
        } else {
            None
        };
        Ok(Self {
            prompt: req
                .prompt
                .local_prompt()
                .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?
                .get_built_prompt_as_tokens()
                .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?,
            prompt_string: Some(
                req.prompt
                    .local_prompt()
                    .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?
                    .get_built_prompt()
                    .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?,
            ),
            grammar: req.grammar_string.clone(),
            cache_prompt,
            logit_bias: req.logit_bias.as_ref().and_then(|lb| lb.get_llama_cpp()),
            frequency_penalty: req.config.frequency_penalty,
            stream: None,
            n_predict: req.config.actual_request_tokens,
            presence_penalty: Some(req.config.presence_penalty),
            stop: Some(req.stop_sequences.to_vec()),
            temperature: Some(req.config.temperature),
            top_p: req.config.top_p,
        })
    }
}
