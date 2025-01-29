use crate::requests::*;
use serde::{Deserialize, Serialize};

impl CompletionResponse {
    #[cfg(feature = "llama_cpp_backend")]
    pub fn new_from_llama(
        req: &CompletionRequest,
        res: LlamaCppCompletionResponse,
    ) -> Result<Self, CompletionError> {
        let finish_reason = if res.stopped_eos {
            CompletionFinishReason::Eos
        } else if res.stopped_limit {
            CompletionFinishReason::StopLimit
        } else if res.stopped_word {
            match req.stop_sequences.parse_string_response(&res.stopping_word) {
                Some(stop_sequence) => {
                    CompletionFinishReason::MatchingStoppingSequence(stop_sequence)
                }
                None => CompletionFinishReason::NonMatchingStoppingSequence(Some(
                    res.stopping_word.clone(),
                )),
            }
        } else {
            return Err(CompletionError::StopReasonUnsupported(
                "No stop reason provided".to_owned(),
            ));
        };

        Ok(Self {
            id: "llama_cpp".to_owned(),
            index: None,
            content: res.content.to_owned(),
            finish_reason,
            completion_probabilities: None,
            truncated: res.truncated,
            generation_settings: GenerationSettings::new_from_llama(&res),
            timing_usage: TimingUsage::new_from_llama(&res, req.start_time),
            token_usage: TokenUsage::new_from_llama(&res),
        })
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct LlamaCppCompletionResponse {
    pub content: String,
    pub model: String,
    pub prompt: Vec<u32>,
    pub generation_settings: LlamaGenerationSettings,
    pub timings: LlamaTimings,
    pub stop: bool,
    pub stopped_eos: bool,
    pub stopped_limit: bool,
    pub stopped_word: bool,
    pub stopping_word: String,
    pub tokens_cached: u16,
    pub tokens_evaluated: u16,
    pub truncated: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct LlamaGenerationSettings {
    pub n_ctx: u16,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub n_predict: i16,
    pub logit_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub grammar: String,
    pub stop: Vec<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct LlamaTimings {
    pub predicted_ms: f32,
    pub prompt_per_token_ms: f32,
    pub predicted_per_token_ms: f32,
    pub prompt_ms: f32,
    pub prompt_per_second: f32,
    pub predicted_n: f32,
    pub prompt_n: f32,
    pub predicted_per_second: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Stop {
    String(String),           // nullable: true
    StringArray(Vec<String>), // minItems: 1; maxItems: 4
}
