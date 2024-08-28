use super::constraints::stop_sequence::StoppingSequence;
use crate::llm_backends::llama_cpp::api::types::LlamaResponse;
use anyhow::Result;
use async_openai::types::{CreateChatCompletionResponse, FinishReason};
use clust::messages::{MessagesResponseBody, StopReason};
#[cfg(feature = "mistralrs_backend")]
use mistralrs::CompletionResponse as MistralCompletionResponse;
use thiserror::Error;

#[derive(Debug)]
pub struct LlmClientResponse {
    pub content: String,
    pub error: Option<LlmClientResponseError>,
    pub stop_reason: LlmClientResponseStopReason,
}

impl LlmClientResponse {
    pub fn new_from_llama(
        res: LlamaResponse,
        stop_sequence: Option<StoppingSequence>,
    ) -> Result<Self, LlmClientResponseError> {
        let stop_reason = if res.stopped_eos {
            LlmClientResponseStopReason::Eos
        } else if res.stopped_limit {
            LlmClientResponseStopReason::StopLimit
        } else if res.stopped_word {
            if let Some(stop_sequence) = stop_sequence {
                LlmClientResponseStopReason::StoppingSequence(stop_sequence)
            } else {
                return Err(LlmClientResponseError::StopSequenceReasonWithoutValue {
                    error: "StopReason::StopSequence but no stop_sequence value provided"
                        .to_owned(),
                });
            }
        } else {
            LlmClientResponseStopReason::Unknown
        };
        Ok(Self {
            content: res.content.to_owned(),
            error: None,
            stop_reason,
        })
    }

    #[cfg(feature = "mistralrs_backend")]
    pub fn new_from_mistral(res: MistralCompletionResponse) -> Self {
        Self {
            content: res.choices[0].text.to_owned(),
            stop_reason: LlmClientResponseStopReason::Unknown,
            error: None,
        }
    }

    pub fn new_from_openai(
        res: CreateChatCompletionResponse,
    ) -> Result<Self, LlmClientResponseError> {
        let choice = if res.choices.is_empty() {
            return Err(LlmClientResponseError::InferenceError {
                error: "OpenAiBackend completion_request error: completion.content.is_empty()"
                    .to_owned(),
            });
        } else if res.choices[0].message.content.is_none() {
            return Err(LlmClientResponseError::InferenceError {
                error: "OpenAiBackend completion_request error: completion.choices[0].message.content.is_none()"
                    .to_owned(),
            });
        } else {
            &res.choices[0]
        };
        let stop_reason = match choice.finish_reason {
            Some(FinishReason::Stop) => LlmClientResponseStopReason::Eos,
            Some(FinishReason::Length) => LlmClientResponseStopReason::StopLimit,
            _ => unreachable!(),
        };
        Ok(Self {
            content: choice.message.content.as_ref().unwrap().to_owned(),
            stop_reason,
            error: None,
        })
    }

    pub fn new_from_anthropic(
        res: &MessagesResponseBody,
        content: &str,
        stop_sequence: Option<StoppingSequence>,
    ) -> Result<Self, LlmClientResponseError> {
        let stop_reason = if let Some(stop_reason) = res.stop_reason {
            match stop_reason {
                StopReason::EndTurn => LlmClientResponseStopReason::Eos,
                StopReason::StopSequence => {
                    if let Some(stop_sequence) = stop_sequence {
                        LlmClientResponseStopReason::StoppingSequence(stop_sequence)
                    } else {
                        return Err(LlmClientResponseError::StopSequenceReasonWithoutValue {
                            error: "StopReason::StopSequence but no stop_sequence value provided"
                                .to_owned(),
                        });
                    }
                }
                StopReason::MaxTokens => LlmClientResponseStopReason::StopLimit,
                _ => unreachable!(),
            }
        } else {
            LlmClientResponseStopReason::Unknown
        };
        Ok(Self {
            content: content.to_owned(),
            error: None,
            stop_reason,
        })
    }
}

#[derive(Debug, PartialEq)]
pub enum LlmClientResponseStopReason {
    Eos,
    StoppingSequence(StoppingSequence),
    StopLimit,
    Unknown,
}

#[derive(Error, Debug)]
pub enum LlmClientResponseError {
    #[error("RequestBuilderError: {error}")]
    RequestBuilderError { error: String },
    #[error("InferenceError: {error}")]
    InferenceError { error: String },
    #[error("StopSequenceError: {error}")]
    StopSequenceError { error: String },
    #[error("StopSequenceReasonWithoutValue: {error}")]
    StopSequenceReasonWithoutValue { error: String },
    #[error("StopLimitError: {error}")]
    StopLimitError { error: String },
    #[error("UnknownStopReason: {error}")]
    UnknownStopReason { error: String },
}
