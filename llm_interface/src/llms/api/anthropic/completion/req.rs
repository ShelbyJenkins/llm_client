use crate::requests::completion::{
    error::CompletionError, request::CompletionRequest, ToolChoice, ToolDefinition,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Default, Debug, Deserialize)]
pub struct AnthropicCompletionRequest {
    /// ID of the model to use.
    ///
    /// See [models](https://docs.anthropic.com/claude/docs/models-overview) for additional details and options.
    pub model: String,

    /// Input messages.
    ///
    /// Our models are trained to operate on alternating user and assistant conversational turns. When creating a new Message, you specify the prior conversational turns with the messages parameter, and the model then generates the next Message in the conversation.
    ///
    /// See [examples](https://docs.anthropic.com/claude/reference/messages-examples) for more input examples.
    ///
    /// Note that if you want to include a [system prompt](https://docs.anthropic.com/claude/docs/system-prompts), you can use the top-level system parameter — there is no "system" role for input messages in the Messages API.
    pub messages: Vec<CompletionRequestMessage>,

    /// The maximum number of tokens to generate before stopping.
    ///
    /// Note that our models may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate.
    ///
    /// Different models have different maximum values for this parameter. See [models](https://docs.anthropic.com/claude/docs/models-overview) for details.
    pub max_tokens: u64,

    /// Custom text sequences that will cause the model to stop generating.
    ///
    /// Our models will normally stop when they have naturally completed their turn, which will result in a response stop_reason of "end_turn".
    ///
    /// If you want the model to stop generating when it encounters custom strings of text, you can use the stop_sequences parameter. If the model encounters one of the custom sequences, the response stop_reason value will be "stop_sequence" and the response stop_sequence value will contain the matched stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// System prompt.
    ///
    /// A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role. See our [guide to system prompts](https://docs.anthropic.com/claude/docs/system-prompts).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Amount of randomness injected into the response.
    ///
    /// Defaults to 0.5. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks.
    ///
    /// Note that even with temperature of 0.0, the results will not be fully deterministic.
    pub temperature: f32,

    /// min: 0.0, max: 1.0, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// The tools for the request, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// The tool choice for the request, default: None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl AnthropicCompletionRequest {
    pub fn new(req: &CompletionRequest) -> crate::Result<Self, CompletionError> {
        let mut messages = Vec::new();
        let mut system_prompt = None;
        match req.prompt.get_built_prompt_hashmap() {
            Ok(prompt_message) => {
                for m in prompt_message {
                    let role = m.get("role").ok_or_else(|| {
                        CompletionError::RequestBuilderError("Role not found".to_string())
                    })?;
                    let content = m.get("content").ok_or_else(|| {
                        CompletionError::RequestBuilderError("Content not found".to_string())
                    })?;

                    match role.as_str() {
                        "user" | "assistant" => messages.push(CompletionRequestMessage {
                            role: role.to_string(),
                            content: content.to_string(),
                        }),
                        "system" => system_prompt = Some(content.to_string()),
                        _ => {
                            return Err(CompletionError::RequestBuilderError(format!(
                                "Role {} not supported",
                                role
                            )))
                        }
                    }
                }
            }
            Err(e) => {
                return Err(CompletionError::RequestBuilderError(format!(
                    "Error building prompt: {}",
                    e
                )))
            }
        }

        let stop = req.stop_sequences.to_vec();
        let stop_sequences = if stop.is_empty() { None } else { Some(stop) };

        Ok(AnthropicCompletionRequest {
            model: req.backend.model_id().to_owned(),
            messages,
            max_tokens: req.config.actual_request_tokens.unwrap(),
            stop_sequences,
            system: system_prompt,
            temperature: temperature(req.config.temperature)?,
            top_p: top_p(req.config.top_p)?,
            tools: if !req.tools.is_empty() {
                Some(req.tools.clone())
            } else {
                None
            },
            tool_choice: if !req.tools.is_empty() {
                Some(req.tool_choice.clone())
            } else {
                None
            },
        })
    }
}

/// Convert the native temperature from 0.0 to 2.0 to 0.0 to 1.0
fn temperature(value: f32) -> crate::Result<f32, CompletionError> {
    if (0.0..=2.0).contains(&value) {
        Ok(value / 2.0)
    } else {
        Err(CompletionError::RequestBuilderError(
            "Temperature must be between 0.0 and 2.0".to_string(),
        ))
    }
}

fn top_p(value: Option<f32>) -> crate::Result<Option<f32>, CompletionError> {
    match value {
        Some(v) => {
            if (0.0..=1.0).contains(&v) {
                Ok(Some(v))
            } else {
                Err(CompletionError::RequestBuilderError(
                    "Top p must be between 0.0 and 1.0".to_string(),
                ))
            }
        }
        None => Ok(None),
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CompletionRequestMessage {
    pub role: String,
    pub content: String,
}
