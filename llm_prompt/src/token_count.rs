use std::sync::Arc;

use crate::PromptTokenizer;
use thiserror::Error;

pub const DEFAULT_SAFETY_TOKENS: usize = 10;

/// Sets and validates the 'max_tokens' or 'n_ctx' or 'n_predict' parameter for a request.
/// First, it checks that the total_prompt_tokens is less than the ctx_size - safety_tokens.
/// Then returns 'available_tokens' as the lower of either:
/// ctx_size - total_prompt_tokens - safety_tokens or if it's provided, inference_ctx_size.
/// If 'requested_tokens' is provided, 'requested_tokens' is returned if less than 'available_tokens'.
/// If 'requested_tokens' is 'None' or 'requested_tokens' is greater than 'available_tokens',
/// 'available_tokens' is returned.
///
/// # Arguments
///
/// * `ctx_size` - The total context length for the for the model or system.
/// * `inference_ctx_size` - Optional output size for models with output generation limits. Defaults to None.
/// * `total_prompt_tokens` - The total prompt tokens as an unsigned 32-bit integer.
/// * `safety_tokens` - Optional padding. Defaults to 10.
/// * `requested_tokens` - Optional 'max_tokens' for the response. Defaults to 'available_tokens'.
///
/// # Returns
///
/// A usize to be used for the 'max_tokens' or 'n_ctx' parameter for inference requests.
///
/// # Errors
///
/// Returns an error if any of the validation checks fail.
pub fn check_and_get_max_tokens(
    ctx_size: usize,
    inference_ctx_size: Option<usize>,
    total_prompt_tokens: usize,
    safety_tokens: Option<usize>,
    requested_tokens: Option<usize>,
) -> Result<usize, RequestTokenLimitError> {
    let available_tokens = available_tokens(
        ctx_size,
        inference_ctx_size,
        total_prompt_tokens,
        safety_tokens,
    )?;
    let requested_tokens = if let Some(requested_tokens) = requested_tokens {
        if requested_tokens > available_tokens {
            crate::error!(
                "requested_tokens ({requested_tokens}) is greater than available_tokens ({}). Using available_tokens for request.", available_tokens
            );
            available_tokens
        } else {
            requested_tokens
        }
    } else {
        available_tokens
    };

    if total_prompt_tokens + requested_tokens >= ctx_size {
        panic!(
            "total_prompt_tokens ({total_prompt_tokens}) + requested_tokens ({requested_tokens}) >= ctx_size ({ctx_size}). This should never happen.",
        );
    }
    Ok(requested_tokens)
}

fn available_tokens(
    ctx_size: usize,
    inference_ctx_size: Option<usize>,
    total_prompt_tokens: usize,
    safety_tokens: Option<usize>,
) -> Result<usize, RequestTokenLimitError> {
    let safety_tokens = safety_tokens.unwrap_or(DEFAULT_SAFETY_TOKENS);

    if total_prompt_tokens as usize >= ctx_size - safety_tokens {
        return Err(RequestTokenLimitError::PromptTokensExceeds {
            total_prompt_tokens,
            ctx_size: ctx_size - safety_tokens,
        });
    }

    let available_tokens = if let Some(inference_ctx_size) = inference_ctx_size {
        std::cmp::min(ctx_size - total_prompt_tokens, inference_ctx_size) - safety_tokens
    } else {
        ctx_size - total_prompt_tokens - safety_tokens
    };
    if available_tokens == 0 {
        panic!("available_tokens == 0. This should never happen.",);
    }
    Ok(available_tokens - safety_tokens)
}

pub(crate) fn total_prompt_tokens_openai_format(
    built_prompt_messages: &Vec<std::collections::HashMap<String, String>>,
    tokens_per_message: Option<usize>,
    tokens_per_name: Option<isize>,
    tokenizer: &Arc<dyn PromptTokenizer>,
) -> usize {
    let tokens_per_message = tokens_per_message.unwrap_or(0);
    let mut num_tokens = 0;
    for message in built_prompt_messages {
        num_tokens += tokens_per_message;

        for (key, value) in message.iter() {
            num_tokens += tokenizer.count_tokens(value) as usize;
            if let Some(tokens_per_name) = tokens_per_name {
                if key == "name" {
                    if tokens_per_name < 0 {
                        // Handles cases for certain models where name doesn't count towards token count
                        num_tokens -= tokens_per_name.unsigned_abs();
                    } else {
                        num_tokens += tokens_per_name as usize;
                    }
                }
            }
        }
    }
    num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
    num_tokens
}

#[derive(Debug, Clone)]
pub struct MaxTokenState {
    pub actual_request: usize,
    pub requested_response: usize,
}

#[derive(Error, Debug, Clone)]
pub enum RequestTokenLimitError {
    #[error("total_prompt_tokens ({total_prompt_tokens}) exceeds ctx_size ({ctx_size})")]
    PromptTokensExceeds {
        total_prompt_tokens: usize,
        ctx_size: usize,
    },
    #[error("GenericPromptError: {e}")]
    GenericPromptError { e: String },
    #[error("PromptTokensNotSet: Prompt tokens not set.")]
    PromptTokensNotSet,
    #[error(
        "TokenLimitIncreaseError: initial_state: {:?}, new_state: {:?}",
        initial_state,
        new_state
    )]
    TokenLimitIncreaseError {
        initial_state: MaxTokenState,
        new_state: MaxTokenState,
    },
}
