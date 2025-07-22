use std::sync::Arc;

use crate::PromptTokenizer;

pub(crate) fn total_prompt_tokens_openai_format(
    built_prompt_messages: &Vec<std::collections::HashMap<String, String>>,
    tokens_per_message: Option<u64>,
    tokens_per_name: Option<i64>,
    tokenizer: &Arc<dyn PromptTokenizer>,
) -> u64 {
    let tokens_per_message = tokens_per_message.unwrap_or(0);
    let mut num_tokens = 0;
    for message in built_prompt_messages {
        num_tokens += tokens_per_message;

        for (key, value) in message.iter() {
            num_tokens += tokenizer.count_tokens(value) as u64;
            if let Some(tokens_per_name) = tokens_per_name {
                if key == "name" {
                    if tokens_per_name < 0 {
                        // Handles cases for certain models where name doesn't count towards token count
                        num_tokens -= tokens_per_name.unsigned_abs();
                    } else {
                        num_tokens += tokens_per_name as u64;
                    }
                }
            }
        }
    }
    num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
    num_tokens
}
