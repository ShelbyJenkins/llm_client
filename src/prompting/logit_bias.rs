use crate::text_utils;
use crate::LlmClient;
use crate::ProviderClient;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;

pub async fn generate_logit_bias_from_chars(
    client: &ProviderClient,
    allowed_chars: Option<Vec<String>>,
    removed_chars: Option<Vec<String>>,
) -> Result<HashMap<String, serde_json::Value>, Box<dyn Error>> {
    let mut logit_bias = HashMap::new();
    match &client.llm_client {
        LlmClient::OpenAiLlm(_) => {
            if let Some(allowed_chars) = allowed_chars {
                let allowed_tokens = text_utils::get_char_tokens(&allowed_chars);
                for token in allowed_tokens {
                    logit_bias.insert(token.to_string(), json!(100));
                }
            }
            if let Some(removed_chars) = removed_chars {
                let removed_tokens = text_utils::get_char_tokens(&removed_chars);
                for token in removed_tokens {
                    logit_bias.insert(token.to_string(), json!(-100));
                }
            }
        }
        LlmClient::LlamaLlm(client) => {
            if let Some(allowed_chars) = allowed_chars {
                let allowed_tokens = client.tokenize_chars(&allowed_chars).await?;
                for token in allowed_tokens {
                    logit_bias.insert(token.to_string(), json!(100.0));
                }
            }
            if let Some(removed_chars) = removed_chars {
                let removed_tokens = client.tokenize_chars(&removed_chars).await?;
                for token in removed_tokens {
                    logit_bias.insert(token.to_string(), json!(-100.0));
                }
            }
        }
    }
    Ok(logit_bias)
}

pub fn generate_punctuation() -> Vec<String> {
    (['.', ',', ';', ':', '!', '?', '\'', '"'].iter())
        .map(|&c| c.to_string())
        .collect()
}

pub fn generate_bad_split_chars() -> Vec<String> {
    [
        "[]", "[", "]", "()", "(", ")", "•", "entry", "Entry", "story", "Story", "Feature",
        "feature",
    ]
    .iter()
    .map(|&c| c.to_string())
    .collect()

    // bad_chars.extend(('1'..='9').map(|c| c.to_string()));
    // bad_chars.extend((1..=9).map(|n| format!("{:02}", n)));
}

pub fn generate_whitespace_chars() -> Vec<String> {
    ([r"\t", r"\r", r"\v", r"\f"].iter())
        .map(|&c| c.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::providers::llama_cpp::models::{TEST_LLM_1_CHAT, TEST_LLM_2_INSTRUCT};
    use crate::providers::llama_cpp::server::kill_server;
    use crate::providers::llm_openai::models::OpenAiDef;
    use crate::LlmDefinition;

    #[tokio::test]
    async fn test_llm_openai() {
        let allowed_chars: Vec<String> = vec![
            "hello".to_string(),
            "there".to_string(),
            "general".to_string(),
        ];
        let removed_chars: Vec<String> = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let llm_client =
            ProviderClient::new(&LlmDefinition::OpenAiLlm(OpenAiDef::Gpt35Turbo), None).await;
        let response =
            generate_logit_bias_from_chars(&llm_client, Some(allowed_chars), Some(removed_chars))
                .await;
        eprintln!("{:?}", response.unwrap());
    }

    #[tokio::test]
    async fn test_llama_cpp() {
        let allowed_chars: Vec<String> = vec![
            "hello".to_string(),
            "there".to_string(),
            "general".to_string(),
        ];
        let removed_chars: Vec<String> = vec!["1".to_string(), "2".to_string()];

        let llm_definition = LlmDefinition::LlamaLlm((*TEST_LLM_1_CHAT).clone());

        let llm_client = ProviderClient::new(&llm_definition, None).await;
        let response = generate_logit_bias_from_chars(
            &llm_client,
            Some(allowed_chars.clone()),
            Some(removed_chars.clone()),
        )
        .await;
        eprintln!("{:?}", response.unwrap());
        let llm_definition = LlmDefinition::LlamaLlm((*TEST_LLM_2_INSTRUCT).clone());

        let llm_client = ProviderClient::new(&llm_definition, None).await;
        let response =
            generate_logit_bias_from_chars(&llm_client, Some(allowed_chars), Some(removed_chars))
                .await;
        eprintln!("{:?}", response.unwrap());
        kill_server();
    }
}
