pub mod logit_bias;
use crate::text_utils;
use crate::LlamaLlm;
use crate::LlamaLlmModels;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn load_system_prompt_template(
    base_prompt: Option<&str>,
    prompt_template_path: Option<&str>,
) -> String {
    if base_prompt.is_none() && prompt_template_path.is_none() {
        panic!("base_prompt and prompt_template_path cannot both be None.");
    }
    let mut system_prompt = String::new();
    if let Some(base_prompt) = base_prompt {
        system_prompt.push_str(&format!("Base Prompt: {}", base_prompt));
    }

    if let Some(template_path) = prompt_template_path {
        let path = Path::new(&template_path);
        match File::open(path) {
            Ok(mut file) => {
                let mut custom_prompt = String::new();
                match file.read_to_string(&mut custom_prompt) {
                    Ok(_) => {
                        if custom_prompt.trim().is_empty() {
                            panic!("prompt_template_path '{}' is empty.", path.display())
                        }
                        system_prompt.push_str(&format!("\nUser Prompt: {}", custom_prompt));
                    }
                    Err(e) => panic!("Failed to read file: {}", e),
                }
            }
            Err(e) => panic!("Failed to open file: {}", e),
        }
    };
    system_prompt
}

pub fn create_prompt_with_default_formatting(
    system_prompt: String,
    feature: Option<&str>,
    user_input: Option<&str>,
    // context_docs: Option<Vec<RetrievalDoc>>,
) -> HashMap<String, HashMap<String, String>> {
    let mut user_prompt = String::new();
    if feature.is_some() || user_input.is_some() {
        if let Some(input) = feature {
            user_prompt.push_str(&format!("\nFeature: {}", input));
        };
        if let Some(input) = user_input {
            user_prompt.push_str(&format!("\nUser Input: {}", input));
        };
    }
    HashMap::from([
        ("system".to_string(), create_message(None, system_prompt)),
        ("user".to_string(), create_message(None, user_prompt)),
    ])
}

fn create_message(name: Option<String>, content: String) -> HashMap<String, String> {
    HashMap::from([
        ("content".to_string(), content),
        ("name".to_string(), name.unwrap_or("".to_string())),
    ])
}

pub fn create_model_formatted_prompt(
    llm_definition: &crate::LlmDefinition,
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    match llm_definition {
        crate::LlmDefinition::OpenAiLlm(_) => prompt_with_default_formatting,
        crate::LlmDefinition::LlamaLlm(LlamaLlmModels::Mistral7BChat(_)) => {
            convert_prompt_to_zephyr_chat(prompt_with_default_formatting)
        }
        crate::LlmDefinition::LlamaLlm(LlamaLlmModels::Mistral7BInstruct(_)) => {
            convert_prompt_to_zephyr_instruct(prompt_with_default_formatting)
        }
        crate::LlmDefinition::LlamaLlm(LlamaLlmModels::Mixtral8X7BInstruct(_)) => {
            convert_prompt_to_mixtral_instruct(prompt_with_default_formatting)
        }
        crate::LlmDefinition::LlamaLlm(LlamaLlmModels::SOLAR107BInstructv1(_)) => {
            convert_prompt_to_upstage_instruct(prompt_with_default_formatting)
        }
    }
}

fn convert_prompt_to_vicuna(
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    todo!()
    // See https://github.com/abetlen/llama-cpp-python/pull/711/files
}

fn convert_prompt_to_zephyr_chat(
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    let mut prompt_string = String::new();
    prompt_string.push_str("<|system|>\n");
    prompt_string.push_str(&prompt_with_default_formatting["system"]["content"]);

    prompt_string.push_str(" </s>\n");
    prompt_string.push_str("<|user|>\n");
    prompt_string.push_str(&prompt_with_default_formatting["user"]["content"]);
    prompt_string.push_str(" </s>");
    prompt_string.push_str("<|assistant|>");
    HashMap::from([(
        "llama_prompt".to_string(),
        HashMap::from([("content".to_string(), prompt_string)]),
    )])
}
fn convert_prompt_to_zephyr_instruct(
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    let mut prompt_string = String::new();
    prompt_string.push_str("<s>[INST] ");
    prompt_string.push_str(&prompt_with_default_formatting["system"]["content"]);
    prompt_string.push_str(&prompt_with_default_formatting["user"]["content"]);
    prompt_string.push_str(" [/INST]");
    HashMap::from([(
        "llama_prompt".to_string(),
        HashMap::from([("content".to_string(), prompt_string)]),
    )])
}
// The same as zephyr_instruct but without the <s>
fn convert_prompt_to_mixtral_instruct(
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    let mut prompt_string = String::new();
    prompt_string.push_str("[INST] ");
    prompt_string.push_str(&prompt_with_default_formatting["system"]["content"]);
    prompt_string.push_str(&prompt_with_default_formatting["user"]["content"]);
    prompt_string.push_str(" [/INST]");
    HashMap::from([(
        "llama_prompt".to_string(),
        HashMap::from([("content".to_string(), prompt_string)]),
    )])
}
fn convert_prompt_to_upstage_instruct(
    prompt_with_default_formatting: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, HashMap<String, String>> {
    let mut prompt_string = String::new();
    prompt_string.push_str("<s> ### User:\n");
    prompt_string.push_str(&prompt_with_default_formatting["system"]["content"]);
    prompt_string.push_str(&prompt_with_default_formatting["user"]["content"]);

    HashMap::from([(
        "llama_prompt".to_string(),
        HashMap::from([("content".to_string(), prompt_string)]),
    )])
}

async fn get_prompt_length(
    llm_definition: &crate::LlmDefinition,
    prompt: &HashMap<String, HashMap<String, String>>,
    model_params: &crate::LlmModelParams,
) -> u16 {
    match llm_definition {
        crate::LlmDefinition::OpenAiLlm(_) => token_count_of_openai_prompt(prompt, model_params),
        crate::LlmDefinition::LlamaLlm(_) => LlamaLlm::default()
            .llama_cpp_count_tokens(&prompt["llama_prompt"]["content"])
            .await
            .unwrap(),
    }
}

fn token_count_of_openai_prompt(
    prompt: &HashMap<String, HashMap<String, String>>,
    model_params: &crate::LlmModelParams,
) -> u16 {
    let mut num_tokens = 0;
    for message in prompt.values() {
        num_tokens += model_params.tokens_per_message;
        num_tokens += text_utils::tiktoken_len(&message["content"]);

        if message["name"].is_empty() {
            if model_params.tokens_per_name < 0 {
                // Handles cases for certain models where name doesn't count towards token count
                num_tokens -= model_params.tokens_per_name.unsigned_abs();
            } else {
                num_tokens += model_params.tokens_per_name as u16;
            }
        }
    }

    num_tokens += 3; // every reply is primed with <|start|>assistant<|message|>
    num_tokens
}

// Checking to ensure we don't exceed the max tokens for the model for text generation
// Important for OpenAI as API requests will fail if
// input tokens + requested output tokens (max_tokens) exceeds
// the max tokens for the model
pub async fn check_available_request_tokens_generation(
    llm_definition: &crate::LlmDefinition,
    prompt: &HashMap<String, HashMap<String, String>>,
    context_to_response_ratio: f32,
    model_token_utilization: f32,
    model_params: &crate::LlmModelParams,
) -> (u16, u16) {
    let total_prompt_tokens = get_prompt_length(llm_definition, prompt, model_params).await;
    // Calculate available tokens for response
    let available_tokens = model_params.max_tokens_for_model - total_prompt_tokens;
    let mut max_response_tokens =
        (available_tokens as f32 * (model_token_utilization)).ceil() as u16;
    // if context_to_response_ratio > 0.0 {
    //     let mut max_response_tokens =
    //         (available_tokens as f32 * context_to_response_ratio).ceil() as u16;

    // }

    //  for safety in case of model changes
    while max_response_tokens > (model_params.max_tokens_for_model - model_params.safety_tokens) {
        max_response_tokens -= 1
    }
    (total_prompt_tokens, max_response_tokens)
}

// Checking to ensure we don't exceed the max tokens for the model for decision making
pub async fn check_available_request_tokens_decision(
    llm_definition: &crate::LlmDefinition,
    prompt: &HashMap<String, HashMap<String, String>>,
    logit_bias_response_tokens: u16,
    model_params: &crate::LlmModelParams,
) -> u16 {
    // llm_model_instance = llm_definition.llm_model_instance
    let total_prompt_tokens = get_prompt_length(llm_definition, prompt, model_params).await;
    let max_response_tokens = total_prompt_tokens + logit_bias_response_tokens;
    //  for safety in case of model changes
    if max_response_tokens > model_params.max_tokens_for_model - model_params.safety_tokens {
        panic!(
            "max_response_tokens {} is greater than available_tokens {}",
            max_response_tokens,
            model_params.max_tokens_for_model - model_params.safety_tokens
        );
    }
    total_prompt_tokens
}
