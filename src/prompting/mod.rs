pub mod logit_bias;
use crate::{providers::llama_cpp::models::LlamaPromptFormat, text_utils};
use core::panic;
use std::{collections::HashMap, fs::File, io::Read, path::Path};

/// Loads the system prompt template based on the provided base prompt and prompt template path.
///
/// If both `base_prompt` and `prompt_template_path` are `None`, the function will panic.
///
/// # Arguments
///
/// * `base_prompt` - An optional string slice representing the base prompt.
///                  If provided, it will be included in the system prompt.
/// * `prompt_template_path` - An optional string slice representing the path to the prompt template file.
///                            If provided, the contents of the file will be read and included in the system prompt.
///
/// # Returns
///
/// A `String` containing the system prompt template.
///
/// # Panics
///
/// * If both `base_prompt` and `prompt_template_path` are `None`.
/// * If the specified `prompt_template_path` is empty or fails to be read.
///
/// # Examples
///
/// ```
/// let base_prompt = "This is the base prompt.";
/// let prompt_template_path = "path/to/prompt/template.txt";
/// let system_prompt = load_system_prompt_template(Some(base_prompt), Some(prompt_template_path));
/// asset_eq!(system_prompt, "Base Prompt: This is the base prompt.\nUser Prompt: This is the user prompt.");
/// ```
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
        crate::LlmDefinition::LlamaLlm(model_definition) => match model_definition.prompt_format {
            LlamaPromptFormat::Mistral7BChat => {
                convert_prompt_to_zephyr_chat(prompt_with_default_formatting)
            }
            LlamaPromptFormat::Mistral7BInstruct => {
                convert_prompt_to_zephyr_instruct(prompt_with_default_formatting)
            }
            LlamaPromptFormat::Mixtral8X7BInstruct => {
                convert_prompt_to_mixtral_instruct(prompt_with_default_formatting)
            }
            LlamaPromptFormat::SOLAR107BInstructv1 => {
                convert_prompt_to_upstage_instruct(prompt_with_default_formatting)
            }
            _ => panic!(
                "Unsupported prompt format: {:?}",
                model_definition.prompt_format
            ),
        },
    }
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

pub fn token_count_of_openai_prompt(
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
