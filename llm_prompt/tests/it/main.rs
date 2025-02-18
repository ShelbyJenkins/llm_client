//
mod api;
mod local;

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};
use llm_models::LocalLlmModel;
use llm_prompt::{apply_chat_template, LlmPrompt, PromptMessages};
use serde_json;
use std::collections::HashMap;
use std::fs;
use tempfile::NamedTempFile;

const SYSTEM_PROMPT_1: &str = "We are bad joke robots";
const USER_PROMPT_1: &str = "tell me a joke";
const ASSISTANT_PROMPT_1: &str = "the clouds";
const USER_PROMPT_2: &str = "funny";
const ASSISTANT_PROMPT_2: &str = "beepboop";
const USER_PROMPT_3: &str = "robot?";

#[test]
fn test_serde() -> crate::Result<()> {
    let model = LocalLlmModel::default();
    let prompt = LlmPrompt::new_local_prompt(
        model.model_base.tokenizer.clone(),
        &model.chat_template.chat_template,
        model.chat_template.bos_token.as_deref(),
        &model.chat_template.eos_token,
        model.chat_template.unk_token.as_deref(),
        model.chat_template.base_generation_prefix.as_deref(),
    );

    prompt.add_user_message()?.set_content(USER_PROMPT_1);
    prompt
        .add_assistant_message()?
        .set_content(ASSISTANT_PROMPT_1);
    prompt.add_user_message()?.set_content(USER_PROMPT_2);

    prompt.local_prompt()?;

    // Write prompt to file
    let temp_file = NamedTempFile::new()?;
    let temp_path = temp_file.path();
    let initial_serialized_prompt = serde_json::to_string_pretty(&prompt)?;
    fs::write(temp_path, &initial_serialized_prompt)?;

    // Write messages to file
    let temp_file = NamedTempFile::new()?;
    let temp_path = temp_file.path();
    let serialized_messages = serde_json::to_string_pretty(&prompt.messages)?;
    fs::write(temp_path, serialized_messages)?;

    // Load messages from file
    let file_content = fs::read_to_string(temp_path)?;
    let deserialized_messages: PromptMessages = serde_json::from_str(&file_content)?;

    assert_eq!(
        format!("{}", prompt.messages),
        format!("{}", deserialized_messages)
    );

    Ok(())
}
