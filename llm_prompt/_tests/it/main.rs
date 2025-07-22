//
mod api;
mod local;

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};
use llm_interface::llms::LlmBackend;
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
    let backend = LlmBackend::llama_cpp(None, None)?;
    let prompt = backend.new_prompt()?;

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
