#[allow(unused_imports)]
pub(crate) use anyhow::{anyhow, bail, Error, Result};
use llm_models::local_model::{gguf::preset::LlmPreset, LocalLlmModel};
use llm_prompt::{apply_chat_template, LlmPrompt};
use std::collections::HashMap;

#[test]
fn test_chat() -> crate::Result<()> {
    let model = LocalLlmModel::default();
    let prompt = LlmPrompt::new_chat_template_prompt(
        &model.chat_template.chat_template,
        &model.chat_template.bos_token,
        &model.chat_template.eos_token,
        model.chat_template.unk_token.as_deref(),
        model.chat_template.base_generation_prefix.as_deref(),
        model.model_base.tokenizer.clone(),
    );

    prompt
        .add_user_message()?
        .set_content("test user content 1");
    prompt
        .add_assistant_message()?
        .set_content("test assistant content");
    prompt
        .add_user_message()?
        .set_content("test user content 2");

    let test_chat = prompt.get_built_prompt_string()?;
    println!("{prompt}",);
    assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|>"
        );
    let token_count = prompt.get_total_prompt_tokens()?;
    let prompt_as_tokens = prompt.get_built_prompt_as_tokens()?;
    assert_eq!(54, token_count);
    assert_eq!(token_count, prompt_as_tokens.len() as u64);

    prompt.set_generation_prefix("Generating 12345:");
    let test_chat = prompt.get_built_prompt_string()?;
    println!("{prompt}");
    assert_eq!(
            test_chat,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ntest assistant content<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntest user content 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerating 12345:"
        );
    let token_count = prompt.get_total_prompt_tokens()?;
    let prompt_as_tokens = prompt.get_built_prompt_as_tokens()?;
    assert_eq!(63, token_count);
    assert_eq!(token_count, prompt_as_tokens.len() as u64);
    Ok(())
}

const USER_PROMPT_1: &str = "tell me a joke";
const ASSISTANT_PROMPT_1: &str = "the clouds";
const USER_PROMPT_2: &str = "funny";
const ASSISTANT_PROMPT_2: &str = "beepboop";
const USER_PROMPT_3: &str = "robot?";

#[test]
fn test_chat_templates() -> crate::Result<()> {
    let expected_outputs = [
                // mistralai/Mistral-7B-Instruct-v0.3
                "<s>[INST] tell me a joke [/INST]the clouds</s>[INST] funny [/INST]beepboop</s>[INST] robot? [/INST]",
                // phi/Phi-3-mini-4k-instruct
                "<s><|user|>\ntell me a joke<|end|>\n<|assistant|>\nthe clouds<|end|>\n<|user|>\nfunny<|end|>\n<|assistant|>\nbeepboop<|end|>\n<|user|>\nrobot?<|end|>\n<|assistant|>\n",
        ];
    let messages: Vec<HashMap<String, String>> = vec![
        HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), USER_PROMPT_1.to_string()),
        ]),
        HashMap::from([
            ("role".to_string(), "assistant".to_string()),
            ("content".to_string(), ASSISTANT_PROMPT_1.to_string()),
        ]),
        HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), USER_PROMPT_2.to_string()),
        ]),
        HashMap::from([
            ("role".to_string(), "assistant".to_string()),
            ("content".to_string(), ASSISTANT_PROMPT_2.to_string()),
        ]),
        HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), USER_PROMPT_3.to_string()),
        ]),
    ];
    let templates = vec![
        LlmPreset::Mistral7bInstructV0_3.load()?.chat_template,
        LlmPreset::Phi3Mini4kInstruct.load()?.chat_template,
    ];

    for (i, chat_template) in templates.iter().enumerate() {
        let res = apply_chat_template(
            &messages,
            &chat_template.chat_template,
            &chat_template.bos_token,
            &chat_template.eos_token,
            chat_template.unk_token.as_deref(),
        );

        assert_eq!(res, expected_outputs[i]);
    }
    Ok(())
}
