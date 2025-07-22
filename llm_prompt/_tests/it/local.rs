use llm_interface::llms::LocalLlmConfig;
use llm_models::llm_types::local::gguf::preset_models::GgufPresetTrait;

use super::*;

#[test]
fn test_local() -> crate::Result<()> {
    let backend = LlmBackend::llama_cpp(
        None,
        Some(LocalLlmConfig::default().llama_3_1_8b_instruct()),
    )?;

    let prompt = backend.new_prompt()?;

    prompt.add_user_message()?.set_content(USER_PROMPT_1);
    prompt
        .add_assistant_message()?
        .set_content(ASSISTANT_PROMPT_1);
    prompt.add_user_message()?.set_content(USER_PROMPT_2);

    let test_local = prompt.local_prompt()?.get_built_prompt()?;
    println!("{prompt}",);
    assert_eq!(
            test_local,
            format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{USER_PROMPT_1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ASSISTANT_PROMPT_1}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{USER_PROMPT_2}<|eot_id|>")
        );
    let token_count = prompt.local_prompt()?.get_total_prompt_tokens()?;
    let prompt_as_tokens = prompt.local_prompt()?.get_built_prompt_as_tokens()?;
    assert_eq!(49, token_count);
    assert_eq!(token_count, prompt_as_tokens.len() as u64);

    prompt.set_generation_prefix("Generating 12345:");
    let test_local = prompt.local_prompt()?.get_built_prompt()?;
    println!("{prompt}");
    assert_eq!(
            test_local,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntell me a joke<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nthe clouds<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nfunny<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nGenerating 12345:"
        );
    let token_count = prompt.local_prompt()?.get_total_prompt_tokens()?;
    let prompt_as_tokens = prompt.local_prompt()?.get_built_prompt_as_tokens()?;
    assert_eq!(58, token_count);
    assert_eq!(token_count, prompt_as_tokens.len() as u64);
    Ok(())
}

#[test]
fn test_local_templates() -> crate::Result<()> {
    let expected_outputs = [
                // mistralai/Mistral-7B-Instruct-v0.3
                format!("<s>[INST] {USER_PROMPT_1} [/INST]{ASSISTANT_PROMPT_1}</s>[INST] {USER_PROMPT_2} [/INST]{ASSISTANT_PROMPT_2}</s>[INST] {USER_PROMPT_3} [/INST]"),
                // phi/Phi-3-mini-4k-instruct
                format!("<s><|user|>\n{USER_PROMPT_1}<|end|>\n<|assistant|>\n{ASSISTANT_PROMPT_1}<|end|>\n<|user|>\n{USER_PROMPT_2}<|end|>\n<|assistant|>\n{ASSISTANT_PROMPT_2}<|end|>\n<|user|>\n{USER_PROMPT_3}<|end|>\n<|assistant|>\n"),
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
        LlmBackend::llama_cpp(
            None,
            Some(LocalLlmConfig::default().mistral_7b_instruct_v0_3()),
        )?,
        LlmBackend::llama_cpp(
            None,
            Some(LocalLlmConfig::default().phi_3_mini_4k_instruct()),
        )?,
    ];

    for (i, backend) in templates.iter().enumerate() {
        let chat_template = backend.chat_template();
        let res = apply_chat_template(
            &messages,
            &chat_template.chat_template,
            chat_template.bos_token.as_deref(),
            &chat_template.eos_token,
            chat_template.unk_token.as_deref(),
        );

        assert_eq!(res, expected_outputs[i]);
    }
    Ok(())
}
