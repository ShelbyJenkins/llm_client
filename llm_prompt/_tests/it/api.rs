use super::*;
use llm_interface::llms::tokenizer::LlmTokenizer;
use llm_models::CloudLlm;

#[test]
fn test_api() -> crate::Result<()> {
    let model = CloudLlm::GPT_3_5_TURBO;
    let prompt = LlmPrompt::new_api_prompt(
        std::sync::Arc::new(LlmTokenizer::load_cloud_tokenizer(
            &model.provider,
            model.model_id(),
        )?),
        Some(model.tokens_per_message),
        model.tokens_per_name,
    );

    prompt.add_system_message()?.set_content(SYSTEM_PROMPT_1);
    prompt.add_user_message()?.set_content(USER_PROMPT_1);
    prompt
        .add_assistant_message()?
        .set_content(ASSISTANT_PROMPT_1);
    prompt.add_user_message()?.set_content(USER_PROMPT_2);

    let test_api: Vec<HashMap<String, String>> = prompt.api_prompt()?.get_built_prompt()?;
    println!("{prompt}");
    let result_api = vec![
        HashMap::from([
            ("content".to_string(), SYSTEM_PROMPT_1.to_string()),
            ("role".to_string(), "system".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), USER_PROMPT_1.to_string()),
            ("role".to_string(), "user".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), ASSISTANT_PROMPT_1.to_string()),
            ("role".to_string(), "assistant".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), USER_PROMPT_2.to_string()),
            ("role".to_string(), "user".to_string()),
        ]),
    ];
    assert_eq!(test_api, result_api);

    let token_count: u64 = prompt.api_prompt()?.get_total_prompt_tokens()?;
    assert_eq!(36, token_count);
    Ok(())
}
