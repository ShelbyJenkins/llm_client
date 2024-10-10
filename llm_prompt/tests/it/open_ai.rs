#[allow(unused_imports)]
pub(crate) use anyhow::{anyhow, bail, Error, Result};
use llm_models::api_model::ApiLlmModel;
use llm_prompt::LlmPrompt;
use std::collections::HashMap;

#[test]
fn test_openai() -> crate::Result<()> {
    let model = ApiLlmModel::gpt_3_5_turbo();
    let prompt = LlmPrompt::new_openai_prompt(
        Some(model.tokens_per_message),
        model.tokens_per_name,
        model.model_base.tokenizer.clone(),
    );

    prompt
        .add_system_message()?
        .set_content("test system content");
    prompt
        .add_user_message()?
        .set_content("test user content 1");
    prompt
        .add_assistant_message()?
        .set_content("test assistant content");
    prompt
        .add_user_message()?
        .set_content("test user content 2");

    let test_openai = prompt.get_built_prompt_hashmap()?;
    println!("{prompt}");
    let result_openai = vec![
        HashMap::from([
            ("content".to_string(), "test system content".to_string()),
            ("role".to_string(), "system".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), "test user content 1".to_string()),
            ("role".to_string(), "user".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), "test assistant content".to_string()),
            ("role".to_string(), "assistant".to_string()),
        ]),
        HashMap::from([
            ("content".to_string(), "test user content 2".to_string()),
            ("role".to_string(), "user".to_string()),
        ]),
    ];
    assert_eq!(test_openai, result_openai);

    let token_count = prompt.get_total_prompt_tokens()?;
    assert_eq!(39, token_count);
    Ok(())
}
