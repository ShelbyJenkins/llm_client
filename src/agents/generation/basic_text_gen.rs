use core::panic;
use std::error::Error;

use crate::{prompting, LlmDefinition, ProviderClient};

pub async fn generate(
    llm_definition: &LlmDefinition,
    base_prompt: Option<&str>,
    user_input: Option<&str>,
    prompt_template_path: Option<&str>,
    model_token_utilization: Option<f32>,
) -> Result<String, Box<dyn Error>> {
    let llm_client = ProviderClient::new(llm_definition, None).await;
    let prompt = prompting::create_prompt_with_default_formatting(
        prompting::load_system_prompt_template(base_prompt, prompt_template_path),
        None,
        user_input,
    );

    let response = llm_client
        .generate_text(&prompt, &None, model_token_utilization, None, None)
        .await;
    if let Err(error) = response {
        panic!("Failed to generate text: {}", error);
    }

    response
}
