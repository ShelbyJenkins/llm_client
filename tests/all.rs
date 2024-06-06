mod common;
use common::*;
pub use std::time::Duration;
pub use tokio::time::sleep;

#[tokio::test]
#[serial]
async fn test_all() -> Result<()> {
    println!("Running all integration tests");
    println!("Testing Llama 3 8b");
    let llm_client = LlmClient::llama_backend()
        .llama_3_8b_instruct()
        .init()
        .await?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;

    // println!("Testing GPT 3.5 Turbo");
    // let llm_client = LlmClient::openai_backend().gpt_3_5_turbo().init()?;
    // decider::decider_tests(&llm_client).await?;
    // text::text_tests(&llm_client).await?;

    // println!("Testing Claude 3 Haiku");
    // let llm_client = LlmClient::anthropic_backend().claude_3_haiku().init()?;
    // decider::decider_tests(&llm_client).await?;
    // sleep(Duration::from_secs(1)).await;
    // text::text_tests(&llm_client).await?;

    Ok(())
}
