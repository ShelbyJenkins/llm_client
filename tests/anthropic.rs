mod common;
use common::*;
pub use std::time::Duration;
pub use tokio::time::sleep;

#[tokio::test]
#[serial]
async fn anthropic_integration_test() -> Result<()> {
    println!("Running anthropic integration tests");
    println!("Testing Claude 3 Haiku");
    let llm_client = LlmClient::anthropic_backend().claude_3_haiku().init()?;
    decider::decider_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    text::text_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    println!("Testing Claude 3 Opus");
    let llm_client = LlmClient::anthropic_backend().claude_3_opus().init()?;
    decider::decider_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    text::text_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    println!("Testing Claude 3 Sonnet");
    let llm_client = LlmClient::anthropic_backend().claude_3_sonnet().init()?;
    decider::decider_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    text::text_tests(&llm_client).await?;
    sleep(Duration::from_secs(1)).await;
    Ok(())
}
