mod common;
use common::*;

#[tokio::test]
#[serial]
async fn openai_integration_test() -> Result<()> {
    let llm_client = LlmClient::openai_backend().gpt_3_5_turbo().init()?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::openai_backend().gpt_4().init()?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::openai_backend().gpt_4_o().init()?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::openai_backend().gpt_4_turbo().init()?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    Ok(())
}
