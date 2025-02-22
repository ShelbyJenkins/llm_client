use super::*;

#[tokio::test]
#[serial]
pub async fn api_backends() -> crate::Result<()> {
    let llm_client = LlmClient::openai().gpt_3_5_turbo().init()?;
    basic_completion_tests::basic_completion_integration_tester(&llm_client).await?;
    let llm_client = LlmClient::anthropic().claude_3_5_haiku().init()?;
    basic_completion_tests::basic_completion_integration_tester(&llm_client).await?;
    let llm_client = LlmClient::perplexity().sonar().init()?;
    basic_completion_tests::basic_completion_integration_tester(&llm_client).await?;
    Ok(())
}
