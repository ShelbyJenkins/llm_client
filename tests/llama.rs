mod common;
use common::*;

#[tokio::test]
#[serial]
async fn llama_integration_test() -> Result<()> {
    let llm_client = LlmClient::llama_backend()
        .mistral_7b_instruct()
        .available_vram(48)
        .init()
        .await?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::llama_backend()
        .mixtral_8x7b_instruct()
        .available_vram(48)
        .init()
        .await?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::llama_backend()
        .llama_3_8b_instruct()
        .available_vram(48)
        .init()
        .await?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;
    let llm_client = LlmClient::llama_backend()
        .llama_3_70b_instruct()
        .available_vram(44)
        .init()
        .await?;
    decider::decider_tests(&llm_client).await?;
    text::text_tests(&llm_client).await?;

    Ok(())
}
