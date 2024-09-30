use super::*;

#[tokio::test]
#[serial]
pub async fn llama_cpp_integration_test() -> crate::Result<()> {
    let llm_client = llama_cpp_tiny_llm().await?;

    basic_completion_tests::basic_completion_integration_tester(&llm_client).await?;
    basic_completion_tests::basic_completion_logit_bias_integration_tester(&llm_client).await?;

    basic_primitive_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    basic_primitive_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;

    reason_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    reason_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;

    decision_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    decision_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;

    extract_tests::extract_urls_integration_tester(&llm_client, &TestLevel::IntegrationTest)
        .await?;
    Ok(())
}

#[ignore]
#[tokio::test]
#[serial]
pub async fn llama_cpp_basic_primitive_integration_test() -> crate::Result<()> {
    let llm_client = llama_cpp_tiny_llm().await?;
    basic_primitive_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    basic_primitive_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;
    Ok(())
}

#[ignore]
#[tokio::test]
#[serial]
pub async fn llama_cpp_reason_integration_test() -> crate::Result<()> {
    let llm_client = llama_cpp_tiny_llm().await?;
    reason_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    reason_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;
    Ok(())
}

#[ignore]
#[tokio::test]
#[serial]
pub async fn llama_cpp_decision_integration_test() -> crate::Result<()> {
    let llm_client = llama_cpp_tiny_llm().await?;
    decision_tests::run(&llm_client, &TestLevel::IntegrationTest).await?;
    decision_tests::run_optional(&llm_client, &TestLevel::IntegrationTest).await?;
    Ok(())
}
