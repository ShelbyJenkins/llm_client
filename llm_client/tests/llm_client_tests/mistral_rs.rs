use super::*;

#[tokio::test]
#[serial]
pub async fn llama_cpp_integration_test() -> crate::Result<()> {
    let mut _builder = mistral_rs_tiny_llm().await?;

    Ok(())
}
