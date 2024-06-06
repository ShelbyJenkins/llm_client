use super::*;

pub async fn decider_tests(llm_client: &LlmClient) -> Result<()> {
    match llm_client.backend {
        LlmBackend::Llama(_) => {
            println!("Llama basic decider tests");
            decider_basic_tests(llm_client).await?;
            println!("Llama logit bias decider tests");
            decider_logit_bias_tests(llm_client).await?;
            println!("Llama grammar decider tests");
            decider_grammar_tests(llm_client).await?;
        }
        #[cfg(feature = "mistralrs_backend")]
        LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(_) => {
            println!("OpenAI decider tests");
            decider_basic_tests(llm_client).await?;
            println!("OpenAI logit bias decider tests");
            decider_logit_bias_tests(llm_client).await?;
        }
        LlmBackend::Anthropic(_) => {
            println!("Anthropic basic decider tests");
            decider_basic_tests(llm_client).await?;
        }
    };
    Ok(())
}

async fn decider_basic_tests(llm_client: &LlmClient) -> Result<()> {
    let decider = llm_client.decider().use_basic_backend().boolean();
    boolean::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_basic_backend().custom();
    custom::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_basic_backend().integer();
    integer::apply_test_questions(decider).await?;
    Ok(())
}
async fn decider_logit_bias_tests(llm_client: &LlmClient) -> Result<()> {
    let decider = llm_client.decider().use_logit_bias_backend().boolean();
    boolean::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_logit_bias_backend().custom();
    custom::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_logit_bias_backend().integer();
    integer::apply_test_questions(decider).await?;
    Ok(())
}
async fn decider_grammar_tests(llm_client: &LlmClient) -> Result<()> {
    let decider = llm_client.decider().use_grammar_backend().boolean();
    boolean::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_grammar_backend().custom();
    custom::apply_test_questions(decider).await?;
    let decider = llm_client.decider().use_grammar_backend().integer();
    integer::apply_test_questions(decider).await?;
    Ok(())
}
