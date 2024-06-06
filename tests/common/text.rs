use super::*;

pub async fn text_tests(llm_client: &LlmClient) -> Result<()> {
    match llm_client.backend {
        LlmBackend::Llama(_) => {
            println!("Llama basic text tests");
            let text_gen = llm_client.text().basic_text();
            unstructured_text::apply_test(text_gen).await?;
            println!("Llama logit bias text tests");
            let text_gen = llm_client.text().logit_bias_text();
            logit_bias_text::apply_test(text_gen).await?;
            println!("Llama grammar text tests");
            let text_gen = llm_client.text().grammar_text();
            grammar_text::apply_test(text_gen).await?;
            println!("Llama grammar list text tests");
            let text_gen = llm_client.text().grammar_list();
            grammar_text_list::apply_test(text_gen).await?;
        }
        #[cfg(feature = "mistralrs_backend")]
        LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(_) => {
            println!("OpenAI basic text tests");
            let text_gen = llm_client.text().basic_text();
            unstructured_text::apply_test(text_gen).await?;
            println!("OpenAI logit bias text tests");
            let text_gen = llm_client.text().logit_bias_text();
            logit_bias_text::apply_test(text_gen).await?;
        }
        LlmBackend::Anthropic(_) => {
            println!("Anthropic basic text tests");
            let text_gen = llm_client.text().basic_text();
            unstructured_text::apply_test(text_gen).await?;
        }
    };
    Ok(())
}
