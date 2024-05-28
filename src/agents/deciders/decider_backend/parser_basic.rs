use super::*;

pub async fn make_basic_parser_request(
    llm_client: &LlmClient,
    req_config: &RequestConfig,
) -> Result<String> {
    match &llm_client.backend {
        LlmBackend::Llama(backend) => backend.decision_request(req_config, None, None, None).await,
        // LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(backend) => backend.decision_request(req_config, None).await,
        LlmBackend::Anthropic(backend) => backend.text_generation_request(req_config).await,
    }
}
