use super::*;
use llm_utils::logit_bias;

pub async fn make_logit_bias_request(
    llm_client: &LlmClient,
    req_config: &RequestConfig,
) -> Result<String> {
    match &llm_client.backend {
        LlmBackend::Llama(backend) => {
            backend
                .decision_request(req_config, req_config.llama_logit_bias.as_ref(), None, None)
                .await
        }
        // LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(backend) => {
            backend
                .decision_request(req_config, req_config.openai_logit_bias.as_ref())
                .await
        }
        LlmBackend::Anthropic(_) => {
            panic!("Anthropic backend is not supported for LogitBias requests.")
        }
    }
}

pub async fn generate_parser_logit_bias(
    backend: &LlmBackend,
    choices: &[DeciderChoice],
) -> Result<HashMap<u32, f32>> {
    let mut logit_bias: HashMap<u32, f32> = HashMap::new();

    for choice in choices {
        let single_token_maybe = backend.try_into_single_token(&choice.parser_key).await;

        match single_token_maybe {
            Ok(token_id) => logit_bias.insert(token_id, 100.0),
            Err(_) => {
                panic!("Failed to convert parser_key to a single token. Sorry, but the logit_bias decided only supports singe token choices. Your choice, {} is larger than one token. If it's a local model use the grammar parser, or otherwise use the basic parser.", choice.parser_key);
            }
        };
    }
    logit_bias::validate_logit_bias_values(&logit_bias)?;
    backend.validate_logit_bias_token_ids(&logit_bias).await?;
    Ok(logit_bias)
}

pub async fn generate_parser_logit_bias_for_backend(
    backend: &LlmBackend,
    req_config: &mut RequestConfig,
) -> Result<()> {
    match backend {
        LlmBackend::Llama(_) => {
            let logit_bias = logit_bias::convert_logit_bias_to_llama_format(
                req_config.logit_bias.as_ref().unwrap(),
            );
            req_config.llama_logit_bias = Some(logit_bias);
        }
        // LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(_) => {
            let logit_bias = logit_bias::convert_logit_bias_to_openai_format(
                req_config.logit_bias.as_ref().unwrap(),
            )?;

            req_config.openai_logit_bias = Some(logit_bias);
        }
        LlmBackend::Anthropic(_) => {
            panic!("Anthropic backend is not supported for LogitBias requests.")
        }
    }
    Ok(())
}
