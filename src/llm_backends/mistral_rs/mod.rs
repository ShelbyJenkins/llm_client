use super::LlmBackend;
use crate::{
    components::{base_request::BaseLlmRequest, response::LlmClientResponse},
    LlmClientResponseError,
};
use llm_utils::models::open_source_model::OsLlm;
use mistralrs::{
    Constraint,
    MistralRs,
    NormalRequest,
    Request,
    RequestMessage,
    Response,
    SamplingParams,
    StopTokens,
};

pub mod builder;
pub mod devices;

pub struct MistraRsBackend {
    client: std::sync::Arc<MistralRs>,
    pub model: OsLlm,
    _tracing_guard: Option<tracing::subscriber::DefaultGuard>,
    pub ctx_size: u32,
}

impl MistraRsBackend {
    pub async fn llm_request(
        &self,
        base_req: &BaseLlmRequest,
    ) -> crate::Result<LlmClientResponse, LlmClientResponseError> {
        let sampling_params = SamplingParams {
            temperature: Some(base_req.config.temperature.into()),
            top_p: Some(base_req.config.top_p.into()),
            frequency_penalty: Some(base_req.config.frequency_penalty),
            presence_penalty: Some(base_req.config.presence_penalty),
            max_len: base_req
                .config
                .actual_request_tokens
                .map(|val| val as usize),
            // logits_bias: req.logit_bias.clone(),
            // stop_toks: Some(StopTokens::Seqs(vec!["5. ".to_string()])),
            ..Default::default()
        };

        let constraint = Constraint::None;

        let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);

        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::CompletionTokens(
                base_req
                    .instruct_prompt
                    .prompt
                    .built_prompt_as_tokens
                    .clone()
                    .unwrap(),
            ),
            sampling_params,
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint,
            suffix: None,
            adapters: None,
            tool_choice: None,
            tools: None,
            logits_processors: None,
        });
        tracing::info!(?request);
        self.client
            .get_sender()
            .map_err(|e| LlmClientResponseError::RequestBuilderError {
                error: format!("MistraRsBackend builder error: {}", e),
            })?
            .send(request)
            .await
            .map_err(|e| LlmClientResponseError::RequestBuilderError {
                error: format!("MistraRsBackend builder error: {}", e),
            })?;

        match rx.recv().await {
            None => Err(LlmClientResponseError::UnknownStopReason {
                error: "MistraRsBackend llm_request error: rx.recv() is None".to_string(),
            }),
            Some(response) => match response {
                Response::InternalError(e) | Response::ValidationError(e) => {
                    Err(LlmClientResponseError::InferenceError {
                        error: format!("LlamaBackend request error: {}", e,),
                    })
                }
                Response::Chunk(_) | Response::Done(_) | Response::CompletionChunk(_) => {
                    Err(LlmClientResponseError::InferenceError {
                        error: "MistraRsBackend request error: Response::Chunk(_) | Response::Done(_) | Response::CompletionChunk(_)".to_string(),
                    })
                }
                Response::ModelError(e, _) | Response::CompletionModelError(e, _) => {
                    Err(LlmClientResponseError::InferenceError {
                        error: format!("MistraRsBackend request error: {}", e,),
                    })
                }
                Response::CompletionDone(completion) => {
                    tracing::info!(?completion);
                    LlmClientResponse::new_from_mistral(completion)
                }
            },
        }
    }
}
