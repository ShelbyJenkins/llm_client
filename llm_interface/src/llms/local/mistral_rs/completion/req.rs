use crate::requests::completion::*;
use mistralrs::{
    Constraint,
    NormalRequest,
    Request as MistralCompletionRequest,
    RequestMessage,
    Response,
    SamplingParams,
};

pub fn new(
    request: &CompletionRequest,
    tx: tokio::sync::mpsc::Sender<Response>,
) -> crate::Result<MistralCompletionRequest, CompletionError> {
    let mut sampling_params = SamplingParams::deterministic();

    sampling_params.temperature = Some(request.config.temperature.into());
    sampling_params.frequency_penalty = request.config.frequency_penalty;
    sampling_params.presence_penalty = Some(request.config.presence_penalty);
    sampling_params.max_len = request.config.actual_request_tokens.map(|val| val as usize);
    // logits_bias = req.logit_bias.clone();
    // stop_toks = Some(StopTokens::Seqs(vec!["5. ".to_string()]));

    let constraint = Constraint::None;

    let request = MistralCompletionRequest::Normal(NormalRequest {
        messages: RequestMessage::CompletionTokens(
            request
                .prompt
                .get_built_prompt_as_tokens()
                .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?,
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
    Ok(request)
}
