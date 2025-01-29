use crate::requests::*;
use mistralrs::CompletionResponse as MistralCompletionResponse;

impl CompletionResponse {
    #[cfg(feature = "mistral_rs_backend")]
    pub fn new_from_mistral(
        req: &CompletionRequest,
        res: MistralCompletionResponse,
    ) -> Result<Self, CompletionError> {
        let choice = if res.choices.is_empty() || res.choices[0].text.is_empty() {
            return Err(CompletionError::ReponseContentEmpty);
        } else {
            &res.choices[0]
        };
        let finish_reason = match choice.finish_reason.as_str() {
            "stop" => CompletionFinishReason::Eos,
            "length" => CompletionFinishReason::StopLimit,
            _ => {
                return Err(CompletionError::StopReasonUnsupported(
                    "No stop reason provided".to_owned(),
                ))
            }
        };

        Ok(Self {
            id: "mistral_rs".to_owned(),
            index: None,
            content: choice.text.to_owned(),
            finish_reason,
            completion_probabilities: None,
            truncated: false,
            generation_settings: GenerationSettings::new_from_mistral(req, &res),
            timing_usage: TimingUsage::new_from_mistral(&res, req.start_time),
            token_usage: TokenUsage::new_from_mistral(&res),
        })
    }
}
