use llm_interface::{llms::LlmBackend, requests::*};
use llm_prompt::LlmPrompt;

#[derive(Clone)]
pub struct BasicCompletion {
    pub base_req: CompletionRequest,
}

impl BasicCompletion {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    pub fn prompt(&mut self) -> &mut LlmPrompt {
        &mut self.base_req.prompt
    }

    pub async fn run(&mut self) -> crate::Result<CompletionResponse> {
        let mut res = self.base_req.request().await?;

        match *self.base_req.backend {
            #[cfg(feature = "llama_cpp_backend")]
            LlmBackend::LlamaCpp(_) => {
                res.content = self.parse_response(&res.content)?;
            }
            #[cfg(feature = "mistral_rs_backend")]
            LlmBackend::MistralRs(_) => {
                res.content = self.parse_response(&res.content)?;
            }
            _ => (),
        }
        Ok(res)
    }

    fn parse_response(&self, content: &str) -> crate::Result<String> {
        if content.is_empty() {
            return Err(anyhow::format_err!(
                "parse_response error: content.is_empty()"
            ));
        }

        let eos = self.base_req.backend.eos_token();
        let content = if let Some(bos) = self.base_req.backend.bos_token() {
            content
                .strip_prefix(&format!("{}\n\n", bos))
                .or_else(|| content.strip_prefix(&format!("{}\n", bos)))
                .or_else(|| content.strip_prefix(bos))
                .unwrap_or(content)
        } else {
            content
        };

        let content = content
            .strip_prefix("assistant\n\n")
            .or_else(|| content.strip_prefix("assistant\n"))
            .or_else(|| content.strip_prefix("assistant"))
            .unwrap_or(content);

        let content = content
            .strip_suffix(&format!("{}\n\n", eos))
            .or_else(|| content.strip_suffix(&format!("{}\n", eos)))
            .or_else(|| content.strip_suffix(&eos))
            .unwrap_or(content)
            .to_string();

        Ok(content.trim().to_owned())
    }
}

impl RequestConfigTrait for BasicCompletion {
    fn config(&mut self) -> &mut RequestConfig {
        &mut self.base_req.config
    }

    fn reset_request(&mut self) {
        self.base_req.reset_completion_request();
    }
}

impl LogitBiasTrait for BasicCompletion {
    fn lb_mut(&mut self) -> &mut Option<LogitBias> {
        &mut self.base_req.logit_bias
    }
}
