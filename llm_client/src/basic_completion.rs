use llm_interface::{
    llms::LlmBackend,
    requests::{
        completion::{CompletionRequest, CompletionResponse},
        constraints::logit_bias::{LogitBias, LogitBiasTrait},
        req_components::{RequestConfig, RequestConfigTrait},
    },
};
use llm_utils::prompting::LlmPrompt;

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
        let bos = self.base_req.backend.bos_token();
        let eos = self.base_req.backend.eos_token();
        let content = content
            .strip_prefix(&format!("{}\n\n", bos))
            .or_else(|| content.strip_prefix(&format!("{}\n", bos)))
            .or_else(|| content.strip_prefix(bos))
            .unwrap_or(content);

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

pub async fn apply_test(mut text_gen: BasicCompletion) {
    text_gen
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("write a buzzfeed style listicle for the given input.")
        .append_content("boy howdy, how ya'll doing?");
    text_gen.max_tokens(500);
    let res = text_gen.run().await.unwrap();
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.is_empty());
}

pub async fn apply_logit_bias_tests(mut text_gen: BasicCompletion) -> crate::Result<()> {
    text_gen.prompt().reset_prompt();
    text_gen
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input! Be excited.")
        .append_content("Boy howdy, how ya'll doing?");
    text_gen
        .max_tokens(100)
        .add_logit_bias_from_char('!', -100.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains(" ! "));

    text_gen.prompt().reset_prompt();
    text_gen.clear_logit_bias();
    text_gen
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    text_gen
        .max_tokens(100)
        .add_logit_bias_from_word("ya'll", -100.0);

    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains("ya'll"));

    text_gen.prompt().reset_prompt();
    text_gen.clear_logit_bias();
    text_gen
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    text_gen
        .max_tokens(100)
        .add_logit_bias_from_text("boy howdy", -100.0);

    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains("boy howdy"));

    text_gen.prompt().reset_prompt();
    text_gen.clear_logit_bias();
    text_gen
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    text_gen
        .max_tokens(100)
        .add_logit_bias_from_word("cowbell", 5.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res.content);
    // assert!(res.content.contains("cowbell"));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[tokio::test]
    #[serial]
    #[cfg(feature = "llama_cpp_backend")]
    pub async fn test_llama() {
        let llm = LlmClient::llama_cpp()
            .phi3_mini4k_instruct()
            .init()
            .await
            .unwrap();
        let text_gen = llm.basic_completion();
        apply_test(text_gen).await;
        // let text_gen = llm.basic_completion();
        // apply_logit_bias_tests(text_gen).await.unwrap();
    }

    #[tokio::test]
    #[serial]
    #[cfg(feature = "mistral_rs_backend")]
    pub async fn test_mistral() {
        let llm = LlmClient::mistral_rs()
            .phi3_mini4k_instruct()
            .init()
            .await
            .unwrap();
        let text_gen = llm.basic_completion();
        apply_test(text_gen).await;
    }

    #[tokio::test]
    #[serial]
    pub async fn test_openai() {
        let llm = LlmClient::openai().gpt_3_5_turbo().init().unwrap();

        let text_gen = llm.basic_completion();
        apply_test(text_gen).await;
        let text_gen = llm.basic_completion();
        apply_logit_bias_tests(text_gen).await.unwrap();
    }

    #[tokio::test]
    #[serial]
    pub async fn test_anthropic() {
        let llm = LlmClient::anthropic().claude_3_haiku().init().unwrap();

        let text_gen = llm.basic_completion();
        apply_test(text_gen).await;
    }
}
