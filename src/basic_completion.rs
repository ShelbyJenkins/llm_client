use crate::{
    components::{
        base_request::{BaseLlmRequest, BaseRequestConfig, BaseRequestConfigTrait},
        constraints::logit_bias::{LogitBias, LogitBiasTrait},
        response::LlmClientResponse,
    },
    llm_backends::LlmBackend,
};
use anyhow::Result;
use llm_utils::prompting::LlmPrompt;
use std::rc::Rc;

#[derive(Clone)]
pub struct BasicCompletion {
    pub base_req: BaseLlmRequest,
}

impl BasicCompletion {
    pub fn new(backend: &Rc<LlmBackend>) -> Self {
        Self {
            base_req: BaseLlmRequest::new_from_backend(backend),
        }
    }

    pub fn prompt(&mut self) -> &mut LlmPrompt {
        &mut self.base_req.instruct_prompt.prompt
    }

    pub async fn run(&mut self) -> Result<LlmClientResponse> {
        self.base_req.instruct_prompt.prompt.build()?;

        let mut res = self.base_req.base_llm_client_request().await?;
        res.content = self.parse_response(&res.content)?;
        Ok(res)
    }

    fn parse_response(&self, content: &str) -> Result<String> {
        if content.is_empty() {
            return Err(anyhow::format_err!(
                "parse_response error: content.is_empty()"
            ));
        }
        let (bos, eos) = self.base_req.config.backend.get_bos_eos();

        let content = content
            .strip_prefix(&format!("{}\n\n", bos))
            .or_else(|| content.strip_prefix(&format!("{}\n", bos)))
            .or_else(|| content.strip_prefix(&bos))
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

impl BaseRequestConfigTrait for BasicCompletion {
    fn base_config(&mut self) -> &mut BaseRequestConfig {
        &mut self.base_req.config
    }

    fn clear_request(&mut self) {
        self.base_req.reset_base_request();
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
        .set_content("write a buzzfeed style listicle for the given input.")
        .append_content("boy howdy, how ya'll doing?");
    text_gen.max_tokens(500);
    let res = text_gen.run().await.unwrap();
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.is_empty());
}

pub async fn apply_logit_bias_tests(mut text_gen: BasicCompletion) -> Result<()> {
    text_gen.prompt().reset_prompt();
    text_gen
        .prompt()
        .add_user_message()
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
    #[cfg(feature = "mistralrs_backend")]
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
