use super::*;

impl<'a> RequestConfigTrait for UnstructuredText<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

#[derive(Clone)]
pub struct UnstructuredText<'a> {
    pub model_token_utilization: Option<f32>,
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
}

impl<'a> UnstructuredText<'a> {
    pub fn new(llm_client: &'a LlmClient, default_config: RequestConfig) -> Self {
        Self {
            model_token_utilization: None,
            req_config: default_config,
            llm_client,
        }
    }

    /// Sets the model token utilization. This is sets the 'max_tokens' parameter for the request to a percent of the available 'ctx_size' for the model or server settings.
    /// The value should be between 0.0 and 1.0.
    ///
    /// # Arguments
    ///
    /// * `model_token_utilization` - The model token utilization value.
    pub fn model_token_utilization(&mut self, model_token_utilization: f32) -> &mut Self {
        self.model_token_utilization = Some(model_token_utilization);
        self
    }

    /// Runs the text generation request and returns the generated text.
    ///
    /// # Returns
    ///
    /// A `Result` containing the generated text or an error.
    pub async fn run(&mut self) -> Result<String> {
        self.req_config
            .build_request(&self.llm_client.backend)
            .await?;

        self.req_config
            .set_max_tokens_for_request(self.model_token_utilization)?;

        match &self.llm_client.backend {
            LlmBackend::Llama(backend) => {
                let res = backend
                    .text_generation_request(&self.req_config, None, None)
                    .await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                Ok(res.content)
            }
            // LlmBackend::MistralRs(backend) => {
            //     let res = backend.text_generation_request(&self.req_config).await?;
            //     if backend.logging_enabled {
            //         tracing::info!(?res);
            //     }
            //     Ok(res)
            // }
            LlmBackend::OpenAi(backend) => {
                let res = backend
                    .text_generation_request(&self.req_config, None)
                    .await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                Ok(res.choices[0].message.content.clone().unwrap())
            }
            LlmBackend::Anthropic(backend) => {
                let res = backend.text_generation_request(&self.req_config).await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                Ok(res)
            }
        }
    }
}

pub async fn apply_test(mut text_gen: UnstructuredText<'_>) -> Result<()> {
    text_gen
        .system_content("write a buzzfeed style listicle for the given input")
        .user_content("boy howdy, how ya'll doing?")
        .max_tokens(100);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res);
    assert!(!res.is_empty());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[tokio::test]
    #[serial]
    pub async fn test_llama() -> Result<()> {
        let llm = LlmClient::llama_backend().init().await?;
        let text_gen = llm.text().basic_text();
        apply_test(text_gen).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_openai() -> Result<()> {
        let llm = LlmClient::openai_backend().gpt_3_5_turbo().init()?;

        let text_gen = llm.text().basic_text();
        apply_test(text_gen).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_anthropic() -> Result<()> {
        let llm = LlmClient::anthropic_backend().claude_3_haiku().init()?;

        let text_gen = llm.text().basic_text();
        apply_test(text_gen).await?;
        Ok(())
    }
}
