use super::*;
use llm_utils::logit_bias;

impl<'a> RequestConfigTrait for LogitBiasText<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

#[derive(Clone)]
pub struct LogitBiasText<'a> {
    pub logit_bias_from_token_ids: Option<HashMap<u32, f32>>,
    pub logit_bias_from_chars: Option<HashMap<char, f32>>,
    pub logit_bias_from_words: Option<HashMap<String, f32>>,
    pub logit_bias_from_texts: Option<HashMap<String, f32>>,
    pub model_token_utilization: Option<f32>,
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
}

impl<'a> LogitBiasText<'a> {
    pub fn new(llm_client: &'a LlmClient, default_config: RequestConfig) -> Self {
        Self {
            logit_bias_from_token_ids: None,
            logit_bias_from_chars: None,
            logit_bias_from_texts: None,
            logit_bias_from_words: None,
            model_token_utilization: None,
            req_config: default_config,
            llm_client,
        }
    }

    /// Adds a logit bias for a specific token ID. In the case you have your own tokenizer or other situations where you have token IDs.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID.
    /// * `bias` - The bias value.
    pub fn add_logit_bias_token_id(&mut self, token_id: u32, bias: f32) -> &mut Self {
        if let Some(logit_bias) = &mut self.logit_bias_from_token_ids {
            logit_bias.entry(token_id).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(token_id).or_insert(bias);
            self.logit_bias_from_token_ids = Some(logit_bias);
        }
        self.clear_req_logit_bias();
        self
    }

    /// Adds multiple logit biases for token IDs. In the case you have your own tokenizer or other situations where you have token IDs.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A `HashMap` containing token IDs as keys and bias values as values.
    pub fn add_logit_bias_token_ids(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias_from_token_ids {
            for (token_id, bias) in logit_bias {
                existing_logit_bias.insert(token_id, bias);
            }
        } else {
            self.logit_bias_from_token_ids = Some(logit_bias);
        }
        self.clear_req_logit_bias();
        self
    }

    /// Adds a logit bias for a specific character.
    /// Not very useful as it does not necessarily remove all instances of that character as the character may be part of other tokens.
    ///
    /// # Arguments
    ///
    /// * `char` - The character.
    /// * `bias` - The bias value.
    pub fn add_logit_bias_from_char(&mut self, char: char, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias_from_chars {
            existing_logit_bias.entry(char.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(char.to_owned()).or_insert(bias);
            self.logit_bias_from_chars = Some(logit_bias);
        }
        self.clear_req_logit_bias();
        self
    }

    /// Adds a logit bias for a specific word. If a word is more than one token, it will be split into multiple tokens.
    /// Errors if the word is empty or contains whitespace.
    ///
    /// # Arguments
    ///
    /// * `word` - The word.
    /// * `bias` - The bias value.
    pub fn add_logit_bias_from_word(&mut self, word: &str, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias_from_words {
            existing_logit_bias.entry(word.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(word.to_owned()).or_insert(bias);
            self.logit_bias_from_words = Some(logit_bias);
        }
        self.clear_req_logit_bias();
        self
    }

    /// Adds a logit bias for a specific text. Splits the text into tokens and applies the bias to each token. It does not add the logit bias value to the whitespace token.
    ///
    /// # Arguments
    ///
    /// * `text` - The text.
    /// * `bias` - The bias value.
    pub fn add_logit_bias_from_text(&mut self, text: &str, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias_from_texts {
            existing_logit_bias.entry(text.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(text.to_owned()).or_insert(bias);
            self.logit_bias_from_texts = Some(logit_bias);
        }
        self.clear_req_logit_bias();
        self
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

    /// Clears the logit bias configuration. To reuse the request object for another request. Mostly for testing.
    pub fn clear_logit_bias(&mut self) -> &mut Self {
        self.logit_bias_from_token_ids = None;
        self.logit_bias_from_chars = None;
        self.logit_bias_from_words = None;
        self.logit_bias_from_texts = None;
        self.clear_req_logit_bias();
        self
    }

    fn clear_req_logit_bias(&mut self) {
        self.req_config.llama_logit_bias = None;
        self.req_config.openai_logit_bias = None;
        self.req_config.logit_bias = None;
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

        if self.req_config.logit_bias.is_none() {
            self.req_config.logit_bias = Some(self.build_logit_bias().await?);
        }

        match &self.llm_client.backend {
            LlmBackend::Llama(backend) => {
                if self.req_config.llama_logit_bias.is_none() {
                    self.req_config.llama_logit_bias =
                        Some(logit_bias::convert_logit_bias_to_llama_format(
                            self.req_config.logit_bias.as_ref().unwrap(),
                        ));
                }

                let res = backend
                    .text_generation_request(
                        &self.req_config,
                        self.req_config.llama_logit_bias.as_ref(),
                        None,
                    )
                    .await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                Ok(res.content)
            }
            // LlmBackend::MistralRs(_) => {
            //     panic!("Mistral backend is not supported for logit bias based calls.")
            // }
            LlmBackend::OpenAi(backend) => {
                if self.req_config.openai_logit_bias.is_none() {
                    self.req_config.openai_logit_bias =
                        Some(logit_bias::convert_logit_bias_to_openai_format(
                            self.req_config.logit_bias.as_ref().unwrap(),
                        )?);
                }

                let res = backend
                    .text_generation_request(
                        &self.req_config,
                        self.req_config.openai_logit_bias.as_ref(),
                    )
                    .await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                Ok(res.choices[0].message.content.clone().unwrap())
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for logit bias based calls.")
            }
        }
    }
    async fn build_logit_bias(&self) -> Result<HashMap<u32, f32>> {
        let validated_logit_bias =
            if let Some(logit_bias_from_token_ids) = &self.logit_bias_from_token_ids {
                self.llm_client
                    .backend
                    .validate_logit_bias_token_ids(logit_bias_from_token_ids)
                    .await?;
                logit_bias_from_token_ids.clone()
            } else {
                HashMap::new()
            };

        let validated_logit_bias = if let Some(logit_bias_from_chars) = &self.logit_bias_from_chars
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &self
                    .llm_client
                    .backend
                    .logit_bias_from_chars(logit_bias_from_chars)
                    .await?,
            ])
        } else {
            validated_logit_bias
        };
        let validated_logit_bias = if let Some(logit_bias_from_words) = &self.logit_bias_from_words
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &self
                    .llm_client
                    .backend
                    .logit_bias_from_words(logit_bias_from_words)
                    .await?,
            ])
        } else {
            validated_logit_bias
        };

        let validated_logit_bias = if let Some(logit_bias_from_texts) = &self.logit_bias_from_texts
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &self
                    .llm_client
                    .backend
                    .logit_bias_from_texts(logit_bias_from_texts)
                    .await?,
            ])
        } else {
            validated_logit_bias
        };
        logit_bias::validate_logit_bias_values(&validated_logit_bias)?;
        Ok(validated_logit_bias)
    }
}

pub async fn apply_test(mut text_gen: LogitBiasText<'_>) -> Result<()> {
    text_gen
        .system_content("Write a buzzfeed style listicle for the given input! Be excited.")
        .user_content("Boy howdy, how ya'll doing?")
        .max_tokens(100)
        .add_logit_bias_from_char('!', -100.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res);
    assert!(!res.contains(" ! "));

    text_gen.clear_logit_bias();
    text_gen
        .system_content("Write a buzzfeed style listicle for the given input!")
        .user_content("Boy howdy, how ya'll doing?")
        .max_tokens(100)
        .add_logit_bias_from_word("ya'll", -100.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res);
    assert!(!res.contains("ya'll"));

    text_gen.clear_logit_bias();
    text_gen
        .system_content("Write a buzzfeed style listicle for the given input!")
        .user_content("Boy howdy, how ya'll doing?")
        .max_tokens(100)
        .add_logit_bias_from_text("boy howdy", -100.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res);
    assert!(!res.contains("boy howdy"));

    text_gen.clear_logit_bias();
    text_gen
        .system_content("Write a buzzfeed style listicle for the given input!")
        .user_content("Boy howdy, how ya'll doing?")
        .max_tokens(100)
        .add_logit_bias_from_text("more cowbell", 10.0);
    let res = text_gen.run().await?;
    println!("Response:\n {}\n", res);
    // assert!(res.contains("cowbell"));

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
        let text_gen = llm.text().logit_bias_text();
        apply_test(text_gen).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_openai() -> Result<()> {
        let llm = LlmClient::openai_backend().gpt_3_5_turbo().init()?;

        let text_gen = llm.text().logit_bias_text();
        apply_test(text_gen).await?;
        Ok(())
    }
}
