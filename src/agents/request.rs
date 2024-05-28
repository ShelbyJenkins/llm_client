use crate::LlmBackend;
use anyhow::Result;
use llm_utils::prompting;
use std::collections::HashMap;

pub trait RequestConfigTrait {
    fn config_mut(&mut self) -> &mut RequestConfig;

    /// Set the system content to be used in the prompt. This is sometimes also called the "system prompt" or "instructions".
    fn system_content(&mut self, system_content: &str) -> &mut Self {
        self.config_mut().system_content = Some(system_content.to_string());
        self
    }

    /// Set the user content to be used in the prompt. This is sometimes also called the "user prompt" or "instructions". This will be appended to the system content.
    fn user_content(&mut self, user_content: &str) -> &mut Self {
        self.config_mut().user_content = Some(user_content.to_string());
        self
    }

    /// A path to a YAML file containing the system content to be used in the prompt. This appended after the `sytem_content`.
    /// This is useful if you have a base prompt in a file, and a dynamic system prompt and/or user prompt that you want to append to it.
    fn system_content_path(&mut self, system_content_path: &str) -> &mut Self {
        self.config_mut().system_content_path = Some(system_content_path.to_string());
        self
    }

    /// Number of tokens to use for the model's output. Not nessecarily what the model will use, but the maxium it's allowed to use.
    /// Before the request is built, the total input (prompt) tokens and the requested output (max_tokens) are used to ensure the request stays within the model's limits.
    fn max_tokens(&mut self, max_tokens: u32) -> &mut Self {
        self.config_mut().requested_response_tokens = Some(max_tokens);
        self
    }

    /// Number of retries to attempt after a failure.
    /// Default is 3.
    fn retry_after_fail_n_times(&mut self, retry_after_fail_n_times: u8) -> &mut Self {
        self.config_mut().retry_after_fail_n_times = retry_after_fail_n_times;
        self
    }

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        match frequency_penalty {
            value if (-2.0..=2.0).contains(&value) => self.config_mut().frequency_penalty = value,
            _ => self.config_mut().frequency_penalty = 0.0,
        };
        self
    }

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        match presence_penalty {
            value if (-2.0..=2.0).contains(&value) => self.config_mut().presence_penalty = value,
            _ => self.config_mut().presence_penalty = 0.0,
        };
        self
    }

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    fn temperature(&mut self, temperature: f32) -> &mut Self {
        match temperature {
            value if (0.0..=2.0).contains(&value) => self.config_mut().temperature = value,
            _ => self.config_mut().temperature = 1.0,
        };
        self
    }

    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    ///  We generally recommend altering this or `temperature` but not both.
    fn top_p(&mut self, top_p: f32) -> &mut Self {
        match top_p {
            value if (0.0..=2.0).contains(&value) => self.config_mut().top_p = value,
            _ => self.config_mut().top_p = 1.0,
        };
        self
    }
}

#[derive(Clone)]
pub struct RequestConfig {
    // Prompting
    pub system_content: Option<String>,
    pub system_content_path: Option<String>,
    pub user_content: Option<String>,
    pub default_formatted_prompt: Option<HashMap<String, HashMap<String, String>>>,
    pub chat_template_prompt: Option<String>,
    // Tokens
    // context_length is the limit of input + output
    pub context_length_for_model: u32,
    pub max_tokens_output_for_model: u32,
    //
    pub requested_response_tokens: Option<u32>,
    pub total_prompt_tokens: Option<u32>,
    pub actual_request_tokens: Option<u32>,
    pub safety_tokens: u32,
    // Config
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub retry_after_fail_n_times: u8,
    // Logit Bias
    pub logit_bias: Option<HashMap<u32, f32>>,
    pub llama_logit_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub openai_logit_bias: Option<HashMap<String, serde_json::Value>>,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            system_content: None,
            system_content_path: None,
            user_content: None,
            default_formatted_prompt: None,
            chat_template_prompt: None,
            context_length_for_model: 1024,
            max_tokens_output_for_model: 1024,
            requested_response_tokens: None,
            total_prompt_tokens: None,
            actual_request_tokens: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            temperature: 1.0,
            top_p: 1.0,
            safety_tokens: 10,
            retry_after_fail_n_times: 3,
            logit_bias: None,
            llama_logit_bias: None,
            openai_logit_bias: None,
        }
    }
}

impl RequestConfig {
    pub fn new(backend: &LlmBackend) -> Self {
        let (context_length_for_model, max_tokens_output_for_model) = match backend {
            LlmBackend::Llama(backend) => (backend.ctx_size, backend.ctx_size),
            // LlmBackend::MistralRs(backend) => (backend.ctx_size, backend.ctx_size),
            LlmBackend::OpenAi(backend) => (
                backend.model.context_length,
                backend.model.max_tokens_output,
            ),
            LlmBackend::Anthropic(backend) => (
                backend.model.context_length,
                backend.model.max_tokens_output,
            ),
        };

        Self {
            context_length_for_model,
            max_tokens_output_for_model,
            ..Default::default()
        }
    }

    pub async fn build_request(&mut self, backend: &LlmBackend) -> Result<()> {
        self.default_formatted_prompt = Some(prompting::default_formatted_prompt(
            &self.system_content,
            &self.system_content_path,
            &self.user_content,
        )?);

        match backend {
            LlmBackend::Llama(backend) => {
                let formatted_prompt = prompting::convert_default_prompt_to_model_format(
                    self.default_formatted_prompt.as_ref().unwrap(),
                    &backend.model.as_ref().unwrap().metadata.chat_template,
                )
                .unwrap();
                self.chat_template_prompt = Some(formatted_prompt);
            }
            // LlmBackend::MistralRs(backend) => {
            //     let formatted_prompt = prompting::convert_default_prompt_to_model_format(
            //         self.default_formatted_prompt.as_ref().unwrap(),
            //         &backend.model.as_ref().unwrap().metadata.chat_template,
            //     )
            //     .unwrap();
            //     self.chat_template_prompt = Some(formatted_prompt);
            // }
            LlmBackend::OpenAi(_) => (),
            LlmBackend::Anthropic(_) => (),
        }

        let total_prompt_tokens = match &backend {
            LlmBackend::Llama(backend) => {
                backend
                    .count_tokens(self.chat_template_prompt.as_ref().unwrap())
                    .await? as u32
            }
            // LlmBackend::MistralRs(_backend) => {
            //     todo!()
            // }
            LlmBackend::OpenAi(backend) => backend.model.openai_token_count_of_prompt(
                backend.tokenizer.as_ref().unwrap(),
                self.default_formatted_prompt.as_ref().unwrap(),
            ),
            LlmBackend::Anthropic(backend) => backend.model.anthropic_token_count_of_prompt(
                &backend.tokenizer,
                self.default_formatted_prompt.as_ref().unwrap(),
            ),
        };

        self.total_prompt_tokens = Some(total_prompt_tokens);

        Ok(())
    }

    pub fn set_max_tokens_for_request(
        &mut self,
        model_token_utilization: Option<f32>,
    ) -> Result<()> {
        let requested_response_tokens = if let Some(requested_response_tokens) =
            self.requested_response_tokens
        {
            if requested_response_tokens > self.max_tokens_output_for_model {
                eprintln!(
                    "requested_response_tokens is greater than max_tokens_output_for_model. using max_tokens_output_for_model for request. requested_response_tokens: {}, than max_tokens_output_for_model: {}",
                    requested_response_tokens,
                    self.max_tokens_output_for_model
                );
                self.max_tokens_output_for_model
            } else {
                requested_response_tokens
            }
        } else {
            self.max_tokens_output_for_model
        };

        let actual_request_tokens = prompting::get_and_check_max_tokens_for_response(
            self.context_length_for_model,
            self.max_tokens_output_for_model,
            self.total_prompt_tokens.unwrap(),
            self.safety_tokens,
            model_token_utilization,
            Some(requested_response_tokens),
        )?;
        self.actual_request_tokens = Some(actual_request_tokens);
        Ok(())
    }

    pub fn set_max_tokens_for_request_for_decision(
        &mut self,
        requested_response_tokens: u32,
    ) -> Result<()> {
        let actual_request_tokens = prompting::get_and_check_max_tokens_for_response(
            self.context_length_for_model,
            self.max_tokens_output_for_model,
            self.total_prompt_tokens.unwrap(),
            0,
            None,
            Some(requested_response_tokens),
        )?;
        self.actual_request_tokens = Some(actual_request_tokens);
        Ok(())
    }
}
