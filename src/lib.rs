use std::collections::HashMap;
use std::error::Error;
pub mod agents;
pub mod prelude;
pub mod prompting;
pub mod providers;
pub mod text_utils;

use crate::providers::llama_cpp::{models::LlamaLlmModels, LlamaLlm};
use crate::providers::llm_openai::{models::OpenAiLlmModels, OpenAiLlm};

pub struct LlmClient {
    pub llm_definition: LlmDefinition,
    pub model_params: LlmModelParams,
    pub retry_after_fail_n_times: u8,
}
impl LlmClient {
    pub fn new(llm_definition: &LlmDefinition, retry_after_fail_n_times: Option<u8>) -> Self {
        let retry_after_fail_n_times = retry_after_fail_n_times.unwrap_or(1);
        Self {
            model_params: get_default_model_params(llm_definition),
            llm_definition: llm_definition.clone(),
            retry_after_fail_n_times,
        }
    }
    pub fn frequency_penalty(&mut self, frequency_penalty: Option<f32>) {
        self.model_params.frequency_penalty = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiLlmModels::frequency_penalty(frequency_penalty),
            LlmDefinition::LlamaLlm(_) => LlamaLlmModels::frequency_penalty(frequency_penalty),
        }
    }
    pub fn presence_penalty(&mut self, presence_penalty: Option<f32>) {
        self.model_params.presence_penalty = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiLlmModels::presence_penalty(presence_penalty),
            LlmDefinition::LlamaLlm(_) => LlamaLlmModels::presence_penalty(presence_penalty),
        }
    }
    pub fn temperature(&mut self, temperature: Option<f32>) {
        self.model_params.temperature = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiLlmModels::temperature(temperature),
            LlmDefinition::LlamaLlm(_) => LlamaLlmModels::temperature(temperature),
        }
    }
    pub fn top_p(&mut self, top_p: Option<f32>) {
        self.model_params.top_p = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiLlmModels::top_p(top_p),
            LlmDefinition::LlamaLlm(_) => LlamaLlmModels::top_p(top_p),
        }
    }
    pub fn max_tokens_for_model(&mut self, max_tokens_for_model: Option<u16>) {
        self.model_params.max_tokens_for_model = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(model_definition) => {
                OpenAiLlmModels::max_tokens_for_model(model_definition, max_tokens_for_model)
            }
            LlmDefinition::LlamaLlm(model_definition) => {
                LlamaLlmModels::max_tokens_for_model(model_definition, max_tokens_for_model)
            }
        }
    }

    pub async fn make_boolean_decision(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        logit_bias: &HashMap<String, serde_json::Value>,
        batch_count: u8,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let prompt =
            crate::prompting::create_model_formatted_prompt(&self.llm_definition, prompt.clone());

        // let total_prompt_tokens = crate::prompting::check_available_request_tokens_decision(
        //     &self.llm_definition,
        //     &prompt,
        //     1,
        //     &self.model_params,
        // );
        match self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => {
                let provider = OpenAiLlm::default();
                let (responses, total_response_tokens) = provider
                    .make_boolean_decision(&prompt, logit_bias, batch_count, &self.model_params)
                    .await?;
                Ok(responses)
            }
            LlmDefinition::LlamaLlm(_) => {
                let provider = LlamaLlm::default();
                let responses = provider
                    .make_boolean_decision(&prompt, logit_bias, batch_count, &self.model_params)
                    .await?;
                Ok(responses)
            }
        }
    }

    pub async fn generate_text(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        logit_bias: &Option<HashMap<String, serde_json::Value>>,
        model_token_utilization: Option<f32>,
        context_to_response_ratio: Option<f32>,
        max_response_tokens: Option<u16>,
    ) -> Result<String, Box<dyn Error>> {
        let context_to_response_ratio = context_to_response_ratio.unwrap_or(0.0);
        let model_token_utilization = model_token_utilization.unwrap_or(0.5);

        let prompt =
            crate::prompting::create_model_formatted_prompt(&self.llm_definition, prompt.clone());

        let (total_prompt_tokens, new_max_response_tokens) =
            crate::prompting::check_available_request_tokens_generation(
                &self.llm_definition,
                &prompt,
                context_to_response_ratio,
                model_token_utilization,
                &self.model_params,
            )
            .await;
        let max_response_tokens = if let Some(mut tokens) = max_response_tokens {
            // If using a custom max_response_tokens from function param
            while tokens
                > (self.model_params.max_tokens_for_model - self.model_params.safety_tokens)
            {
                tokens -= 1;
            }
            tokens
        } else {
            // This value already checked in check_available_request_tokens_generation
            new_max_response_tokens
        };

        match self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => {
                let provider = OpenAiLlm::default();
                let (responses, total_response_tokens) = provider
                    .generate_text(&prompt, max_response_tokens, logit_bias, &self.model_params)
                    .await?;
                Ok(responses)
            }
            LlmDefinition::LlamaLlm(_) => {
                let provider = LlamaLlm::default();
                let (responses, total_response_tokens) = provider
                    .generate_text(&prompt, max_response_tokens, logit_bias, &self.model_params)
                    .await?;
                Ok(responses)
            }
        }
    }
}

fn get_default_model_params(llm_definition: &LlmDefinition) -> LlmModelParams {
    match llm_definition {
        LlmDefinition::OpenAiLlm(model_definition) => {
            OpenAiLlmModels::get_default_model_params(model_definition)
        }
        LlmDefinition::LlamaLlm(model_definition) => {
            LlamaLlmModels::get_default_model_params(model_definition)
        }
    }
}

#[derive(Debug, Clone)]
pub enum LlmDefinition {
    LlamaLlm(LlamaLlmModels),
    OpenAiLlm(OpenAiLlmModels),
}

pub struct LlmModelParams {
    pub model_id: String,
    pub model_filename: Option<String>,
    pub max_tokens_for_model: u16,
    pub cost_per_k: f32,
    pub tokens_per_message: u16,
    pub tokens_per_name: i16,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub safety_tokens: u16,
}
