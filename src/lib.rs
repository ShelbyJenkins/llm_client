use std::collections::HashMap;
use std::error::Error;
pub mod agents;
pub mod prelude;
pub mod prompting;
pub mod providers;
pub mod text_utils;

use crate::providers::llama_cpp::{models::LlamaDef, LlamaClient};
use crate::providers::llm_openai::{models::OpenAiDef, OpenAiClient};
#[macro_use]
extern crate lazy_static;

pub struct ProviderClient {
    pub llm_definition: LlmDefinition,
    pub llm_client: LlmClient,
    pub model_params: LlmModelParams,
    pub retry_after_fail_n_times: u8,
}
fn get_default_model_params(llm_definition: &LlmDefinition) -> LlmModelParams {
    match llm_definition {
        LlmDefinition::OpenAiLlm(model_definition) => {
            OpenAiDef::get_default_model_params(model_definition)
        }
        LlmDefinition::LlamaLlm(model_definition) => {
            LlamaDef::get_default_model_params(model_definition)
        }
    }
}

#[derive(Debug, Clone)]
pub enum LlmDefinition {
    LlamaLlm(LlamaDef),
    OpenAiLlm(OpenAiDef),
}

pub enum LlmClient {
    LlamaLlm(LlamaClient),
    OpenAiLlm(OpenAiClient),
}

pub struct LlmModelParams {
    pub model_id: String,
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

impl ProviderClient {
    pub async fn new(llm_definition: &LlmDefinition, retry_after_fail_n_times: Option<u8>) -> Self {
        let retry_after_fail_n_times = retry_after_fail_n_times.unwrap_or(1);
        let llm_client = match llm_definition {
            LlmDefinition::OpenAiLlm(_) => LlmClient::OpenAiLlm(OpenAiClient::default()),
            LlmDefinition::LlamaLlm(def) => LlmClient::LlamaLlm(LlamaClient::new(def).await),
        };
        Self {
            model_params: get_default_model_params(llm_definition),
            llm_definition: llm_definition.clone(),
            retry_after_fail_n_times,
            llm_client,
        }
    }

    pub fn frequency_penalty(&mut self, frequency_penalty: Option<f32>) {
        self.model_params.frequency_penalty = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiDef::frequency_penalty(frequency_penalty),
            LlmDefinition::LlamaLlm(_) => LlamaDef::frequency_penalty(frequency_penalty),
        }
    }
    pub fn presence_penalty(&mut self, presence_penalty: Option<f32>) {
        self.model_params.presence_penalty = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiDef::presence_penalty(presence_penalty),
            LlmDefinition::LlamaLlm(_) => LlamaDef::presence_penalty(presence_penalty),
        }
    }
    pub fn temperature(&mut self, temperature: Option<f32>) {
        self.model_params.temperature = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiDef::temperature(temperature),
            LlmDefinition::LlamaLlm(_) => LlamaDef::temperature(temperature),
        }
    }
    pub fn top_p(&mut self, top_p: Option<f32>) {
        self.model_params.top_p = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(_) => OpenAiDef::top_p(top_p),
            LlmDefinition::LlamaLlm(_) => LlamaDef::top_p(top_p),
        }
    }
    pub fn max_tokens_for_model(&mut self, max_tokens_for_model: Option<u16>) {
        self.model_params.max_tokens_for_model = match &self.llm_definition {
            LlmDefinition::OpenAiLlm(model_definition) => {
                OpenAiDef::max_tokens_for_model(model_definition, max_tokens_for_model)
            }
            LlmDefinition::LlamaLlm(model_definition) => {
                LlamaDef::max_tokens_for_model(model_definition, max_tokens_for_model)
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

        // let _ = crate::prompting::check_available_request_tokens_decision(
        //     &self.llm_definition,
        //     &prompt,
        //     1,
        //     &self.model_params,
        // );
        match &self.llm_client {
            LlmClient::OpenAiLlm(client) => {
                let (responses, _) = client
                    .make_boolean_decision(&prompt, logit_bias, batch_count, &self.model_params)
                    .await?;
                Ok(responses)
            }
            LlmClient::LlamaLlm(client) => {
                let responses = client
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

        let (_, new_max_response_tokens) = Self::check_available_request_tokens_generation(
            self,
            &prompt,
            context_to_response_ratio,
            model_token_utilization,
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

        match &self.llm_client {
            LlmClient::OpenAiLlm(client) => {
                let (responses, _) = client
                    .generate_text(&prompt, max_response_tokens, logit_bias, &self.model_params)
                    .await?;
                Ok(responses)
            }
            LlmClient::LlamaLlm(client) => {
                let (responses, _) = client
                    .generate_text(&prompt, max_response_tokens, logit_bias, &self.model_params)
                    .await?;
                Ok(responses)
            }
        }
    }

    // Checking to ensure we don't exceed the max tokens for the model for text generation
    // Important for OpenAI as API requests will fail if
    // input tokens + requested output tokens (max_tokens) exceeds
    // the max tokens for the model
    pub async fn check_available_request_tokens_generation(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        _context_to_response_ratio: f32,
        model_token_utilization: f32,
    ) -> (u16, u16) {
        let total_prompt_tokens = Self::get_prompt_length(self, prompt).await;
        // Calculate available tokens for response
        let available_tokens = self.model_params.max_tokens_for_model - total_prompt_tokens;
        let mut max_response_tokens =
            (available_tokens as f32 * (model_token_utilization)).ceil() as u16;
        // if context_to_response_ratio > 0.0 {
        //     let mut max_response_tokens =
        //         (available_tokens as f32 * context_to_response_ratio).ceil() as u16;

        // }

        //  for safety in case of model changes
        while max_response_tokens
            > (self.model_params.max_tokens_for_model - self.model_params.safety_tokens)
        {
            max_response_tokens -= 1
        }
        (total_prompt_tokens, max_response_tokens)
    }

    // Checking to ensure we don't exceed the max tokens for the model for decision making
    pub async fn check_available_request_tokens_decision(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        logit_bias_response_tokens: u16,
    ) -> u16 {
        let total_prompt_tokens = Self::get_prompt_length(self, prompt).await;
        let max_response_tokens = total_prompt_tokens + logit_bias_response_tokens;
        //  for safety in case of model changes
        if max_response_tokens
            > self.model_params.max_tokens_for_model - self.model_params.safety_tokens
        {
            panic!(
                "max_response_tokens {} is greater than available_tokens {}",
                max_response_tokens,
                self.model_params.max_tokens_for_model - self.model_params.safety_tokens
            );
        }
        total_prompt_tokens
    }

    async fn get_prompt_length(&self, prompt: &HashMap<String, HashMap<String, String>>) -> u16 {
        match &self.llm_client {
            LlmClient::OpenAiLlm(_) => {
                crate::prompting::token_count_of_openai_prompt(prompt, &self.model_params)
            }
            LlmClient::LlamaLlm(client) => client
                .llama_cpp_count_tokens(&prompt["llama_prompt"]["content"])
                .await
                .unwrap(),
        }
    }
}
