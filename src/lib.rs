use std::{collections::HashMap, error::Error, io, vec};
pub mod agents;
pub mod prelude;
pub mod prompting;
pub mod providers;
pub mod text_utils;

use crate::providers::{
    llama_cpp::{models::LlamaDef, LlamaClient},
    llm_openai::{models::OpenAiDef, OpenAiClient},
};
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

#[derive(PartialEq)]
pub enum EmbeddingExceedsMaxTokensBehavior {
    Panic,
    Skip,
    // Truncate,
    // Reduce,
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

        let _ = Self::check_available_request_tokens_decision(self, &prompt, 1).await;

        match &self.llm_client {
            LlmClient::OpenAiLlm(client) => {
                let responses = client
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

    // Checking to ensure we don't exceed the max tokens for the model for decision making
    async fn check_available_request_tokens_decision(
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
                let responses = client
                    .generate_text(&prompt, max_response_tokens, logit_bias, &self.model_params)
                    .await?;
                Ok(responses)
            }
            LlmClient::LlamaLlm(client) => {
                let responses = client
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
    async fn check_available_request_tokens_generation(
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

    pub async fn generate_embeddings(
        &self,
        inputs: &Vec<String>,
        exceeds_max_tokens_behavior: Option<EmbeddingExceedsMaxTokensBehavior>,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let exceeds_max_tokens_behavior =
            exceeds_max_tokens_behavior.unwrap_or(EmbeddingExceedsMaxTokensBehavior::Panic);

        let mut outputs: Vec<Vec<f32>> = vec![];
        for input in inputs {
            match Self::check_embedding_input_tokens(self, input).await {
                Ok(_) => match &self.llm_client {
                    LlmClient::OpenAiLlm(client) => {
                        let response = client.generate_embedding(input, &self.model_params).await?;
                        outputs.push(response);
                    }
                    LlmClient::LlamaLlm(client) => {
                        let response = client.generate_embedding(input).await?;
                        outputs.push(response);
                    }
                },
                Err(error) => match exceeds_max_tokens_behavior {
                    EmbeddingExceedsMaxTokensBehavior::Panic => {
                        println!("Panicking  due to EmbeddingExceedsMaxTokensBehavior::Panic");
                        panic!("Error in generate_embedding: {}", error)
                    }
                    EmbeddingExceedsMaxTokensBehavior::Skip => {
                        println!("Error in generate_embedding: {}", error);
                        println!("Appending empty vector to outputs due to EmbeddingExceedsMaxTokensBehavior::Skip");
                        outputs.push(vec![]);
                    }
                },
            }
        }
        Ok(outputs)
    }

    async fn check_embedding_input_tokens(&self, input: &String) -> Result<(), Box<dyn Error>> {
        let token_count = match &self.llm_client {
            LlmClient::OpenAiLlm(_) => text_utils::tiktoken_len(input),
            LlmClient::LlamaLlm(client) => client.llama_cpp_count_tokens(input).await.unwrap(),
        };
        if token_count > self.model_params.max_tokens_for_model {
            let preview_input = &input.chars().take(24).collect::<String>();
            let error_message = format!(
                "token_count {} is greater than max_tokens_for_model {} for input '{}...'",
                token_count, self.model_params.max_tokens_for_model, preview_input
            );
            return Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                error_message,
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{EmbeddingExceedsMaxTokensBehavior, LlmDefinition, ProviderClient};
    use crate::{
        providers::{
            llama_cpp::{
                models::{
                    LlamaDef,
                    DEFAULT_N_GPU_LAYERS,
                    DEFAULT_THREADS,
                    TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT,
                    TEST_LLM_URL_2_INSTRUCT,
                },
                server::kill_server,
            },
            llm_openai::models::OpenAiDef,
        },
        text_utils::load_content,
    };

    async fn get_clients() -> Vec<ProviderClient> {
        let client_openai: ProviderClient = ProviderClient::new(&EMBEDDING_OPENAI, None).await;

        let client_llama: ProviderClient = ProviderClient::new(
            &LlmDefinition::LlamaLlm((*LLAMA_EMBEDDING_LLM).clone()),
            None,
        )
        .await;

        vec![client_openai, client_llama]
    }

    // For embedding tests
    const EMBEDDING_CONTENT_1: &str = "I enjoy walking with my cute dog.";
    const EMBEDDING_CONTENT_2_PATH: &str = "tests/prompt_templates/split_by_topic_content.yaml";
    const EMBEDDING_OPENAI: LlmDefinition = LlmDefinition::OpenAiLlm(OpenAiDef::EmbeddingAda002);
    lazy_static! {
        #[derive(Debug)]
        pub static ref EMBEDDING_TEST_NORMAL: Vec<String> = vec![EMBEDDING_CONTENT_1.to_string(), load_content(EMBEDDING_CONTENT_2_PATH)];
        pub static ref EMBEDDING_TEST_EXCEEDS: Vec<String> = vec![EMBEDDING_CONTENT_1.to_string(), load_content(EMBEDDING_CONTENT_2_PATH), load_content(EMBEDDING_CONTENT_2_PATH) + load_content(EMBEDDING_CONTENT_2_PATH).as_str()+ load_content(EMBEDDING_CONTENT_2_PATH).as_str()];
        pub static ref LLAMA_EMBEDDING_LLM: LlamaDef = LlamaDef::new(
            TEST_LLM_URL_2_INSTRUCT,
            TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT,
            Some(8000),
            Some(DEFAULT_THREADS),
            Some(DEFAULT_N_GPU_LAYERS),
            Some(true),
            Some(true)
        );
    }

    #[tokio::test]
    async fn embeddings_normal_behavior() -> Result<(), Box<dyn std::error::Error>> {
        kill_server();
        let clients = get_clients().await;

        for client in clients {
            // Normal test
            let response = client
                .generate_embeddings(&EMBEDDING_TEST_NORMAL, None)
                .await?;
            for embedding in response {
                assert!(embedding.len() > 1);
            }
            // Test skipping input that exceeds max tokens
            let response = client
                .generate_embeddings(
                    &EMBEDDING_TEST_EXCEEDS,
                    Some(EmbeddingExceedsMaxTokensBehavior::Skip),
                )
                .await?;
            assert!(response.len() == 3);
            assert!(response[2].is_empty());
        }
        kill_server();
        Ok(())
    }

    #[tokio::test]
    #[should_panic]
    async fn embeddings_panic_behavior_openai() {
        let client_openai: ProviderClient = ProviderClient::new(&EMBEDDING_OPENAI, None).await;

        let _ = client_openai
            .generate_embeddings(
                &EMBEDDING_TEST_EXCEEDS,
                Some(EmbeddingExceedsMaxTokensBehavior::Panic),
            )
            .await;
    }
    #[tokio::test]
    #[should_panic]
    async fn embeddings_panic_behavior_llama() {
        kill_server();
        let client_llama: ProviderClient = ProviderClient::new(
            &LlmDefinition::LlamaLlm((*LLAMA_EMBEDDING_LLM).clone()),
            None,
        )
        .await;

        let _ = client_llama
            .generate_embeddings(
                &EMBEDDING_TEST_EXCEEDS,
                Some(EmbeddingExceedsMaxTokensBehavior::Panic),
            )
            .await;

        kill_server();
    }
}
