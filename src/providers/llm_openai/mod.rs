use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs,
    },
    Client,
};
use dotenv::dotenv;
pub use models::OpenAiDef;
use std::{collections::HashMap, error::Error};
pub mod models;

pub struct OpenAiClient {
    pub safety_tokens: u16,
    client: Client<OpenAIConfig>,
}

impl Default for OpenAiClient {
    fn default() -> Self {
        dotenv().ok(); // Load .env file
        Self {
            safety_tokens: 10,
            client: Self::setup_client(),
        }
    }
}
impl OpenAiClient {
    fn setup_client() -> Client<OpenAIConfig> {
        let backoff = backoff::ExponentialBackoffBuilder::new()
            .with_max_elapsed_time(Some(std::time::Duration::from_secs(60)))
            .build();
        let api_key = dotenv::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
        let config = OpenAIConfig::new().with_api_key(api_key);
        Client::with_config(config).with_backoff(backoff)
    }

    pub async fn make_boolean_decision(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        logit_bias: &HashMap<String, serde_json::Value>,
        batch_count: u8,
        model_params: &crate::LlmModelParams,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model_params.model_id.to_string())
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(&prompt["system"]["content"].clone())
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt["user"]["content"].clone())
                    .build()?
                    .into(),
            ])
            .max_tokens(1_u16)
            .n(batch_count)
            .logit_bias(logit_bias.clone())
            .build()?;

        let response = self.client.chat().create(request).await?;
        let mut output = vec![];

        for choice in response.choices {
            output.push(choice.message.content.clone().unwrap());
        }
        Ok(output)
    }

    pub async fn generate_text(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        max_response_tokens: u16,
        logit_bias: &Option<HashMap<String, serde_json::Value>>,
        model_params: &crate::LlmModelParams,
    ) -> Result<String, Box<dyn Error>> {
        let mut request_builder = CreateChatCompletionRequestArgs::default()
            .model(model_params.model_id.to_string())
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(prompt["system"]["content"].clone())
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt["user"]["content"].clone())
                    .build()?
                    .into(),
            ])
            .max_tokens(max_response_tokens)
            .frequency_penalty(model_params.frequency_penalty)
            .presence_penalty(model_params.presence_penalty)
            .temperature(model_params.temperature)
            .top_p(model_params.top_p)
            .clone();

        if let Some(logit_bias) = logit_bias {
            request_builder.logit_bias(logit_bias.clone());
        }
        let request = request_builder.build()?;
        let response = self.client.chat().create(request).await?;

        let output = response.choices[0].message.content.clone().unwrap();
        Ok(output)
    }

    pub async fn generate_embedding(
        &self,
        input: &String,
        model_params: &crate::LlmModelParams,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(model_params.model_id.to_string())
            .input(input)
            .build()?;

        let response: async_openai::types::CreateEmbeddingResponse =
            self.client.embeddings().create(request).await?;

        Ok(response.data[0].embedding.to_owned())
    }
}
