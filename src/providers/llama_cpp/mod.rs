use std::{collections::HashMap, error::Error, io};

pub mod api;
pub mod model_loader;
pub mod models;
pub mod server;
use api::{
    client::Client,
    config::LlamaConfig,
    types::{
        LlamaCreateCompletionsRequestArgs,
        LlamaCreateDetokenizeRequestArgs,
        LlamaCreateEmbeddingRequestArgs,
        LlamaCreateTokenizeRequestArgs,
    },
};
pub use models::LlamaDef;

const LLAMA_PATH: &str = "src/providers/llama_cpp/llama_cpp";

pub struct LlamaClient {
    pub safety_tokens: u16,
    client: Client<LlamaConfig>,
}

// Because this function is run for LlamaClient::new(), it cannot be implemented as part of the LlamaClient struct.
// Otherwise it would cause a recusive loop.
pub async fn get_model_info() -> Result<String, Box<dyn Error>> {
    let default_client = LlamaClient::setup_client().await;
    let request = LlamaCreateCompletionsRequestArgs::default()
        .prompt("test")
        .n_predict(1u16)
        .build()?;

    let response = default_client.completions().create(request).await;
    if let Err(error) = response {
        let error_message = format!("get_model_info failed with error: {}.", error);

        Err(Box::new(io::Error::new(
            io::ErrorKind::Other,
            error_message,
        )))
    } else {
        Ok(response.unwrap().model)
    }
}

impl LlamaClient {
    /// Creates a new instance of `LlamaClient`.
    ///
    /// # Returns
    ///
    /// A new instance of `LlamaClient`.
    pub async fn new(llama_def: &LlamaDef) -> Self {
        let check = server::check_and_or_start_server(llama_def).await;
        if let Err(error) = check {
            panic!("Failed to start server with error: {}", error);
        }

        Self {
            safety_tokens: 10,
            client: Self::setup_client().await,
        }
    }
    /// Sets up the client with the necessary configuration.
    ///
    /// # Returns
    ///
    /// A configured `Client` instance.
    async fn setup_client() -> Client<LlamaConfig> {
        let backoff = backoff::ExponentialBackoffBuilder::new()
            .with_max_elapsed_time(Some(std::time::Duration::from_secs(60)))
            .build();

        let config =
            LlamaConfig::new().with_api_base(format!("http://{}", *server::SERVER_ADDRESS));

        Client::with_config(config).with_backoff(backoff)
    }
    /// Tokenizes a list of characters. Used for generating logit bias.
    ///
    /// # Arguments
    ///
    /// * `text` - A vector of strings representing the characters to tokenize.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of token indices on success, or a boxed `dyn Error` on failure.
    pub async fn tokenize_chars(&self, text: &Vec<String>) -> Result<Vec<usize>, Box<dyn Error>> {
        let mut tokens: Vec<usize> = Vec::new();
        for char in text {
            let request = LlamaCreateTokenizeRequestArgs::default()
                .content(char)
                .build()?;
            let response = self.client.tokenize().create(request).await;
            if let Err(error) = response {
                return Err(Box::new(error));
            } else {
                let response_tokens = response.unwrap().tokens;
                // Llama.cpp sometimes returns an extra whitespace '/s' for a single character,
                // so we need to filter out the correct one.
                let mut good_tokens: Vec<usize> = vec![];
                for token in &response_tokens {
                    let response_content: Vec<String> = self
                        .detokenize(vec![*token])
                        .await?
                        .split_ascii_whitespace()
                        .map(|s| s.to_string())
                        .collect();

                    if response_content.len() > 1 {
                        panic!(
                            "More than one string found in response_content from detokenize: {:?}",
                            response_content
                        );
                    } else if response_content.is_empty() {
                        continue;
                    } else if response_content[0] == *char {
                        good_tokens.push(*token);
                    }
                }
                if !good_tokens.is_empty() {
                    tokens.push(good_tokens[0])
                }
            }
        }
        Ok(tokens)
    }

    /// Counts the number of tokens in a given text. Used for calculating prompt length
    ///
    /// # Arguments
    ///
    /// * `text` - A string representing the text to count tokens in.
    ///
    /// # Returns
    ///
    /// A `Result` containing the token count as a `u16` on success, or a boxed `dyn Error` on failure.
    pub async fn llama_cpp_count_tokens(&self, text: &String) -> Result<u16, Box<dyn Error>> {
        let token_count: u16;

        let request = LlamaCreateTokenizeRequestArgs::default()
            .content(text)
            .build()?;
        let response = self.client.tokenize().create(request).await;
        if let Err(error) = response {
            return Err(Box::new(error));
        } else {
            let response_tokens = response.unwrap().tokens;
            token_count = response_tokens.len() as u16;
        }

        Ok(token_count)
    }

    /// Detokenizes a list of token indices. Used for checking returned tokens from tokenize_chars.
    ///
    /// # Arguments
    ///
    /// * `tokens` - A vector of token indices to detokenize.
    ///
    /// # Returns
    ///
    /// A `Result` containing the detokenized string on success, or a boxed `dyn Error` on failure.
    pub async fn detokenize(&self, tokens: Vec<usize>) -> Result<String, Box<dyn Error>> {
        let request = LlamaCreateDetokenizeRequestArgs::default()
            .tokens(tokens.clone())
            .build()?;
        let response = self.client.detokenize().create(request).await;
        if let Err(error) = response {
            Err(Box::new(error))
        } else {
            Ok(response.unwrap().content)
        }
    }

    /// Makes a boolean decision based on a given prompt and model parameters.
    ///
    /// # Arguments
    ///
    /// * `prompt` - A hashmap representing the prompt for the decision.
    /// * `logit_bias` - A hashmap representing the logit bias for the decision.
    /// * `batch_count` - The number of decisions to make.
    /// * `model_params` - The model parameters for the decision.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of decision outputs on success, or a boxed `dyn Error` on failure.
    pub async fn make_boolean_decision(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        logit_bias: &HashMap<String, serde_json::Value>,
        batch_count: u8,
        model_params: &crate::LlmModelParams,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let mut output = vec![];

        let logit_bias_vec: Vec<serde_json::Value> = logit_bias
            .iter()
            .filter_map(|(k, v)| {
                k.parse::<i64>().ok().map(|num| {
                    serde_json::json!([serde_json::Value::Number(num.into()), v.clone()])
                })
            })
            .collect();

        for _ in 0..batch_count {
            let request = LlamaCreateCompletionsRequestArgs::default()
                .prompt(prompt["llama_prompt"]["content"].clone())
                .frequency_penalty(model_params.frequency_penalty)
                .presence_penalty(model_params.presence_penalty)
                .temperature(1.5)
                .top_p(model_params.top_p)
                .n_predict(1_u16)
                .logit_bias(logit_bias_vec.clone())
                .stop(vec!["0".to_string(), "1".to_string()])
                .build()?;

            let response = self.client.completions().create(request).await;
            if let Err(error) = response {
                return Err(Box::new(error));
            } else {
                let response = response.unwrap();
                if response.stopping_word.is_empty() {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::Other,
                        "Stopping word is empty when it should be '1' or '0'.",
                    )));
                } else {
                    output.push(response.stopping_word);
                }
            }
        }
        Ok(output)
    }

    /// Generates text based on a given prompt and model parameters.
    ///
    /// # Arguments
    ///
    /// * `prompt` - A hashmap representing the prompt for text generation.
    /// * `max_response_tokens` - The maximum number of tokens to generate.
    /// * `logit_bias` - An optional hashmap representing the logit bias for text generation.
    /// * `model_params` - The model parameters for text generation.
    ///
    /// # Returns
    ///
    /// A `Result` containing the generated text as a string and the number of tokens used on success,
    /// or a boxed `dyn Error` on failure.
    pub async fn generate_text(
        &self,
        prompt: &HashMap<String, HashMap<String, String>>,
        max_response_tokens: u16,
        logit_bias: &Option<HashMap<String, serde_json::Value>>,
        model_params: &crate::LlmModelParams,
    ) -> Result<String, Box<dyn Error>> {
        let mut request_builder = LlamaCreateCompletionsRequestArgs::default()
            .prompt(prompt["llama_prompt"]["content"].clone())
            .n_predict(max_response_tokens)
            .frequency_penalty(model_params.frequency_penalty)
            .presence_penalty(model_params.presence_penalty)
            .temperature(model_params.temperature)
            .top_p(model_params.top_p)
            .clone();
        if let Some(logit_bias) = logit_bias {
            // Have to convert the HashMap to a Vec<Vec<Value>> for the API
            let logit_bias_vec: Vec<Vec<String>> = logit_bias
                .iter()
                .map(|(k, v)| vec![k.to_string(), v.to_string()])
                .collect();
            request_builder.logit_bias(logit_bias_vec.clone());
        }
        let request = request_builder.build()?;
        let response = self.client.completions().create(request).await;
        if let Err(error) = response {
            let error_message = format!("Failed generating tokens with error: {}.", error);
            Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                error_message,
            )))
        } else {
            Ok(response.unwrap().content)
        }
    }
    pub async fn generate_embedding(&self, input: &String) -> Result<Vec<f32>, Box<dyn Error>> {
        let request = LlamaCreateEmbeddingRequestArgs::default()
            .content(input)
            .build()?;

        let response = self.client.embedding().create(request).await?;

        Ok(response.embedding.to_owned())
    }
}

pub fn get_llama_cpp_path() -> String {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let llama_path =
        std::fs::canonicalize(manifest_dir.join(LLAMA_PATH)).expect("Failed to canonicalize path");

    llama_path
        .to_str()
        .expect("Failed to convert path to string")
        .to_string()
}
