use llm_models::tokenizer::LlmTokenizer;

use crate::requests::req_components::RequestConfigTrait;

use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Default)]
pub struct LogitBias {
    pub base_logit_bias: Option<HashMap<u32, f32>>,
    pub built_llama_cpp_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub built_openai_bias: Option<HashMap<String, serde_json::Value>>,
    pub logit_bias_from_token_ids: Option<HashMap<u32, f32>>,
    pub logit_bias_from_chars: Option<HashMap<char, f32>>,
    pub logit_bias_from_words: Option<HashMap<String, f32>>,
    pub logit_bias_from_texts: Option<HashMap<String, f32>>,
}

impl LogitBias {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        self.on_add();
        self.base_logit_bias = Some(logit_bias);
        self
    }

    pub fn on_add(&mut self) -> &mut Self {
        self.base_logit_bias = None;
        self.built_llama_cpp_bias = None;
        self.built_openai_bias = None;

        self
    }

    pub fn build_base(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if self.logit_bias_from_token_ids.is_none()
            && self.logit_bias_from_chars.is_none()
            && self.logit_bias_from_words.is_none()
            && self.logit_bias_from_texts.is_none()
        {
            return Ok(());
        }
        let validated_logit_bias = if let Some(logit_bias_from_token_ids) =
            &self.logit_bias_from_token_ids
        {
            logit_bias_impl::validate_logit_bias_token_ids(tokenizer, logit_bias_from_token_ids)?;
            logit_bias_from_token_ids.clone()
        } else {
            HashMap::new()
        };
        self.logit_bias_from_token_ids = None;

        let validated_logit_bias = if let Some(logit_bias_from_chars) = &self.logit_bias_from_chars
        {
            logit_bias_impl::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias_impl::logit_bias_from_chars(tokenizer, logit_bias_from_chars)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_chars = None;

        let validated_logit_bias = if let Some(logit_bias_from_words) = &self.logit_bias_from_words
        {
            logit_bias_impl::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias_impl::logit_bias_from_words(tokenizer, logit_bias_from_words)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_words = None;

        let validated_logit_bias = if let Some(logit_bias_from_texts) = &self.logit_bias_from_texts
        {
            logit_bias_impl::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias_impl::logit_bias_from_texts(tokenizer, logit_bias_from_texts)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_texts = None;

        logit_bias_impl::validate_logit_bias_values(&validated_logit_bias)?;
        if !validated_logit_bias.is_empty() {
            self.base_logit_bias = Some(validated_logit_bias);
        }
        Ok(())
    }

    pub fn build_llama(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if self.built_llama_cpp_bias.is_some() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base(tokenizer)?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            let built_llama_cpp_bias =
                logit_bias_impl::convert_logit_bias_to_llama_format(base_logit_bias);
            self.built_llama_cpp_bias = Some(built_llama_cpp_bias);
        }
        Ok(())
    }

    pub fn build_openai(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if self.built_openai_bias.is_some() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base(tokenizer)?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            let built_openai_bias =
                logit_bias_impl::convert_logit_bias_to_openai_format(base_logit_bias);
            self.built_openai_bias = Some(built_openai_bias);
        }
        Ok(())
    }
}

pub trait LogitBiasTrait: RequestConfigTrait {
    fn lb_mut(&mut self) -> &mut Option<LogitBias>;

    fn logit_bias(&mut self) -> &mut LogitBias {
        if self.lb_mut().is_none() {
            *self.lb_mut() = Some(LogitBias::default());
        }
        self.lb_mut().as_mut().unwrap()
    }

    /// Adds a logit bias for a specific token ID. In the case you have your own tokenizer or other situations where you have token IDs.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The token ID.
    /// * `bias` - The bias value.
    fn add_logit_bias_token_id(&mut self, token_id: u32, bias: f32) -> &mut Self {
        if let Some(logit_bias) = &mut self.logit_bias().logit_bias_from_token_ids {
            logit_bias.entry(token_id).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(token_id).or_insert(bias);
            self.logit_bias().logit_bias_from_token_ids = Some(logit_bias);
        }
        self.logit_bias().on_add();
        self
    }

    /// Adds multiple logit biases for token IDs. In the case you have your own tokenizer or other situations where you have token IDs.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A `HashMap` containing token IDs as keys and bias values as values.
    fn add_logit_bias_token_ids(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias().logit_bias_from_token_ids {
            for (token_id, bias) in logit_bias {
                existing_logit_bias.insert(token_id, bias);
            }
        } else {
            self.logit_bias().logit_bias_from_token_ids = Some(logit_bias);
        }
        self.logit_bias().on_add();
        self
    }

    /// Adds a logit bias for a specific character.
    /// Not very useful as it does not necessarily remove all instances of that character as the character may be part of other tokens.
    ///
    /// # Arguments
    ///
    /// * `char` - The character.
    /// * `bias` - The bias value.
    fn add_logit_bias_from_char(&mut self, char: char, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias().logit_bias_from_chars {
            existing_logit_bias.entry(char.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(char.to_owned()).or_insert(bias);
            self.logit_bias().logit_bias_from_chars = Some(logit_bias);
        }
        self.logit_bias().on_add();
        self
    }

    /// Adds a logit bias for a specific word. If a word is more than one token, it will be split into multiple tokens.
    /// Errors if the word is empty or contains whitespace.
    ///
    /// # Arguments
    ///
    /// * `word` - The word.
    /// * `bias` - The bias value.
    fn add_logit_bias_from_word(&mut self, word: &str, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias().logit_bias_from_words {
            existing_logit_bias.entry(word.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(word.to_owned()).or_insert(bias);
            self.logit_bias().logit_bias_from_words = Some(logit_bias);
        }
        self.logit_bias().on_add();
        self
    }

    /// Adds a logit bias for a specific text. Splits the text into tokens and applies the bias to each token. It does not add the logit bias value to the whitespace token.
    ///
    /// # Arguments
    ///
    /// * `text` - The text.
    /// * `bias` - The bias value.
    fn add_logit_bias_from_text(&mut self, text: &str, bias: f32) -> &mut Self {
        if let Some(existing_logit_bias) = &mut self.logit_bias().logit_bias_from_texts {
            existing_logit_bias.entry(text.to_owned()).or_insert(bias);
        } else {
            let mut logit_bias = HashMap::new();
            logit_bias.entry(text.to_owned()).or_insert(bias);
            self.logit_bias().logit_bias_from_texts = Some(logit_bias);
        }
        self.logit_bias().on_add();
        self
    }

    /// Clearss the logit bias configuration. To reuse the request object for another request. Mostly for testing.
    fn clear_logit_bias(&mut self) -> &mut Self {
        self.logit_bias().logit_bias_from_token_ids = None;
        self.logit_bias().logit_bias_from_chars = None;
        self.logit_bias().logit_bias_from_words = None;
        self.logit_bias().logit_bias_from_texts = None;
        self.logit_bias().base_logit_bias = None;
        self.logit_bias().built_llama_cpp_bias = None;
        self.logit_bias().built_openai_bias = None;
        self
    }
}

pub mod logit_bias_impl {

    use super::*;

    /// Validates the logit bias token IDs by checking if each token ID can be converted to a single token.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<(), anyhow::Error>` indicating success or an error if any of the token IDs are invalid.
    pub fn validate_logit_bias_token_ids(
        tokenizer: &Arc<LlmTokenizer>,
        logit_bias: &HashMap<u32, f32>,
    ) -> crate::Result<()> {
        for token_id in logit_bias.keys() {
            tokenizer.try_from_single_token_id(*token_id)?;
        }
        Ok(())
    }

    /// Converts logit biases from characters to a `HashMap` with token IDs as keys and bias values as values.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with characters as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
    pub fn logit_bias_from_chars(
        tokenizer: &Arc<LlmTokenizer>,
        logit_bias: &HashMap<char, f32>,
    ) -> crate::Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (char, bias) in logit_bias {
            let token_id = tokenizer.try_into_single_token(&char.to_string())?;
            token_logit_bias.insert(token_id, *bias);
        }
        Ok(token_logit_bias)
    }

    /// Converts logit biases from words to a `HashMap` with token IDs as keys and bias values as values.
    /// If a word is longer than a single token, it will be split into multiple tokens.
    /// Returns an error if the word is empty or contains whitespace.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with words as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
    pub fn logit_bias_from_words(
        tokenizer: &Arc<LlmTokenizer>,
        logit_bias: &HashMap<String, f32>,
    ) -> crate::Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (word_maybe, bias) in logit_bias {
            let mut words_maybe: Vec<String> = word_maybe
                .trim()
                .split_ascii_whitespace()
                .map(|s| s.trim().to_string())
                .collect();
            let word = if words_maybe.is_empty() {
                return Err(crate::anyhow!(
                    "logit_bias contains an empty word. Given word: {}",
                    word_maybe
                ));
            } else if words_maybe.len() > 1 {
                return Err(crate::anyhow!(
                    "logit_bias contains a word seperated by whitespace. Given word: {}",
                    word_maybe
                ));
            } else {
                words_maybe.remove(0)
            };
            let token_ids = tokenizer.tokenize(&word);
            for id in token_ids {
                if id == tokenizer.white_space_token_id {
                    panic!(
                        "logit_bias contains a whitespace token. Given word: {}",
                        word
                    )
                }
                token_logit_bias.insert(id, *bias);
            }
        }
        Ok(token_logit_bias)
    }

    /// Converts logit biases from texts to a `HashMap` with token IDs as keys and bias values as values.
    /// This is for long texts that are tokenized into multiple tokens. It does not tokenize whitespace.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A reference to the `LlmTokenizer` used for tokenization.
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with texts as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<HashMap<u32, f32>, anyhow::Error>` containing the converted logit biases.
    pub fn logit_bias_from_texts(
        tokenizer: &Arc<LlmTokenizer>,
        logit_bias: &HashMap<String, f32>,
    ) -> crate::Result<HashMap<u32, f32>> {
        let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
        for (text, bias) in logit_bias {
            let token_ids = tokenizer.tokenize(text);
            for id in token_ids {
                if id == tokenizer.white_space_token_id {
                    continue;
                }
                token_logit_bias.insert(id, *bias);
            }
        }
        Ok(token_logit_bias)
    }

    /// Merges multiple logit biases into a single `HashMap` of token IDs and bias values.
    ///
    /// # Arguments
    ///
    /// * `logit_biases` - A vector of references to `HashMap`s containing logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns a `HashMap<u32, f32>` containing the merged logit biases.
    pub fn merge_logit_biases(logit_biases: Vec<&HashMap<u32, f32>>) -> HashMap<u32, f32> {
        let mut merged_logit_bias: HashMap<u32, f32> = HashMap::new();
        for logit_bias in logit_biases {
            for (token_id, bias) in logit_bias {
                merged_logit_bias.insert(*token_id, *bias);
            }
        }
        merged_logit_bias
    }

    /// Validates the logit bias values by checking if they are within the range of -100.0 to 100.0.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<(), anyhow::Error>` indicating success or an error if any of the bias values are out of range.
    pub fn validate_logit_bias_values(logit_bias: &HashMap<u32, f32>) -> crate::Result<()> {
        for value in logit_bias.values() {
            if *value > 100.0 || *value < -100.0 {
                return Err(crate::anyhow!(
                    "logit_bias value must be between -100.0 and 100.0. Given value: {}",
                    value
                ));
            }
        }
        Ok(())
    }

    /// Converts logit biases to the OpenAI format, where token IDs are represented as strings and bias values are rounded up to the nearest integer.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns `Result<HashMap<String, serde_json::Value>, anyhow::Error>` containing the converted logit biases in the OpenAI format.
    pub fn convert_logit_bias_to_openai_format(
        logit_bias: &HashMap<u32, f32>,
    ) -> HashMap<String, serde_json::Value> {
        let mut openai_logit_bias: HashMap<String, serde_json::Value> = HashMap::new();
        for (token_id, value) in logit_bias {
            openai_logit_bias.insert(
                token_id.to_string(),
                serde_json::Value::Number(serde_json::Number::from(value.ceil() as i32)),
            );
        }
        openai_logit_bias
    }

    /// Converts logit biases to the LLAMA format, where each logit bias is represented as a vector of two `serde_json::Value` elements: token ID and bias value.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A reference to the `HashMap` containing the logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns a `Vec<Vec<serde_json::Value>>` containing the converted logit biases in the LLAMA format.
    pub fn convert_logit_bias_to_llama_format(
        logit_bias: &HashMap<u32, f32>,
    ) -> Vec<Vec<serde_json::Value>> {
        let mut llama_logit_bias: Vec<Vec<serde_json::Value>> = Vec::new();
        for (token_id, value) in logit_bias {
            llama_logit_bias.push(vec![
                serde_json::Value::Number(serde_json::Number::from(*token_id)),
                serde_json::Value::Number(
                    serde_json::Number::from_f64((*value).into()).expect("Invalid float value"),
                ),
            ]);
        }
        llama_logit_bias
    }
}
