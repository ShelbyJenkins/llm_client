use crate::components::base_request::BaseRequestConfigTrait;
use anyhow::Result;
use llm_utils::{logit_bias, tokenizer::LlmTokenizer};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone)]
pub struct LogitBias {
    pub base_logit_bias: Option<HashMap<u32, f32>>,
    pub llama_logit_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub openai_logit_bias: Option<HashMap<String, serde_json::Value>>,
    pub logit_bias_from_token_ids: Option<HashMap<u32, f32>>,
    pub logit_bias_from_chars: Option<HashMap<char, f32>>,
    pub logit_bias_from_words: Option<HashMap<String, f32>>,
    pub logit_bias_from_texts: Option<HashMap<String, f32>>,
    tokenizer: Arc<LlmTokenizer>,
}

impl LogitBias {
    pub fn new(tokenizer: Arc<LlmTokenizer>) -> Self {
        Self {
            base_logit_bias: None,
            llama_logit_bias: None,
            openai_logit_bias: None,
            logit_bias_from_token_ids: None,
            logit_bias_from_chars: None,
            logit_bias_from_words: None,
            logit_bias_from_texts: None,
            tokenizer,
        }
    }

    pub fn set(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        self.on_add();
        self.base_logit_bias = Some(logit_bias);
        self
    }

    pub fn on_add(&mut self) -> &mut Self {
        self.base_logit_bias = None;
        self.llama_logit_bias = None;
        self.openai_logit_bias = None;

        self
    }

    pub fn build_base(&mut self) -> Result<()> {
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
            logit_bias::validate_logit_bias_token_ids(&self.tokenizer, logit_bias_from_token_ids)?;
            logit_bias_from_token_ids.clone()
        } else {
            HashMap::new()
        };
        self.logit_bias_from_token_ids = None;

        let validated_logit_bias = if let Some(logit_bias_from_chars) = &self.logit_bias_from_chars
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias::logit_bias_from_chars(&self.tokenizer, logit_bias_from_chars)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_chars = None;

        let validated_logit_bias = if let Some(logit_bias_from_words) = &self.logit_bias_from_words
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias::logit_bias_from_words(&self.tokenizer, logit_bias_from_words)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_words = None;

        let validated_logit_bias = if let Some(logit_bias_from_texts) = &self.logit_bias_from_texts
        {
            logit_bias::merge_logit_biases(vec![
                &validated_logit_bias,
                &logit_bias::logit_bias_from_texts(&self.tokenizer, logit_bias_from_texts)?,
            ])
        } else {
            validated_logit_bias
        };
        self.logit_bias_from_texts = None;

        logit_bias::validate_logit_bias_values(&validated_logit_bias)?;
        if !validated_logit_bias.is_empty() {
            self.base_logit_bias = Some(validated_logit_bias);
        }
        Ok(())
    }

    pub fn build_llama(&mut self) -> Result<()> {
        if self.llama_logit_bias.is_some() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base()?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            let llama_logit_bias = logit_bias::convert_logit_bias_to_llama_format(base_logit_bias);
            self.llama_logit_bias = Some(llama_logit_bias);
        }
        Ok(())
    }

    pub fn build_openai(&mut self) -> Result<()> {
        if self.openai_logit_bias.is_some() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base()?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            let openai_logit_bias =
                logit_bias::convert_logit_bias_to_openai_format(base_logit_bias);
            self.openai_logit_bias = Some(openai_logit_bias);
        }
        Ok(())
    }

    pub fn llama(&self) -> Vec<Vec<serde_json::Value>> {
        if let Some(llama_logit_bias) = &self.llama_logit_bias {
            llama_logit_bias.clone()
        } else {
            panic!("logit_bias for llama called without logit_bias being set.")
        }
    }

    pub fn openai(&self) -> HashMap<String, serde_json::Value> {
        if let Some(openai_logit_bias) = &self.openai_logit_bias {
            openai_logit_bias.clone()
        } else {
            panic!("logit_bias for openai called without logit_bias being set.")
        }
    }
}

pub trait LogitBiasTrait: BaseRequestConfigTrait {
    fn lb_mut(&mut self) -> &mut Option<LogitBias>;

    fn logit_bias(&mut self) -> &mut LogitBias {
        if self.lb_mut().is_none() {
            let logit_bias = self.base_config().backend.get_logit_bias();
            *self.lb_mut() = Some(logit_bias);
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
        self.logit_bias().llama_logit_bias = None;
        self.logit_bias().openai_logit_bias = None;
        self
    }
}
