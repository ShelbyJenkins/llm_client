use llm_models::tokenizer::LlmTokenizer;

use crate::requests::req_components::RequestConfigTrait;

use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Default)]
pub struct LogitBias {
    pub base_logit_bias: Option<HashMap<u32, f32>>,
    pub built_llama_cpp_bias: LlamaCppLogitBias,
    pub built_openai_bias: OpenAiLogitBias,
    from_token_ids: FromTokenIds,
    from_chars: FromChars,
    from_words: FromWords,
    from_texts: FromTexts,
}

impl LogitBias {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_token_id(&mut self, token_id: u32, bias: f32) -> &mut Self {
        self.from_token_ids.add_token_id(token_id, bias);
        self.clear_built();
        self
    }

    pub fn add_token_ids(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        self.from_token_ids.add_token_ids(logit_bias);
        self.clear_built();
        self
    }

    pub fn add_from_char(&mut self, char: char, bias: f32) -> &mut Self {
        self.from_chars.add_char(char, bias);
        self.clear_built();
        self
    }

    pub fn add_from_word(&mut self, word: &str, bias: f32) -> &mut Self {
        self.from_words.add_word(word, bias);
        self.clear_built();
        self
    }

    pub fn add_from_text(&mut self, text: &str, bias: f32) -> &mut Self {
        self.from_texts.add_text(text, bias);
        self.clear_built();
        self
    }

    pub fn clear_logit_bias(&mut self) -> &mut Self {
        self.from_token_ids.clear();
        self.from_chars.clear();
        self.from_words.clear();
        self.from_texts.clear();
        self.clear_built();
        self
    }

    pub(crate) fn build_llama(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if !self.built_llama_cpp_bias.is_none() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base(tokenizer)?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            self.built_llama_cpp_bias.build(base_logit_bias);
        }
        Ok(())
    }

    pub(crate) fn build_openai(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if !self.built_openai_bias.is_none() {
            return Ok(());
        }
        if self.base_logit_bias.is_none() {
            self.build_base(tokenizer)?;
        }
        if let Some(base_logit_bias) = &self.base_logit_bias {
            self.built_openai_bias.build(base_logit_bias);
        }
        Ok(())
    }

    pub(crate) fn get_openai(&self) -> Option<HashMap<String, serde_json::Value>> {
        self.built_openai_bias.get()
    }

    pub(crate) fn get_llama_cpp(&self) -> Option<Vec<Vec<serde_json::Value>>> {
        self.built_llama_cpp_bias.get()
    }

    fn build_base(&mut self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<()> {
        if self.from_token_ids.is_none()
            && self.from_chars.is_none()
            && self.from_words.is_none()
            && self.from_texts.is_none()
        {
            return Ok(());
        }
        let validated_logit_bias = self.from_token_ids.get(tokenizer)?;
        self.from_token_ids.clear();

        let validated_logit_bias = Self::merge_logit_biases(vec![
            &validated_logit_bias,
            &self.from_chars.get(tokenizer)?,
        ]);
        self.from_chars.clear();

        let validated_logit_bias = Self::merge_logit_biases(vec![
            &validated_logit_bias,
            &self.from_words.get(tokenizer)?,
        ]);
        self.from_words.clear();

        let validated_logit_bias = Self::merge_logit_biases(vec![
            &validated_logit_bias,
            &self.from_texts.get(tokenizer)?,
        ]);
        self.from_texts.clear();

        if !validated_logit_bias.is_empty() {
            Self::validate_logit_bias_values(&validated_logit_bias)?;
            self.base_logit_bias = Some(validated_logit_bias);
        }
        Ok(())
    }

    fn clear_built(&mut self) -> &mut Self {
        self.base_logit_bias = None;
        self.built_llama_cpp_bias.clear();
        self.built_openai_bias.clear();
        self
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
    fn validate_logit_bias_values(logit_bias: &HashMap<u32, f32>) -> crate::Result<()> {
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

    /// Merges multiple logit biases into a single `HashMap` of token IDs and bias values.
    ///
    /// # Arguments
    ///
    /// * `logit_biases` - A vector of references to `HashMap`s containing logit biases with token IDs as keys and bias values as values.
    ///
    /// # Returns
    ///
    /// Returns a `HashMap<u32, f32>` containing the merged logit biases.
    fn merge_logit_biases(logit_biases: Vec<&HashMap<u32, f32>>) -> HashMap<u32, f32> {
        let mut merged_logit_bias: HashMap<u32, f32> = HashMap::new();
        for logit_bias in logit_biases {
            for (token_id, bias) in logit_bias {
                merged_logit_bias.insert(*token_id, *bias);
            }
        }
        merged_logit_bias
    }
}

#[derive(Clone, Default)]
struct FromTokenIds {
    pub token_ids: Option<HashMap<u32, f32>>,
}

impl FromTokenIds {
    fn is_none(&self) -> bool {
        self.token_ids.is_none()
    }

    fn clear(&mut self) {
        self.token_ids = None;
    }

    fn get(&self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<HashMap<u32, f32>> {
        if let Some(token_ids) = &self.token_ids {
            for token_id in token_ids.keys() {
                tokenizer.try_from_single_token_id(*token_id)?;
            }
            Ok(token_ids.clone())
        } else {
            Ok(HashMap::new())
        }
    }

    fn add_token_id(&mut self, token_id: u32, bias: f32) {
        self.token_ids
            .get_or_insert_with(HashMap::new)
            .entry(token_id)
            .or_insert(bias);
    }

    fn add_token_ids(&mut self, logit_bias: HashMap<u32, f32>) {
        self.token_ids
            .get_or_insert_with(HashMap::new)
            .extend(logit_bias);
    }
}

#[derive(Clone, Default)]
struct FromChars {
    pub chars: Option<HashMap<char, f32>>,
}

impl FromChars {
    fn is_none(&self) -> bool {
        self.chars.is_none()
    }

    fn clear(&mut self) {
        self.chars = None;
    }

    fn get(&self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<HashMap<u32, f32>> {
        if let Some(chars) = &self.chars {
            let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
            for (char, bias) in chars {
                let token_id = tokenizer.try_into_single_token(&char.to_string())?;
                token_logit_bias.insert(token_id, *bias);
            }
            Ok(token_logit_bias)
        } else {
            Ok(HashMap::new())
        }
    }

    fn add_char(&mut self, char: char, bias: f32) {
        self.chars
            .get_or_insert_with(HashMap::new)
            .entry(char)
            .or_insert(bias);
    }
}

#[derive(Clone, Default)]
struct FromWords {
    pub words: Option<HashMap<String, f32>>,
}

impl FromWords {
    fn is_none(&self) -> bool {
        self.words.is_none()
    }

    fn clear(&mut self) {
        self.words = None;
    }

    fn get(&self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<HashMap<u32, f32>> {
        if let Some(words) = &self.words {
            let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
            for (word_maybe, bias) in words {
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
        } else {
            Ok(HashMap::new())
        }
    }

    fn add_word(&mut self, word: &str, bias: f32) {
        self.words
            .get_or_insert_with(HashMap::new)
            .entry(word.to_owned())
            .or_insert(bias);
    }
}

#[derive(Clone, Default)]
struct FromTexts {
    pub texts: Option<HashMap<String, f32>>,
}

impl FromTexts {
    fn is_none(&self) -> bool {
        self.texts.is_none()
    }

    fn clear(&mut self) {
        self.texts = None;
    }

    fn get(&self, tokenizer: &Arc<LlmTokenizer>) -> crate::Result<HashMap<u32, f32>> {
        if let Some(texts) = &self.texts {
            let mut token_logit_bias: HashMap<u32, f32> = HashMap::new();
            for (text, bias) in texts {
                let token_ids = tokenizer.tokenize(text);
                for id in token_ids {
                    if id == tokenizer.white_space_token_id {
                        continue;
                    }
                    token_logit_bias.insert(id, *bias);
                }
            }
            Ok(token_logit_bias)
        } else {
            Ok(HashMap::new())
        }
    }

    fn add_text(&mut self, text: &str, bias: f32) {
        self.texts
            .get_or_insert_with(HashMap::new)
            .entry(text.to_owned())
            .or_insert(bias);
    }
}

#[derive(Clone, Default)]
pub struct OpenAiLogitBias {
    pub built_logit_bias: Option<HashMap<String, serde_json::Value>>,
}

impl OpenAiLogitBias {
    fn is_none(&self) -> bool {
        self.built_logit_bias.is_none()
    }

    fn clear(&mut self) {
        self.built_logit_bias = None;
    }

    fn build(&mut self, logit_bias: &HashMap<u32, f32>) {
        let mut openai_logit_bias: HashMap<String, serde_json::Value> = HashMap::new();
        for (token_id, value) in logit_bias {
            openai_logit_bias.insert(
                token_id.to_string(),
                serde_json::Value::Number(serde_json::Number::from(value.ceil() as i32)),
            );
        }
    }

    fn get(&self) -> Option<HashMap<String, serde_json::Value>> {
        self.built_logit_bias.clone()
    }
}

#[derive(Clone, Default)]
pub struct LlamaCppLogitBias {
    pub built_logit_bias: Option<Vec<Vec<serde_json::Value>>>,
}

impl LlamaCppLogitBias {
    fn is_none(&self) -> bool {
        self.built_logit_bias.is_none()
    }

    fn clear(&mut self) {
        self.built_logit_bias = None;
    }

    fn build(&mut self, logit_bias: &HashMap<u32, f32>) {
        let mut llama_logit_bias: Vec<Vec<serde_json::Value>> = Vec::new();
        for (token_id, value) in logit_bias {
            llama_logit_bias.push(vec![
                serde_json::Value::Number(serde_json::Number::from(*token_id)),
                serde_json::Value::Number(
                    serde_json::Number::from_f64((*value).into()).expect("Invalid float value"),
                ),
            ]);
        }
        self.built_logit_bias = Some(llama_logit_bias);
    }

    fn get(&self) -> Option<Vec<Vec<serde_json::Value>>> {
        self.built_logit_bias.clone()
    }
}

impl std::fmt::Display for LogitBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LogitBias: {:?}", self.base_logit_bias)?;
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
        self.logit_bias().add_token_id(token_id, bias);
        self
    }

    /// Adds multiple logit biases for token IDs. In the case you have your own tokenizer or other situations where you have token IDs.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A `HashMap` containing token IDs as keys and bias values as values.
    fn add_logit_bias_token_ids(&mut self, logit_bias: HashMap<u32, f32>) -> &mut Self {
        self.logit_bias().add_token_ids(logit_bias);
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
        self.logit_bias().add_from_char(char, bias);
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
        self.logit_bias().add_from_word(word, bias);
        self
    }

    /// Adds a logit bias for a specific text. Splits the text into tokens and applies the bias to each token. It does not add the logit bias value to the whitespace token.
    ///
    /// # Arguments
    ///
    /// * `text` - The text.
    /// * `bias` - The bias value.
    fn add_logit_bias_from_text(&mut self, text: &str, bias: f32) -> &mut Self {
        self.logit_bias().add_from_text(text, bias);
        self
    }

    /// Clearss the logit bias configuration. To reuse the request object for another request. Mostly for testing.
    fn clear_logit_bias(&mut self) -> &mut Self {
        self.logit_bias().clear_logit_bias();
        self
    }
}
