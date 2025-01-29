use super::local_model::hf_loader::{HfTokenTrait, HuggingFaceLoader};
use llm_prompt::PromptTokenizer;
use std::{
    fmt,
    path::{Path, PathBuf},
};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tokenizers::Tokenizer as HFTokenizer;

pub enum TokenizerBackend {
    HuggingFacesTokenizer(HFTokenizer),
    Tiktoken(CoreBPE),
}

impl fmt::Debug for TokenizerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerBackend::HuggingFacesTokenizer(_) => {
                write!(f, "TokenizerBackend::HuggingFacesTokenizer")
            }
            TokenizerBackend::Tiktoken(_) => {
                write!(f, "TokenizerBackend::Tiktoken")
            }
        }
    }
}

#[derive(Debug)]
pub struct LlmTokenizer {
    pub tokenizer: TokenizerBackend,
    pub tokenizer_path: Option<PathBuf>,
    pub with_special_tokens: bool,
    pub white_space_token_id: u32,
}

impl LlmTokenizer {
    pub fn new_tiktoken<T: AsRef<str>>(model_id: T) -> Result<Self, crate::Error> {
        let tokenizer = get_bpe_from_model(model_id.as_ref())?;
        let white_space_token_id = u32::try_from(tokenizer.encode_ordinary(" ").remove(0))?;
        Ok(Self {
            tokenizer: TokenizerBackend::Tiktoken(tokenizer),
            tokenizer_path: None,
            with_special_tokens: false,
            white_space_token_id,
        })
    }

    pub fn new_from_tokenizer(tokenizer: HFTokenizer) -> Result<Self, crate::Error> {
        let white_space_token_id = tokenizer.encode(" ", false).unwrap().get_ids()[0];
        Ok(Self {
            tokenizer: TokenizerBackend::HuggingFacesTokenizer(tokenizer),
            tokenizer_path: None,
            with_special_tokens: false,
            white_space_token_id,
        })
    }

    pub fn new_from_tokenizer_json<T: AsRef<Path>>(local_path: T) -> Result<Self, crate::Error> {
        let tokenizer = HFTokenizer::from_file(&local_path).map_err(|e| crate::anyhow!(e))?;
        let white_space_token_id = tokenizer.encode(" ", false).unwrap().get_ids()[0];
        Ok(Self {
            tokenizer: TokenizerBackend::HuggingFacesTokenizer(tokenizer),
            tokenizer_path: Some(local_path.as_ref().to_path_buf()),
            with_special_tokens: false,
            white_space_token_id,
        })
    }

    pub fn new_from_hf_repo(hf_token: Option<&str>, repo_id: &str) -> Result<Self, crate::Error> {
        let mut api: HuggingFaceLoader = HuggingFaceLoader::new();
        if let Some(hf_token) = hf_token {
            *api.hf_token_mut() = Some(hf_token.to_owned());
        }

        let local_path = api.load_file("tokenizer.json", repo_id)?;
        LlmTokenizer::new_from_tokenizer_json(&local_path)
    }

    pub fn tokenize<T: AsRef<str>>(&self, str: T) -> Vec<u32> {
        self.encode(str.as_ref())
    }

    pub fn detokenize_one(&self, token: u32) -> Result<String, crate::Error> {
        self.decode(&[token])
    }

    pub fn detokenize_many(&self, tokens: &[u32]) -> Result<String, crate::Error> {
        self.decode(tokens)
    }

    pub fn count_tokens(&self, str: &str) -> u32 {
        let tokens = self.tokenize(str);
        u32::try_from(tokens.len()).unwrap()
    }

    pub fn try_from_single_token_id(
        &self,
        try_from_single_token_id: u32,
    ) -> Result<String, crate::Error> {
        let detokenize_response = self.detokenize_one(try_from_single_token_id)?;
        println!("detokenize_response: {}", detokenize_response);
        let mut strings_maybe: Vec<String> = detokenize_response
            .split_ascii_whitespace()
            .map(|s| s.to_string())
            .collect();
        match strings_maybe.len() {
            0 => Err(crate::anyhow!(
                "token_id is empty for try_from_single_token_id: {}",
                try_from_single_token_id
            )),
            1 => Ok(strings_maybe.remove(0)),
            n => Err(crate::anyhow!(
                "Found more than one token ({n} total) in try_from_single_token_id: {}",
                try_from_single_token_id
            )),
        }
    }

    pub fn try_into_single_token(&self, try_into_single_token: &str) -> Result<u32, crate::Error> {
        let mut tokens = self.tokenize(try_into_single_token);
        match tokens.len() {
            0 => Err(crate::anyhow!(
                "No token found in text: {}",
                try_into_single_token
            )),
            1 => Ok(tokens.remove(0)),
            n => Err(crate::anyhow!(
                "Found more than one token ({n} total) in text: {}",
                try_into_single_token
            )),
        }
    }

    /// Creates a window of text normalized to the specified token size in the center of the text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to create a window from.
    /// * `target_token_size` - The desired number of tokens in the window.
    ///
    /// # Returns
    ///
    /// A new string that represents the normalized window of text, or the original
    /// text if its token count is less than or equal to `target_token_size`.
    pub fn create_text_window(&self, text: &str, target_token_size: u32) -> String {
        let tokens = self.tokenize(text);
        if tokens.len() <= target_token_size as usize {
            return text.to_string();
        }

        let start_token_index = (tokens.len() - target_token_size as usize) / 2;
        let end_token_index = start_token_index + target_token_size as usize;

        let preserved_tokens = &tokens[start_token_index..end_token_index];
        self.detokenize_many(preserved_tokens).unwrap()
    }

    /// Creates a range of text from the specified start and end token indices.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to create a window from.
    /// * `target_token_size` - The desired number of tokens in the window.
    ///
    /// # Returns
    ///
    /// A new string that represents the normalized window of text, or the original
    /// text if its token count is less than or equal to `target_token_size`.
    pub fn create_text_range(
        &self,
        text: &str,
        start_token_index: u32,
        end_token_index: u32,
    ) -> String {
        let tokens = self.tokenize(text);
        let end_token_index = if tokens.len() <= end_token_index as usize {
            tokens.len()
        } else {
            end_token_index as usize
        };

        let preserved_tokens = &tokens[start_token_index as usize..end_token_index];
        self.detokenize_many(preserved_tokens).unwrap()
    }

    fn encode_tiktoken(&self, tokenizer: &CoreBPE, str: &str) -> Vec<u32> {
        let tokens = if self.with_special_tokens {
            tokenizer
                .encode_with_special_tokens(str)
                .iter()
                .map(|&x| u32::try_from(x).unwrap())
                .collect()
        } else {
            tokenizer
                .encode_ordinary(str)
                .iter()
                .map(|&x| u32::try_from(x).unwrap())
                .collect()
        };
        tokens
    }

    fn encode_hf(&self, tokenizer: &HFTokenizer, str: &str) -> Vec<u32> {
        let tokens = if self.with_special_tokens {
            tokenizer.encode(str, true)
        } else {
            tokenizer.encode(str, false)
        };
        tokens.unwrap().get_ids().to_vec()
    }

    fn encode(&self, str: &str) -> Vec<u32> {
        match &self.tokenizer {
            TokenizerBackend::HuggingFacesTokenizer(tokenizer) => self.encode_hf(tokenizer, str),
            TokenizerBackend::Tiktoken(tokenizer) => self.encode_tiktoken(tokenizer, str),
        }
    }

    fn decode_tiktoken(&self, tokenizer: &CoreBPE, tokens: &[u32]) -> Result<String, crate::Error> {
        tokenizer
            .decode(tokens.into())
            .map_err(|e| crate::anyhow!(e))
    }

    fn decode_hf(&self, tokenizer: &HFTokenizer, tokens: &[u32]) -> Result<String, crate::Error> {
        tokenizer
            .decode(tokens, true)
            .map_err(|e| crate::anyhow!(e))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, crate::Error> {
        match &self.tokenizer {
            TokenizerBackend::HuggingFacesTokenizer(tokenizer) => self.decode_hf(tokenizer, tokens),
            TokenizerBackend::Tiktoken(tokenizer) => self.decode_tiktoken(tokenizer, tokens),
        }
    }
}

impl PromptTokenizer for LlmTokenizer {
    fn tokenize(&self, input: &str) -> Vec<u32> {
        self.tokenize(input)
    }

    fn count_tokens(&self, str: &str) -> u32 {
        self.count_tokens(str)
    }
}
