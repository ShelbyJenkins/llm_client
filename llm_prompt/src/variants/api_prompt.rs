use crate::{token_count::total_prompt_tokens_openai_format, PromptTokenizer};
use serde::Serialize;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

/// A prompt formatter for API-based language models that follow OpenAI's message format.
///
/// `ApiPrompt` handles formatting messages into the standard role/content pairs used by
/// API-based LLMs. It manages token counting specific to these
/// models, including per-message and per-name token overhead.
///
/// The struct maintains thread-safe interior mutability for built messages and token counts,
/// rebuilding them as needed when the prompt content changes.
#[derive(Serialize)]
pub struct ApiPrompt {
    #[serde(skip)]
    tokenizer: Arc<dyn PromptTokenizer>,
    tokens_per_message: Option<u32>,
    tokens_per_name: Option<i32>,
    built_prompt_messages: Mutex<Option<Vec<HashMap<String, String>>>>,
    total_prompt_tokens: Mutex<Option<u64>>,
}

impl ApiPrompt {
    pub fn new(
        tokenizer: Arc<dyn PromptTokenizer>,
        tokens_per_message: Option<u32>,
        tokens_per_name: Option<i32>,
    ) -> Self {
        Self {
            tokenizer,
            tokens_per_message,
            tokens_per_name,
            total_prompt_tokens: None.into(),
            built_prompt_messages: None.into(),
        }
    }

    // Setter methods
    //

    pub(crate) fn clear_built_prompt(&self) {
        *self.built_prompt_messages() = None;
        *self.total_prompt_tokens() = None;
    }

    // Getter methods
    //

    /// Retrieves the built prompt messages in OpenAI API format.
    ///
    /// Returns the messages as a vector of hashmaps, where each message contains
    /// a "role" key (system/user/assistant) and a "content" key with the message text.
    /// Messages must be built before they can be retrieved.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<HashMap<String, String>>)` containing the formatted messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt has not been built yet.
    pub fn get_built_prompt(&self) -> Result<Vec<HashMap<String, String>>, crate::Error> {
        match &*self.built_prompt_messages() {
            Some(prompt) => Ok(prompt.clone()),
            None => crate::bail!(
                "ApiPrompt Error - built_prompt_messages not available - prompt not built"
            ),
        }
    }

    /// Gets the total number of tokens in the prompt, including any model-specific overhead.
    ///
    /// The total includes the base tokens from all messages plus any additional tokens
    /// specified by `tokens_per_message` and `tokens_per_name`. This count is useful for
    /// ensuring prompts stay within model context limits.
    ///
    /// # Returns
    ///
    /// Returns `Ok(u64)` containing the total token count.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt has not been built yet.
    pub fn get_total_prompt_tokens(&self) -> Result<u64, crate::Error> {
        match &*self.total_prompt_tokens() {
            Some(prompt) => Ok(*prompt),
            None => crate::bail!(
                "ApiPrompt Error - total_prompt_tokens not available - prompt not built"
            ),
        }
    }

    // Builder methods
    //

    pub(crate) fn build_prompt(&self, built_prompt_messages: &Vec<HashMap<String, String>>) {
        *self.total_prompt_tokens() = Some(total_prompt_tokens_openai_format(
            &built_prompt_messages,
            self.tokens_per_message,
            self.tokens_per_name,
            &self.tokenizer,
        ));

        *self.built_prompt_messages() = Some(built_prompt_messages.clone());
    }

    // Helper methods
    //

    fn built_prompt_messages(&self) -> MutexGuard<'_, Option<Vec<HashMap<String, String>>>> {
        self.built_prompt_messages.lock().unwrap_or_else(|e| {
            panic!(
                "ApiPrompt Error - built_prompt_messages not available: {:?}",
                e
            )
        })
    }

    fn total_prompt_tokens(&self) -> MutexGuard<'_, Option<u64>> {
        self.total_prompt_tokens.lock().unwrap_or_else(|e| {
            panic!(
                "ApiPrompt Error - total_prompt_tokens not available: {:?}",
                e
            )
        })
    }
}

impl Clone for ApiPrompt {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            tokens_per_message: self.tokens_per_message,
            tokens_per_name: self.tokens_per_name,
            total_prompt_tokens: self.total_prompt_tokens().clone().into(),
            built_prompt_messages: self.built_prompt_messages().clone().into(),
        }
    }
}

impl std::fmt::Display for ApiPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ApiPrompt")?;

        // Because the main LlmPrompt display impl already prints the built_prompt_messages
        // match *self.built_prompt_messages() {
        //     Some(ref prompt) => {
        //         writeln!(f, "built_prompt_messages:\n{:?}", prompt)?;
        //         writeln!(f)?;
        //     }
        //     None => writeln!(f, "built_prompt_messages: None")?,
        // };

        match *self.total_prompt_tokens() {
            Some(ref prompt) => {
                writeln!(f, "total_prompt_tokens:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "total_prompt_tokens: None")?,
        };

        Ok(())
    }
}
