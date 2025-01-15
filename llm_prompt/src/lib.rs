mod concatenator;
mod prompt_message;
mod prompt_tokenizer;
mod token_count;
mod variants;

pub use concatenator::{TextConcatenator, TextConcatenatorTrait};
pub use prompt_message::{PromptMessage, PromptMessageType, PromptMessages};
pub use prompt_tokenizer::PromptTokenizer;
pub use token_count::{check_and_get_max_tokens, MaxTokenState, RequestTokenLimitError};
pub use variants::{apply_chat_template, ApiPrompt, LocalPrompt};

pub(crate) use anyhow::{bail, Error, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// A prompt management system that supports both API-based LLMs (like OpenAI) and local LLMs.
///
/// `LlmPrompt` provides a unified interface for building and managing prompts in different formats,
/// with support for both API-style messaging (system/user/assistant) and local LLM chat templates.
/// It handles token counting, message validation, and proper prompt formatting.
/// ```
#[derive(Serialize)]
pub struct LlmPrompt {
    pub local_prompt: Option<LocalPrompt>,
    pub api_prompt: Option<ApiPrompt>,
    pub messages: PromptMessages,
    pub concatenator: TextConcatenator,
    pub built_prompt_messages: Mutex<Option<Vec<HashMap<String, String>>>>,
}

impl LlmPrompt {
    /// Creates a new prompt instance configured for local LLMs using chat templates.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A tokenizer implementation for counting tokens
    /// * `chat_template` - The chat template string used to format messages
    /// * `bos_token` - Optional beginning of sequence token
    /// * `eos_token` - End of sequence token
    /// * `unk_token` - Optional unknown token
    /// * `base_generation_prefix` - Optional prefix to add before generation
    ///
    /// # Returns
    ///
    /// A new `LlmPrompt` instance configured for local LLM usage.
    pub fn new_local_prompt(
        tokenizer: std::sync::Arc<dyn PromptTokenizer>,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: &str,
        unk_token: Option<&str>,
        base_generation_prefix: Option<&str>,
    ) -> Self {
        Self {
            local_prompt: Some(LocalPrompt::new(
                tokenizer,
                chat_template,
                bos_token,
                eos_token,
                unk_token,
                base_generation_prefix,
            )),
            ..Default::default()
        }
    }

    /// Creates a new prompt instance configured for API-based LLMs like OpenAI.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A tokenizer implementation for counting tokens
    /// * `tokens_per_message` - Optional number of tokens to add per message (model-specific)
    /// * `tokens_per_name` - Optional number of tokens to add for names (model-specific)
    ///
    /// # Returns
    ///
    /// A new `LlmPrompt` instance configured for API usage.
    pub fn new_api_prompt(
        tokenizer: std::sync::Arc<dyn PromptTokenizer>,
        tokens_per_message: Option<u32>,
        tokens_per_name: Option<i32>,
    ) -> Self {
        Self {
            api_prompt: Some(ApiPrompt::new(
                tokenizer,
                tokens_per_message,
                tokens_per_name,
            )),
            ..Default::default()
        }
    }

    // Setter methods
    //

    /// Adds a system message to the prompt.
    ///
    /// System messages must be the first message in the sequence.
    /// Returns an error if attempting to add a system message after other messages.
    ///
    /// # Returns
    ///
    /// A reference to the newly created message for setting content, or an error if validation fails.
    pub fn add_system_message(&self) -> Result<Arc<PromptMessage>, crate::Error> {
        {
            let mut messages = self.messages();

            if !messages.is_empty() {
                crate::bail!("System message must be first message.");
            };

            let message = Arc::new(PromptMessage::new(
                PromptMessageType::System,
                &self.concatenator,
            ));
            messages.push(message);
        }
        self.clear_built_prompt();
        Ok(self.last_message())
    }

    /// Adds a user message to the prompt.
    ///
    /// Cannot add a user message directly after another user message.
    /// Returns an error if attempting to add consecutive user messages.
    ///
    /// # Returns
    ///
    /// A reference to the newly created message for setting content, or an error if validation fails.
    pub fn add_user_message(&self) -> Result<Arc<PromptMessage>, crate::Error> {
        {
            let mut messages = self.messages();

            if let Some(last) = messages.last() {
                if last.message_type == PromptMessageType::User {
                    crate::bail!("Cannot add user message when previous message is user message.");
                }
            }

            let message = Arc::new(PromptMessage::new(
                PromptMessageType::User,
                &self.concatenator,
            ));
            messages.push(message);
        }
        self.clear_built_prompt();
        Ok(self.last_message())
    }

    /// Adds an assistant message to the prompt.
    ///
    /// Cannot be the first message or follow another assistant message.
    /// Returns an error if attempting to add as first message or after another assistant message.
    ///
    /// # Returns
    ///
    /// A reference to the newly created message for setting content, or an error if validation fails.
    pub fn add_assistant_message(&self) -> Result<Arc<PromptMessage>, crate::Error> {
        {
            let mut messages = self.messages();

            if messages.is_empty() {
                crate::bail!("Cannot add assistant message as first message.");
            } else {
                if let Some(last) = messages.last() {
                    if last.message_type == PromptMessageType::Assistant {
                        crate::bail!( "Cannot add assistant message when previous message is assistant message.");
                    }
                }
            };

            let message = Arc::new(PromptMessage::new(
                PromptMessageType::Assistant,
                &self.concatenator,
            ));
            messages.push(message);
        }
        self.clear_built_prompt();
        Ok(self.last_message())
    }

    /// Sets a prefix to be added before generation for local LLMs.
    ///
    /// This is typically used to prime the model's response.
    /// Only applies to local LLM prompts, has no effect on API prompts.
    ///
    /// # Arguments
    ///
    /// * `generation_prefix` - The text to add before generation
    pub fn set_generation_prefix<T: AsRef<str>>(&self, generation_prefix: T) {
        self.clear_built_prompt();
        if let Some(local_prompt) = &self.local_prompt {
            local_prompt.set_generation_prefix(generation_prefix);
        };
    }

    /// Clears any previously set generation prefix.
    pub fn clear_generation_prefix(&self) {
        self.clear_built_prompt();
        if let Some(local_prompt) = &self.local_prompt {
            local_prompt.clear_generation_prefix();
        };
    }

    /// Resets the prompt, clearing all messages and built state.
    pub fn reset_prompt(&self) {
        self.messages().clear();
        self.clear_built_prompt();
    }

    /// Clears any built prompt state, forcing a rebuild on next access.
    pub fn clear_built_prompt(&self) {
        if let Some(api_prompt) = &self.api_prompt {
            api_prompt.clear_built_prompt();
        };
        if let Some(local_prompt) = &self.local_prompt {
            local_prompt.clear_built_prompt();
        };
    }

    // Getter methods
    //

    /// Gets and builds the local prompt if this is prompt has one.
    ///
    /// # Returns
    ///
    /// A reference to the `LocalPrompt` if present, otherwise returns an error
    pub fn local_prompt(&self) -> Result<&LocalPrompt, crate::Error> {
        if let Some(local_prompt) = &self.local_prompt {
            if local_prompt.get_built_prompt().is_err() {
                self.precheck_build()?;
                self.build_prompt()?;
            }
            Ok(local_prompt)
        } else {
            crate::bail!("LocalPrompt is None");
        }
    }

    /// Gets and builds the API prompt if this is prompt has one.
    ///
    /// # Returns
    ///
    /// A reference to the `ApiPrompt` if present, otherwise returns an error
    pub fn api_prompt(&self) -> Result<&ApiPrompt, crate::Error> {
        if let Some(api_prompt) = &self.api_prompt {
            if api_prompt.get_built_prompt().is_err() {
                self.precheck_build()?;
                self.build_prompt()?;
            }
            Ok(api_prompt)
        } else {
            crate::bail!("ApiPrompt is None");
        }
    }

    /// Retrieves the prompt messages in a standardized format compatible with API calls.
    ///
    /// This method returns messages in the same format as `ApiPrompt::get_built_prompt()`,
    /// making it useful for consistent message handling across different LLM implementations.
    /// The method handles lazy building of the prompt - if the messages haven't been built yet,
    /// it will trigger the build process automatically.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<HashMap<String, String>>)` containing the formatted messages on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The current message sequence violates prompt rules (e.g., assistant message first)
    /// - The build process fails
    /// - The built messages are unexpectedly None after building
    pub fn get_built_prompt_messages(&self) -> Result<Vec<HashMap<String, String>>, crate::Error> {
        let built_prompt_messages = self.built_prompt_messages();

        if let Some(built_prompt_messages) = &*built_prompt_messages {
            return Ok(built_prompt_messages.clone());
        };

        self.precheck_build()?;
        self.build_prompt()?;
        if let Some(built_prompt_messages) = &*built_prompt_messages {
            Ok(built_prompt_messages.clone())
        } else {
            crate::bail!("built_prompt_messages is None after building!");
        }
    }

    // Builder methods
    //

    fn precheck_build(&self) -> crate::Result<()> {
        if let Some(last) = self.messages().last() {
            if last.message_type == PromptMessageType::Assistant {
                crate::bail!(
                    "Cannot build prompt when the current inference message is PromptMessageType::Assistant"
                )
            } else if last.message_type == PromptMessageType::System {
                crate::bail!("Cannot build prompt when the current inference message is PromptMessageType::System")
            } else {
                Ok(())
            }
        } else {
            crate::bail!("Cannot build prompt when there are no messages.")
        }
    }

    fn build_prompt(&self) -> crate::Result<()> {
        let messages = self.messages();
        let mut built_prompt_messages: Vec<HashMap<String, String>> = Vec::new();
        let mut last_message_type = None;

        for (i, message) in messages.iter().enumerate() {
            let message_type = &message.message_type;
            // Should these checks be moved elsewhere?
            // Rule 1: System message can only be the first message
            if *message_type == PromptMessageType::System && i != 0 {
                panic!("System message can only be the first message.");
            }
            // Rule 2: First message must be either System or User
            if i == 0
                && *message_type != PromptMessageType::System
                && *message_type != PromptMessageType::User
            {
                panic!("Conversation must start with either a System or User message.");
            }
            // Rule 3: Ensure alternating User/Assistant messages after the first message
            if i > 0 {
                match (last_message_type, message_type) {
                    (Some(PromptMessageType::User), PromptMessageType::Assistant) => {},
                    (Some(PromptMessageType::Assistant), PromptMessageType::User) => {},
                    (Some(PromptMessageType::System), PromptMessageType::User) => {},
                    _ => panic!("Messages must alternate between User and Assistant after the first message (which can be System)."),
                }
            }
            last_message_type = Some(message_type.clone());

            if let Some(built_message_string) = &*message.built_prompt_message() {
                built_prompt_messages.push(HashMap::from([
                    ("role".to_string(), message.message_type.as_str().to_owned()),
                    ("content".to_string(), built_message_string.to_owned()),
                ]));
            } else {
                eprintln!("message.built_content is empty and skipped");
                continue;
                // This should be an error? Unless we're just building to display?
            }
        }

        *self.built_prompt_messages.lock().unwrap_or_else(|e| {
            panic!(
                "LlmPrompt Error - built_prompt_messages not available: {:?}",
                e
            )
        }) = Some(built_prompt_messages.clone());

        if let Some(api_prompt) = &self.api_prompt {
            api_prompt.build_prompt(&built_prompt_messages);
        };
        if let Some(local_prompt) = &self.local_prompt {
            local_prompt.build_prompt(&built_prompt_messages);
        };

        Ok(())
    }

    // Helper methods
    //

    fn messages(&self) -> MutexGuard<'_, Vec<Arc<PromptMessage>>> {
        self.messages.messages()
    }

    fn last_message(&self) -> Arc<PromptMessage> {
        self.messages()
            .last()
            .expect("LlmPrompt Error - last message not available")
            .clone()
    }

    fn built_prompt_messages(&self) -> MutexGuard<'_, Option<Vec<HashMap<String, String>>>> {
        self.built_prompt_messages.lock().unwrap_or_else(|e| {
            panic!(
                "LlmPrompt Error - built_prompt_messages not available: {:?}",
                e
            )
        })
    }
}

impl Default for LlmPrompt {
    fn default() -> Self {
        Self {
            local_prompt: None,
            api_prompt: None,
            messages: PromptMessages::default(),
            concatenator: TextConcatenator::default(),
            built_prompt_messages: Mutex::new(None),
        }
    }
}

impl Clone for LlmPrompt {
    fn clone(&self) -> Self {
        Self {
            local_prompt: self.local_prompt.clone(),
            api_prompt: self.api_prompt.clone(),
            messages: self.messages.clone(),
            concatenator: self.concatenator.clone(),
            built_prompt_messages: self.built_prompt_messages().clone().into(),
        }
    }
}

impl std::fmt::Display for LlmPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LlmPrompt")?;

        // Builds prompt if not already built, but skips the precheck.
        if self.get_built_prompt_messages().is_err() {
            match self.build_prompt() {
                Ok(_) => {}
                Err(e) => {
                    writeln!(f, "Error building prompt: {:?}", e)?;
                }
            }
        }

        // match *self.built_prompt_messages() {
        //     Some(ref prompt) => {
        //         writeln!(f, "built_prompt_messages:\n{:?}", prompt)?;
        //         writeln!(f)?;
        //     }
        //     None => writeln!(f, "built_prompt_messages: None")?,
        // };

        if let Some(local_prompt) = &self.local_prompt {
            write!(f, "{}", local_prompt)?;
        }

        if let Some(api_prompt) = &self.api_prompt {
            write!(f, "{}", api_prompt)?;
        }

        Ok(())
    }
}

impl TextConcatenatorTrait for LlmPrompt {
    fn concatenator_mut(&mut self) -> &mut TextConcatenator {
        &mut self.concatenator
    }

    fn clear_built(&self) {
        self.clear_built_prompt();
    }
}
