use serde::{Deserialize, Serialize};

use super::TextConcatenator;
use std::sync::{Arc, Mutex, MutexGuard};

/// Represents the type of message in a prompt sequence.
///
/// Message types follow standard LLM conventions, supporting system-level instructions,
/// user inputs, and assistant responses. The ordering and placement of these types
/// is validated during prompt construction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PromptMessageType {
    /// A system-level instruction that guides model behavior.
    /// Must be the first message if present.
    System,

    /// A user input message. Cannot follow another user message directly.
    User,

    /// A response from the assistant. Cannot be the first message or
    /// follow another assistant message.
    Assistant,
}

impl PromptMessageType {
    pub fn as_str(&self) -> &str {
        match self {
            PromptMessageType::System => "system",
            PromptMessageType::User => "user",
            PromptMessageType::Assistant => "assistant",
        }
    }
}

/// A collection of prompt messages with thread-safe mutability.
///
/// Manages an ordered sequence of messages while ensuring thread safety through
/// mutex protection. The collection maintains message ordering rules and
/// provides access to the underlying messages.
#[derive(Serialize, Deserialize, Default, Debug)]
pub struct PromptMessages(Mutex<Vec<Arc<PromptMessage>>>);

impl PromptMessages {
    pub(crate) fn messages(&self) -> MutexGuard<'_, Vec<Arc<PromptMessage>>> {
        self.0
            .lock()
            .unwrap_or_else(|e| panic!("PromptMessages Error - messages not available: {:?}", e))
    }
}

impl Clone for PromptMessages {
    fn clone(&self) -> Self {
        let cloned_messages: Vec<Arc<PromptMessage>> = self
            .messages()
            .iter()
            .map(|message| Arc::new((**message).clone()))
            .collect();
        Self(cloned_messages.into())
    }
}

impl std::fmt::Display for PromptMessages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let messages = self.messages();
        for message in messages.iter() {
            writeln!(f, "{}", message)?;
        }
        Ok(())
    }
}

/// An individual message within a prompt sequence.
///
/// Represents a single message with its content, type, and concatenation rules.
/// Maintains thread-safe interior mutability for content manipulation and
/// provides methods for building and accessing the message content.
#[derive(Serialize, Deserialize, Debug)]
pub struct PromptMessage {
    pub content: Mutex<Vec<String>>,
    pub built_prompt_message: Mutex<Option<String>>,
    pub message_type: PromptMessageType,
    pub concatenator: TextConcatenator,
}

impl PromptMessage {
    pub fn new(message_type: PromptMessageType, concatenator: &TextConcatenator) -> Self {
        Self {
            content: Vec::new().into(),
            built_prompt_message: None.into(),
            message_type,
            concatenator: concatenator.clone(),
        }
    }

    // Setter methods
    //

    /// Sets the primary content of the message, replacing any existing content.
    ///
    /// If the provided content is empty, the message remains unchanged. Otherwise,
    /// replaces all existing content with the new content and rebuilds the message.
    ///
    /// # Arguments
    ///
    /// * `content` - The new content to set for the message
    ///
    /// # Returns
    ///
    /// A reference to self for method chaining
    pub fn set_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        let mut content_guard = self.content();
        let should_update = content_guard
            .first()
            .map_or(true, |first| first != content.as_ref());

        if should_update {
            *content_guard = vec![content.as_ref().to_owned()];
            self.build(content_guard);
        }

        self
    }

    /// Adds content to the beginning of the message.
    ///
    /// If the provided content is empty, the message remains unchanged. Otherwise,
    /// inserts the new content at the start of the existing content and rebuilds
    /// the message.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to prepend to the message
    ///
    /// # Returns
    ///
    /// A reference to self for method chaining
    pub fn prepend_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        let mut content_guard = self.content();
        let should_update = content_guard
            .first()
            .map_or(true, |first| first != content.as_ref());

        if should_update {
            content_guard.insert(0, content.as_ref().to_owned());
            self.build(content_guard);
        }

        self
    }

    /// Adds content to the end of the message.
    ///
    /// If the provided content is empty, the message remains unchanged. Otherwise,
    /// adds the new content after existing content and rebuilds the message.
    ///
    /// # Arguments
    ///
    /// * `content` - The content to append to the message
    ///
    /// # Returns
    ///
    /// A reference to self for method chaining
    pub fn append_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        let mut content_guard = self.content();
        let should_update = content_guard
            .last()
            .map_or(true, |last| last != content.as_ref());

        if should_update {
            content_guard.push(content.as_ref().to_owned());
            self.build(content_guard);
        }

        self
    }

    // Getter methods
    //

    /// Retrieves the built message content.
    ///
    /// Returns the complete message content with all parts properly concatenated
    /// according to the message's concatenation rules.
    ///
    /// # Returns
    ///
    /// Returns `Ok(String)` containing the built message content.
    ///
    /// # Errors
    ///
    /// Returns an error if the message has not been built yet.
    pub fn get_built_prompt_message(&self) -> Result<String, crate::Error> {
        match &*self.built_prompt_message() {
            Some(prompt) => Ok(prompt.clone()),
            None => crate::bail!(
                " PromptMessage Error - built_prompt_string not available - message not built"
            ),
        }
    }

    // Builder methods
    //

    fn build(&self, content_guard: MutexGuard<'_, Vec<String>>) {
        let mut built_prompt_message = String::new();

        for c in content_guard.iter() {
            if !built_prompt_message.is_empty() {
                built_prompt_message.push_str(self.concatenator.as_str());
            }
            built_prompt_message.push_str(c.as_str());
        }

        *self.built_prompt_message() = Some(built_prompt_message);
    }

    // Helper methods
    //

    fn content(&self) -> MutexGuard<'_, Vec<String>> {
        self.content
            .lock()
            .unwrap_or_else(|e| panic!("PromptMessage Error - content not available: {:?}", e))
    }

    pub(crate) fn built_prompt_message(&self) -> MutexGuard<'_, Option<String>> {
        self.built_prompt_message.lock().unwrap_or_else(|e| {
            panic!(
                "PromptMessage Error - built_prompt_message not available: {:?}",
                e
            )
        })
    }
}

impl Clone for PromptMessage {
    fn clone(&self) -> Self {
        Self {
            content: self.content().clone().into(),
            built_prompt_message: self.built_prompt_message().clone().into(),
            message_type: self.message_type.clone(),
            concatenator: self.concatenator.clone(),
        }
    }
}

impl std::fmt::Display for PromptMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message_type = match self.message_type {
            PromptMessageType::System => "System",
            PromptMessageType::User => "User",
            PromptMessageType::Assistant => "Assistant",
        };
        let message = match &*self.built_prompt_message() {
            Some(built_message_string) => {
                if built_message_string.len() > 300 {
                    format!(
                        "{}...",
                        built_message_string.chars().take(300).collect::<String>()
                    )
                } else {
                    built_message_string.clone()
                }
            }
            None => "debug message: empty or unbuilt".to_owned(),
        };

        writeln!(f, "\x1b[1m{message_type}\x1b[0m:\n{:?}", message)
    }
}
