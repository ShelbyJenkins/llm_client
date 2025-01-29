use crate::PromptTokenizer;
use minijinja::value::{from_args, Value, ValueKind};
use minijinja::{context, Environment, Error, ErrorKind};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::{Arc, MutexGuard};

/// A prompt formatter for local LLMs that use chat templates.
///
/// `LocalPrompt` handles formatting messages according to a model's chat template,
/// managing special tokens (BOS, EOS, UNK), and supporting generation prefixes.
/// Unlike API prompts, local prompts need to handle the specific formatting requirements
/// and token conventions of locally-run models.
///
/// The struct maintains both string and tokenized representations of the built prompt,
/// along with thread-safe interior mutability for managing prompt state. It supports
/// token counting and generation prefix management for model outputs.
#[derive(Serialize)]
pub struct LocalPrompt {
    // Skip the tokenizer field
    #[serde(skip)]
    tokenizer: Arc<dyn PromptTokenizer>,
    chat_template: String,
    bos_token: Option<String>,
    eos_token: String,
    unk_token: Option<String>,
    base_generation_prefix: Option<String>,
    pub generation_prefix: Mutex<Option<String>>,
    pub built_prompt_string: Mutex<Option<String>>,
    pub built_prompt_as_tokens: Mutex<Option<Vec<u32>>>,
    pub total_prompt_tokens: Mutex<Option<u64>>,
}

impl LocalPrompt {
    pub(crate) fn new(
        tokenizer: Arc<dyn PromptTokenizer>,
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: &str,
        unk_token: Option<&str>,
        base_generation_prefix: Option<&str>,
    ) -> Self {
        Self {
            tokenizer,
            chat_template: chat_template.to_owned(),
            bos_token: bos_token.map(|s| s.to_owned()),
            eos_token: eos_token.to_owned(),
            unk_token: unk_token.map(|s| s.to_owned()),
            base_generation_prefix: base_generation_prefix.map(|s| s.to_owned()),
            generation_prefix: None.into(),
            built_prompt_string: None.into(),
            built_prompt_as_tokens: None.into(),
            total_prompt_tokens: None.into(),
        }
    }

    // Setter methods
    //

    pub(crate) fn set_generation_prefix<T: AsRef<str>>(&self, generation_prefix: T) {
        let mut self_generation_prefix = self.generation_prefix();
        if self_generation_prefix.is_none()
            || self_generation_prefix.as_deref() != Some(generation_prefix.as_ref())
        {
            *self_generation_prefix = Some(generation_prefix.as_ref().to_string());
        }
    }

    pub(crate) fn clear_generation_prefix(&self) {
        *self.generation_prefix() = None;
    }

    pub(crate) fn clear_built_prompt(&self) {
        *self.built_prompt_string() = None;
        *self.built_prompt_as_tokens() = None;
        *self.total_prompt_tokens() = None;
    }

    // Getter methods
    //

    /// Retrieves the built prompt as a formatted string.
    ///
    /// Returns the complete prompt string with all messages formatted according to
    /// the chat template, including any special tokens and generation prefix.
    ///
    /// # Returns
    ///
    /// Returns `Ok(String)` containing the formatted prompt string.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt has not been built yet.
    pub fn get_built_prompt(&self) -> Result<String, crate::Error> {
        match &*self.built_prompt_string() {
            Some(prompt) => Ok(prompt.clone()),
            None => crate::bail!(
                "LocalPrompt Error - built_prompt_string not available - prompt not built"
            ),
        }
    }

    /// Retrieves the built prompt as a vector of tokens.
    ///
    /// Returns the complete prompt converted to model tokens using the configured
    /// tokenizer. This is useful for operations that need to work directly with
    /// token IDs rather than text.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<u32>)` containing the token IDs for the prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt has not been built yet.
    pub fn get_built_prompt_as_tokens(&self) -> Result<Vec<u32>, crate::Error> {
        match &*self.built_prompt_as_tokens() {
            Some(prompt) => Ok(prompt.clone()),
            None => crate::bail!(
                "LocalPrompt Error - built_prompt_as_tokens not available - prompt not built"
            ),
        }
    }

    /// Gets the total number of tokens in the built prompt.
    ///
    /// Returns the exact token count of the built prompt, which is useful for
    /// ensuring prompts stay within model context limits. This count reflects
    /// all content, special tokens, and any generation prefix.
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
                "LocalPrompt Error - total_prompt_tokens not available - prompt not built"
            ),
        }
    }

    // Builder methods
    //

    pub(crate) fn build_prompt(&self, built_prompt_messages: &Vec<HashMap<String, String>>) {
        let mut built_prompt_string = apply_chat_template(
            built_prompt_messages,
            &self.chat_template,
            self.bos_token.as_deref(),
            &self.eos_token,
            self.unk_token.as_deref(),
        );

        {
            if let Some(generation_prefix) = &*self.generation_prefix() {
                if let Some(base_generation_prefix) = &self.base_generation_prefix {
                    built_prompt_string.push_str(base_generation_prefix);
                }
                built_prompt_string.push_str(generation_prefix);
            }
        }

        let built_prompt_as_tokens = self.tokenizer.tokenize(&built_prompt_string);
        *self.total_prompt_tokens() = Some(built_prompt_as_tokens.len() as u64);
        *self.built_prompt_as_tokens() = Some(built_prompt_as_tokens);
        *self.built_prompt_string() = Some(built_prompt_string);
    }

    // Helper methods
    //

    fn generation_prefix(&self) -> MutexGuard<'_, Option<String>> {
        self.generation_prefix.lock().unwrap_or_else(|e| {
            panic!(
                "LocalPrompt Error - generation_prefix not available: {:?}",
                e
            )
        })
    }

    fn built_prompt_string(&self) -> MutexGuard<'_, Option<String>> {
        self.built_prompt_string.lock().unwrap_or_else(|e| {
            panic!(
                "LocalPrompt Error - built_prompt_string not available: {:?}",
                e
            )
        })
    }

    fn built_prompt_as_tokens(&self) -> MutexGuard<'_, Option<Vec<u32>>> {
        self.built_prompt_as_tokens.lock().unwrap_or_else(|e| {
            panic!(
                "LocalPrompt Error - built_prompt_as_tokens not available: {:?}",
                e
            )
        })
    }

    fn total_prompt_tokens(&self) -> MutexGuard<'_, Option<u64>> {
        self.total_prompt_tokens.lock().unwrap_or_else(|e| {
            panic!(
                "LocalPrompt Error - total_prompt_tokens not available: {:?}",
                e
            )
        })
    }
}

impl Clone for LocalPrompt {
    fn clone(&self) -> Self {
        Self {
            built_prompt_string: self.built_prompt_string().clone().into(),
            built_prompt_as_tokens: self.built_prompt_as_tokens().clone().into(),
            total_prompt_tokens: self.total_prompt_tokens().clone().into(),
            generation_prefix: self.generation_prefix().clone().into(),
            tokenizer: self.tokenizer.clone(),
            chat_template: self.chat_template.clone(),
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            unk_token: self.unk_token.clone(),
            base_generation_prefix: self.base_generation_prefix.clone(),
        }
    }
}

impl std::fmt::Display for LocalPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LocalPrompt")?;

        match *self.built_prompt_string() {
            Some(ref prompt) => {
                writeln!(f, "built_prompt_string:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "built_prompt_string: None")?,
        };

        match *self.total_prompt_tokens() {
            Some(ref prompt) => {
                writeln!(f, "total_prompt_tokens: {}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "total_prompt_tokens: None")?,
        };

        Ok(())
    }
}

/// Applies a chat template to a message, given a message and a chat template.
///
/// # Arguments
///
/// * `message` - The message as a HashMap.
/// * `chat_template` - The chat template as a String.
///
/// # Returns
///
/// The formatted message as a String.
pub fn apply_chat_template(
    messages: &Vec<HashMap<String, String>>,
    chat_template: &str,
    bos_token: Option<&str>,
    eos_token: &str,
    unk_token: Option<&str>,
) -> String {
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_template("chat_template", chat_template)
        .expect("Failed to add template");
    env.add_function("raise_exception", raise_exception);

    env.set_unknown_method_callback(|state, value, method, args| match (value.kind(), method) {
        (ValueKind::String, "strip") => {
            let _: () = from_args(args)?;
            Ok(Value::from(value.as_str().unwrap_or("").trim()))
        }
        (ValueKind::Map, "items") => {
            let _: () = from_args(args)?;
            state.apply_filter("items", &[value.clone()])
        }
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("object has no method named {}", method),
        )),
    });

    let tmpl = env
        .get_template("chat_template")
        .expect("Failed to get template");

    let unk_token = unk_token.unwrap_or("");
    let bos_token = bos_token.unwrap_or("");

    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => bos_token,
        eos_token => eos_token,
        unk_token => unk_token,
    })
    .expect("Failed to render template without system prompt")
}

/// This exists specifically for the minijinja template engine to raise an exception.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}
