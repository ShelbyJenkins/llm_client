use super::{PromptMessage, TextConcatenator};
use crate::PromptTokenizer;
use minijinja::value::{from_args, Value, ValueKind};
use minijinja::{context, Environment, Error, ErrorKind};
use std::collections::HashMap;

#[derive(Clone)]
pub struct ChatTemplatePrompt {
    pub built_prompt_string: std::cell::RefCell<Option<String>>,
    pub built_prompt_as_tokens: std::cell::RefCell<Option<Vec<u32>>>,
    pub total_prompt_tokens: std::cell::RefCell<Option<u64>>,
    pub concatenator: TextConcatenator,
    pub messages: std::cell::RefCell<Vec<PromptMessage>>,
    pub generation_prefix: std::cell::RefCell<Option<String>>,
    pub tokenizer: std::sync::Arc<dyn PromptTokenizer>,
    chat_template: String,
    bos_token: Option<String>,
    eos_token: String,
    unk_token: Option<String>,
    base_generation_prefix: Option<String>,
}

impl ChatTemplatePrompt {
    pub fn new(
        chat_template: &str,
        bos_token: Option<&str>,
        eos_token: &str,
        unk_token: Option<&str>,
        base_generation_prefix: Option<&str>,
        tokenizer: std::sync::Arc<dyn PromptTokenizer>,
    ) -> Self {
        Self {
            built_prompt_string: std::cell::RefCell::new(None),
            built_prompt_as_tokens: std::cell::RefCell::new(None),
            total_prompt_tokens: std::cell::RefCell::new(None),
            concatenator: TextConcatenator::default(),
            messages: std::cell::RefCell::new(Vec::new()),
            generation_prefix: std::cell::RefCell::new(None),
            tokenizer,
            chat_template: chat_template.to_owned(),
            bos_token: bos_token.map(|s| s.to_owned()),
            eos_token: eos_token.to_owned(),
            unk_token: unk_token.map(|s| s.to_owned()),
            base_generation_prefix: base_generation_prefix.map(|s| s.to_owned()),
        }
    }

    // Setter functions
    pub fn set_generation_prefix<T: AsRef<str>>(&self, generation_prefix: T) {
        if self.generation_prefix.borrow().is_none()
            || self.generation_prefix.borrow().as_deref() != Some(generation_prefix.as_ref())
        {
            self.clear_built_prompt();
            *self.generation_prefix.borrow_mut() = Some(generation_prefix.as_ref().to_string());
        }
    }

    pub fn clear_generation_prefix(&self) {
        self.clear_built_prompt();
        *self.generation_prefix.borrow_mut() = None;
    }

    pub fn build_prompt(&self) -> String {
        self.clear_built_prompt();
        let prompt_messages =
            super::prompt_message::build_messages(&mut self.messages.borrow_mut());

        let mut built_prompt_string = apply_chat_template(
            &prompt_messages,
            &self.chat_template,
            self.bos_token.as_deref(),
            &self.eos_token,
            self.unk_token.as_deref(),
        );

        if let Some(ref generation_prefix) = *self.generation_prefix.borrow() {
            if let Some(base_generation_prefix) = &self.base_generation_prefix {
                built_prompt_string.push_str(base_generation_prefix);
            }
            built_prompt_string.push_str(generation_prefix);
        }

        let built_prompt_as_tokens = self.tokenizer.tokenize(&built_prompt_string);
        *self.total_prompt_tokens.borrow_mut() = Some(built_prompt_as_tokens.len() as u64);
        *self.built_prompt_as_tokens.borrow_mut() = Some(built_prompt_as_tokens);
        *self.built_prompt_string.borrow_mut() = Some(built_prompt_string.clone());
        built_prompt_string
    }

    pub fn clear_built_prompt(&self) {
        *self.built_prompt_string.borrow_mut() = None;
        *self.built_prompt_as_tokens.borrow_mut() = None;
        *self.total_prompt_tokens.borrow_mut() = None;
    }
}

impl std::fmt::Display for ChatTemplatePrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ChatTemplatePrompt")?;
        // for message in self.messages.borrow().iter() {
        //     writeln!(f, "{}", message)?;
        // }

        match *self.built_prompt_string.borrow() {
            Some(ref prompt) => {
                writeln!(f, "built_prompt_string:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "built_prompt_string: None")?,
        };

        match *self.total_prompt_tokens.borrow() {
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
