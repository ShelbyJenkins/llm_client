use super::{PromptMessage, TextConcatenator};
use crate::PromptTokenizer;
use std::collections::HashMap;

#[derive(Clone)]
pub struct OpenAiPrompt {
    pub built_prompt_hashmap: std::cell::RefCell<Option<Vec<HashMap<String, String>>>>,
    pub total_prompt_tokens: std::cell::RefCell<Option<u64>>,
    pub concatenator: TextConcatenator,
    pub messages: std::cell::RefCell<Vec<PromptMessage>>,
    tokenizer: std::sync::Arc<dyn PromptTokenizer>,
    tokens_per_message: Option<u32>,
    tokens_per_name: Option<i32>,
}

impl OpenAiPrompt {
    pub fn new(
        tokens_per_message: Option<u32>,
        tokens_per_name: Option<i32>,
        tokenizer: std::sync::Arc<dyn PromptTokenizer>,
    ) -> Self {
        Self {
            built_prompt_hashmap: std::cell::RefCell::new(None),
            total_prompt_tokens: std::cell::RefCell::new(None),
            concatenator: TextConcatenator::default(),
            messages: std::cell::RefCell::new(Vec::new()),
            tokenizer,
            tokens_per_message,
            tokens_per_name,
        }
    }

    pub fn build_prompt(&self) -> Vec<HashMap<String, String>> {
        self.clear_built_prompt();
        let built_prompt_hashmap =
            super::prompt_message::build_messages(&mut self.messages.borrow_mut());
        *self.total_prompt_tokens.borrow_mut() =
            Some(super::token_count::total_prompt_tokens_openai_format(
                &built_prompt_hashmap,
                self.tokens_per_message,
                self.tokens_per_name,
                &self.tokenizer,
            ));
        *self.built_prompt_hashmap.borrow_mut() = Some(built_prompt_hashmap.clone());
        built_prompt_hashmap
    }

    pub fn clear_built_prompt(&self) {
        *self.built_prompt_hashmap.borrow_mut() = None;
        *self.total_prompt_tokens.borrow_mut() = None;
    }
}

impl std::fmt::Display for OpenAiPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "OpenAiPrompt")?;
        for message in self.messages.borrow().iter() {
            writeln!(f, "{}", message)?;
        }

        match *self.built_prompt_hashmap.borrow() {
            Some(ref prompt) => {
                writeln!(f, "built_prompt_hashmap:\n{:?}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "built_prompt_hashmap: None")?,
        };

        match *self.total_prompt_tokens.borrow() {
            Some(ref prompt) => {
                writeln!(f, "total_prompt_tokens:\n\n{}", prompt)?;
                writeln!(f)?;
            }
            None => writeln!(f, "total_prompt_tokens: None")?,
        };

        Ok(())
    }
}
