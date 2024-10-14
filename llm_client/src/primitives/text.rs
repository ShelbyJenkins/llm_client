use super::PrimitiveTrait;
use crate::components::grammar::{Grammar, TextGrammar};
use anyhow::Result;
pub struct TextPrimitive {
    pub text_token_length: u32,
    pub disallowed_chars: Vec<char>,
    pub allow_newline: bool,
}

impl Default for TextPrimitive {
    fn default() -> Self {
        TextPrimitive {
            text_token_length: 200,
            disallowed_chars: vec![],
            allow_newline: false,
        }
    }
}

impl TextPrimitive {
    pub fn text_token_length(&mut self, text_token_length: u32) -> &mut Self {
        self.text_token_length = text_token_length;
        self
    }

    pub fn disallowed_char(&mut self, disallowed_char: char) -> &mut Self {
        self.disallowed_chars.push(disallowed_char);
        self
    }

    pub fn disallowed_chars(&mut self, disallowed_chars: Vec<char>) -> &mut Self {
        self.disallowed_chars.extend(disallowed_chars);
        self
    }

    pub fn allow_newline(&mut self, allow_newline: bool) -> &mut Self {
        self.allow_newline = allow_newline;
        self
    }

    fn grammar_inner(&self) -> TextGrammar {
        Grammar::text()
            .item_token_length(self.text_token_length)
            .disallowed_chars(self.disallowed_chars.clone())
            .allow_newline(self.allow_newline)
    }
}

impl PrimitiveTrait for TextPrimitive {
    type PrimitiveResult = String;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "text or 'None.'"
        } else {
            "text"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            "text or 'None.'".to_string()
        } else {
            "text".to_string()
        }
    }

    fn stop_word_result_is_none(&self, result_can_be_none: bool) -> Option<String> {
        if result_can_be_none {
            Some("None.".to_string())
        } else {
            None
        }
    }

    fn grammar(&self) -> Grammar {
        self.grammar_inner().wrap()
    }

    fn parse_to_primitive(&self, content: &str) -> Result<Self::PrimitiveResult> {
        let parsed: Self::PrimitiveResult = self.grammar_inner().grammar_parse(content)?;
        Ok(parsed)
    }
}
