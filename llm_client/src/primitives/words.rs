use super::PrimitiveTrait;
use crate::components::grammar::{Grammar, WordsGrammar};
use anyhow::Result;
pub struct WordsPrimitive {
    pub min_count: u8,
    pub max_count: u8,
    pub word_char_length: u8,
    pub concatenator: String,
}

impl Default for WordsPrimitive {
    fn default() -> Self {
        WordsPrimitive {
            min_count: 1,
            max_count: 3,
            word_char_length: 12,
            concatenator: " ".to_string(),
        }
    }
}

impl WordsPrimitive {
    /// Set the lower bound of the integer range. Default is 0.
    pub fn min_count(&mut self, min_count: u8) -> &mut Self {
        if self.min_count != min_count {
            self.min_count = min_count;
        }
        self
    }

    /// Set the upper bound of the integer range. Default is 9.
    pub fn max_count(&mut self, max_count: u8) -> &mut Self {
        if self.max_count != max_count {
            self.max_count = max_count;
        }
        self
    }

    pub fn word_char_length(&mut self, word_char_length: u8) -> &mut Self {
        self.word_char_length = word_char_length;
        self
    }

    pub fn concatenator(&mut self, concatenator: &str) -> &mut Self {
        self.concatenator = concatenator.to_string();
        self
    }

    fn grammar_inner(&self) -> WordsGrammar {
        Grammar::words()
            .min_count(self.min_count)
            .max_count(self.max_count)
            .word_char_length(self.word_char_length)
            .concatenator(&self.concatenator)
    }
}

impl PrimitiveTrait for WordsPrimitive {
    type PrimitiveResult = String;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "words or 'None.'"
        } else {
            "words"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            format!(
                "between {}-{} words, or, possibly, 'None.'",
                self.min_count, self.max_count
            )
        } else {
            format!("between {}-{} words", self.min_count, self.max_count)
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
