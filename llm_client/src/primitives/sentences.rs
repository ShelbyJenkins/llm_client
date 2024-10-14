use super::PrimitiveTrait;
use crate::components::grammar::{Grammar, SentencesGrammar};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct SentencesPrimitive {
    pub min_count: u8,
    pub max_count: u8,
    pub capitalize_first: bool,
    pub concatenator: String,
    pub disallowed_chars: Vec<char>,
}

impl Default for SentencesPrimitive {
    fn default() -> Self {
        SentencesPrimitive {
            min_count: 1,
            max_count: 1,
            capitalize_first: true,
            concatenator: " ".to_string(),
            disallowed_chars: vec![],
        }
    }
}

impl SentencesPrimitive {
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

    pub fn capitalize_first(&mut self, capitalize_first: bool) -> &mut Self {
        self.capitalize_first = capitalize_first;
        self
    }

    pub fn concatenator(&mut self, concatenator: &str) -> &mut Self {
        self.concatenator = concatenator.to_string();
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

    fn grammar_inner(&self) -> SentencesGrammar {
        Grammar::sentences()
            .min_count(self.min_count)
            .max_count(self.max_count)
            .capitalize_first(self.capitalize_first)
            .concatenator(&self.concatenator)
            .disallowed_chars(self.disallowed_chars.clone())
    }
}

impl PrimitiveTrait for SentencesPrimitive {
    type PrimitiveResult = String;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "sentences or 'None.'"
        } else {
            "sentences"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            format!(
                "between {}-{} sentences, or, possibly, 'None.'",
                self.min_count, self.max_count
            )
        } else {
            format!("between {}-{} sentences", self.min_count, self.max_count)
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
