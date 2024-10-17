use super::PrimitiveTrait;
use crate::components::grammar::{text::text_list::TextListGrammar, Grammar};
use anyhow::Result;
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

#[derive(Debug, Clone)]
pub struct TextListPrimitive {
    pub min_count: u8,
    pub max_count: u8,
    pub text_token_length: u32,
    pub item_prefix: Option<String>,
    pub disallowed_chars: Vec<char>,
}

impl Default for TextListPrimitive {
    fn default() -> Self {
        TextListPrimitive {
            min_count: 1,
            max_count: 5,
            text_token_length: 50,
            item_prefix: None,
            disallowed_chars: vec![],
        }
    }
}

impl TextListPrimitive {
    pub fn text_token_length(&mut self, text_token_length: u32) -> &mut Self {
        self.text_token_length = text_token_length;
        self
    }

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

    pub fn item_prefix<S: Into<String>>(&mut self, item_prefix: S) -> &mut Self {
        self.item_prefix = Some(item_prefix.into());
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

    fn grammar_inner(&self) -> TextListGrammar {
        Grammar::text_list()
            .item_token_length(self.text_token_length)
            .min_count(self.min_count)
            .max_count(self.max_count)
            .item_prefix_option(self.item_prefix.clone())
            .disallowed_chars(self.disallowed_chars.clone())
    }
}

pub struct TextListType(Vec<String>);

impl std::fmt::Display for TextListType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0.join(", "))
    }
}
impl IntoIterator for TextListType {
    type Item = String;
    type IntoIter = IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TextListType {
    type Item = &'a String;
    type IntoIter = Iter<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a mut TextListType {
    type Item = &'a mut String;
    type IntoIter = IterMut<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl PrimitiveTrait for TextListPrimitive {
    type PrimitiveResult = TextListType;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "list items or 'None.'"
        } else {
            "list items"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            format!(
                "between {}-{} list items, or, possibly, 'None.'",
                self.min_count, self.max_count
            )
        } else {
            format!("between {}-{} list items", self.min_count, self.max_count)
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
        let parsed: Self::PrimitiveResult =
            TextListType(self.grammar_inner().grammar_parse(content)?);
        Ok(parsed)
    }
}
