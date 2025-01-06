use super::PrimitiveTrait;
use crate::components::grammar::{ExactStringGrammar, Grammar};
use crate::workflows::reason::ReasonTrait;
use anyhow::Result;

#[derive(Default, Debug, Clone)]
pub struct ExactStringPrimitive {
    pub allowed_strings: Vec<String>,
}

impl ExactStringPrimitive {
    pub fn add_strings_to_allowed<T: AsRef<str>>(&mut self, words: &[T]) -> &mut Self {
        words.iter().for_each(|word| {
            self.add_string_to_allowed(word);
        });
        self
    }

    pub fn add_string_to_allowed<T: AsRef<str>>(&mut self, word: T) -> &mut Self {
        if !self.allowed_strings.is_empty()
            && self
                .allowed_strings
                .iter()
                .any(|text| text == word.as_ref())
        {
            return self;
        }
        self.allowed_strings.push(word.as_ref().to_owned());
        self
    }

    pub fn remove_string_from_allowed<T: AsRef<str>>(&mut self, word: T) -> &mut Self {
        self.allowed_strings.retain(|w| w != word.as_ref());
        self
    }

    fn grammar_inner(&self) -> ExactStringGrammar {
        Grammar::exact_string().add_exact_strings(&self.allowed_strings)
    }
}

impl PrimitiveTrait for ExactStringPrimitive {
    type PrimitiveResult = String;

    fn clear_primitive(&mut self) {
        self.allowed_strings.clear();
    }

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "string or 'None of the above.'"
        } else {
            "string"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            format!(
                "one of the the following strings: {}, or, possibly, 'None of the above.'",
                self.allowed_strings.join(", ")
            )
        } else {
            format!(
                "one of the the following strings: {}",
                self.allowed_strings.join(", ")
            )
        }
    }

    fn stop_word_result_is_none(&self, result_can_be_none: bool) -> Option<String> {
        if result_can_be_none {
            Some("None of the above.".to_string())
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

impl ReasonTrait for ExactStringPrimitive {
    fn primitive_to_result_index(&self, content: &str) -> u32 {
        let output = self.parse_to_primitive(content).unwrap();
        if let Some(index) = self.allowed_strings.iter().position(|s| s == &output) {
            index as u32
        } else {
            panic!("This shouldn't happen.")
        }
    }

    fn result_index_to_primitive(&self, result_index: Option<u32>) -> Result<Option<String>> {
        if let Some(result_index) = result_index {
            if let Some(result) = self.allowed_strings.get(result_index as usize) {
                Ok(Some(result.clone()))
            } else {
                panic!("This shouldn't happen.")
            }
        } else {
            Ok(None)
        }
    }
}
