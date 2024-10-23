use super::{Grammar, GrammarError};

#[derive(Clone, PartialEq)]
pub struct NoneGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
}

impl Default for NoneGrammar {
    fn default() -> Self {
        Self {
            stop_word_done: None,
            stop_word_no_result: None,
        }
    }
}

impl NoneGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::NoneGrammar(self)
    }

    pub fn grammar_string(&self) -> String {
        String::new()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }
}
