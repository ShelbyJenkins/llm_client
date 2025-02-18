use super::{Grammar, GrammarError, GrammarSetterTrait};

#[derive(Clone, PartialEq, Debug)]
pub struct CustomGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub custom_grammar: Option<String>,
}

impl Default for CustomGrammar {
    fn default() -> Self {
        Self {
            stop_word_done: None,
            stop_word_no_result: None,
            custom_grammar: None,
        }
    }
}

impl CustomGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Custom(self)
    }

    pub fn custom_grammar(mut self, custom_grammar: String) -> Self {
        self.custom_grammar = Some(custom_grammar);
        self
    }

    pub fn grammar_string(&self) -> String {
        self.custom_grammar.clone().expect("custom_grammar not set")
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        Ok(content.to_owned())
    }
}

impl GrammarSetterTrait for CustomGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}
