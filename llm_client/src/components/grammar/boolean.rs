use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone, Default, PartialEq)]
pub struct BooleanGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    grammar_string: RefCell<Option<String>>,
}

impl BooleanGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Boolean(self)
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(boolean_grammar(
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        boolean_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<bool, GrammarError> {
        boolean_parse(content)
    }
}

impl GrammarSetterTrait for BooleanGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn boolean_grammar<T: AsRef<str>>(
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= \" \" ( \"true\" | \"false\" | \"{}\" ) \" {}\"",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= \" \" ( \"true\" | \"false\" | \"{}\" ) ",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" ( \"true\" | \"false\" ) \" {}\"",
                stop_word_done.as_ref()
            )
        }
        (None, None) => "root ::= \" \" ( \"true\" | \"false\" )".to_owned(),
    }
}

pub fn boolean_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content = content.trim();
    if boolean_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "boolean".to_string(),
        })
    }
}

pub fn boolean_parse(content: &str) -> Result<bool, GrammarError> {
    content
        .trim()
        .parse::<bool>()
        .map_err(|_| GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "boolean".to_string(),
        })
}
