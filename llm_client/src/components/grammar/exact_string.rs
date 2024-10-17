use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone, Default, PartialEq)]
pub struct ExactStringGrammar {
    pub exact_strings: Vec<String>,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    grammar_string: RefCell<Option<String>>,
}

impl ExactStringGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::ExactString(self)
    }

    pub fn add_exact_strings<T: AsRef<str>>(mut self, exact_strings: &[T]) -> Self {
        for incoming_string in exact_strings {
            if !self.exact_strings.is_empty()
                && self
                    .exact_strings
                    .iter()
                    .any(|s| s == incoming_string.as_ref())
            {
                continue;
            }
            self.exact_strings.push(incoming_string.as_ref().to_owned());
        }
        self
    }

    pub fn add_exact_string<T: AsRef<str>>(self, exact_string: T) -> Self {
        self.add_exact_strings(&[exact_string])
    }

    pub fn grammar_string(&self) -> String {
        if self.exact_strings.is_empty() {
            panic!("ExactStringGrammar must have at least one exact string");
        }
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(exact_string_grammar(
                &self.exact_strings,
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        exact_string_validate_clean(content, &self.exact_strings)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        exact_string_parse(content, &self.exact_strings)
    }
}

impl GrammarSetterTrait for ExactStringGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn exact_string_grammar<T: AsRef<str>>(
    exact_strings: &[String],
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    let mut pattern = String::new();
    for text in exact_strings {
        if pattern.is_empty() {
            pattern.push('(');
        } else {
            pattern.push('|');
        }
        pattern.push_str(&format!(" \"{}\" ", text));
    }
    pattern.push(')');
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= ( {pattern} | \"{}\" ) \" {}\"",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= ( {pattern} | \"{}\" )",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!("root ::= {pattern} \" {}\"", stop_word_done.as_ref())
        }
        (None, None) => format!("root ::= {pattern}"),
    }
}

pub fn exact_string_validate_clean(
    content: &str,
    exact_strings: &[String],
) -> Result<String, GrammarError> {
    let content = content.trim();
    if exact_string_parse(content, exact_strings).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "exact_string".to_string(),
        })
    }
}

pub fn exact_string_parse(content: &str, exact_strings: &[String]) -> Result<String, GrammarError> {
    exact_strings
        .iter()
        .find(|&text| content.contains(text))
        .map(|text| text.to_string())
        .ok_or_else(|| GrammarError::ParseValueError {
            content: format!("Content: {}, Exact Strings: {:?}", content, exact_strings),
            parse_type: "exact_string".to_string(),
        })
}
