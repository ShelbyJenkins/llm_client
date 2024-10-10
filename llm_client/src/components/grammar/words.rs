use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone, Default)]
pub struct WordsGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub word_char_length: u8,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub concatenator: String,
    grammar_string: RefCell<Option<String>>,
}

impl WordsGrammar {
    pub fn new() -> Self {
        Self {
            min_count: 1,
            max_count: 3,
            word_char_length: 12,
            stop_word_done: None,
            stop_word_no_result: None,
            concatenator: " ".to_string(),
            grammar_string: RefCell::new(None),
        }
    }
}

impl WordsGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Words(self)
    }

    pub fn min_count(mut self, min_count: u8) -> Self {
        self.min_count = min_count;
        self
    }

    pub fn max_count(mut self, max_count: u8) -> Self {
        self.max_count = max_count;
        self
    }

    pub fn word_char_length(mut self, word_char_length: u8) -> Self {
        self.word_char_length = word_char_length;
        self
    }

    pub fn concatenator(mut self, concatenator: &str) -> Self {
        self.concatenator = concatenator.to_string();
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(words_grammar(
                self.min_count,
                self.max_count,
                self.word_char_length,
                &self.concatenator,
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        words_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        words_parse(content)
    }
}

impl GrammarSetterTrait for WordsGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn words_grammar<T: AsRef<str>>(
    min_count: u8,
    max_count: u8,
    word_char_length: u8,
    concatenator: &str,
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    let range = create_range(min_count, max_count, stop_word_done);
    let item = format!("item ::= [a-z]{{1,{word_char_length}}} \"{concatenator}\"",);
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= \" \" ( {range} | \"{}\" ) \" {}\"\n{item}",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= \" \" ( {range} | \"{}\" )\n{item}",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" {range} \" {}\"\n{item}",
                stop_word_done.as_ref()
            )
        }
        (None, None) => format!("root ::= \" \" {range}\n{item}"),
    }
}

fn create_range<T: AsRef<str>>(min_count: u8, max_count: u8, stop_word_done: &Option<T>) -> String {
    let max_count = match max_count.cmp(&min_count) {
        std::cmp::Ordering::Less => {
            eprintln!("Max count must be greater than or equal to min count. Setting max count to min count.");
            min_count
        }
        _ => max_count,
    };
    if min_count == 0 && max_count == 0 {
        "item{0,1}".to_owned()
    } else {
        let mut range = String::new();
        if min_count > 0 {
            range.push_str(&format!("item{{{min_count}}} "));
        }
        if max_count > min_count {
            let opt_count = max_count - min_count;
            if let Some(stop_word_done) = stop_word_done {
                range.push_str(&format!(
                    "( item | \"{}\" ){{0,{opt_count}}}",
                    stop_word_done.as_ref()
                ))
            } else {
                range.push_str(&format!("item{{0,{opt_count}}}"));
            };
        }
        range
    }
}

pub fn words_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches(|c: char| !(c.is_alphanumeric() || c.is_ascii_punctuation()));

    if words_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        })
    }
}

pub fn words_parse(content: &str) -> Result<String, GrammarError> {
    if content.is_empty() {
        return Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        });
    }
    Ok(content.to_string())
}
