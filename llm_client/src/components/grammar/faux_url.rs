use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone)]
pub struct FauxUrlGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub word_char_length: u8,
    pub base_url: String,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    grammar_string: RefCell<Option<String>>,
}

impl Default for FauxUrlGrammar {
    fn default() -> Self {
        Self {
            min_count: 1,
            max_count: 3,
            word_char_length: 12,
            base_url: "https://example.com/".to_string(),
            stop_word_done: None,
            stop_word_no_result: None,
            grammar_string: RefCell::new(None),
        }
    }
}

impl FauxUrlGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::FauxUrl(self)
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

    pub fn base_url<T: AsRef<str>>(mut self, base_url: T) -> Self {
        self.base_url = base_url.as_ref().to_string();
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(faux_url_grammar(
                self.min_count,
                self.max_count,
                self.word_char_length,
                &self.base_url,
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        faux_url_validate_clean(content, &self.base_url)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<Vec<String>, GrammarError> {
        faux_url_parse(content, &self.base_url)
    }
}

impl GrammarSetterTrait for FauxUrlGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn faux_url_grammar<T: AsRef<str>>(
    min_count: u8,
    max_count: u8,
    word_char_length: u8,
    base_url: &str,
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    let range = create_range(min_count, max_count, stop_word_done);
    let first = format!("first ::= [a-z]{{3,{word_char_length}}}");
    let item = format!("item ::= \"-\" [a-z]{{3,{word_char_length}}}");
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= \" \" ( \"{base_url}\" {range} | \"{}\" ) \" {}\"\n{item}\n{first}",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= \" \" ( \"{base_url}\" {range} | \"{}\" )\n{item}\n{first}",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" \"{base_url}\" {range} \" {}\"\n{item}\n{first}",
                stop_word_done.as_ref()
            )
        }
        (None, None) => format!("root ::= \" \" \"{base_url}\" {range}\n{item}\n{first}"),
    }
}

fn create_range<T: AsRef<str>>(min_count: u8, max_count: u8, stop_word_done: &Option<T>) -> String {
    let min_count = if min_count == 0 {
        eprintln!("Min count must be greater than 0. Setting min count to 1.");
        1
    } else {
        min_count
    };
    let max_count = match max_count.cmp(&min_count) {
        std::cmp::Ordering::Less => {
            eprintln!("Max count must be greater than or equal to min count. Setting max count to min count.");
            min_count
        }
        _ => max_count,
    };
    if min_count == 1 && max_count == 1 {
        "first".to_owned()
    } else {
        let mut range = String::new();
        range.push_str("first ");
        if min_count > 1 {
            range.push_str(&format!("item{{{}}} ", min_count - 1));
        }
        if max_count > min_count {
            let opt_count = max_count - min_count;
            if let Some(stop_word_done) = stop_word_done {
                range.push_str(&format!(
                    "( \"{}\" | item ){{0,{opt_count}}}",
                    stop_word_done.as_ref()
                ))
            } else {
                range.push_str(&format!("item{{0,{opt_count}}}"));
            };
        }
        range
    }
}
pub fn faux_url_validate_clean(content: &str, base: &str) -> Result<String, GrammarError> {
    if faux_url_parse(content, base).is_ok() {
        Ok(content
            .trim()
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "FauxUrl".to_string(),
        })
    }
}

pub fn faux_url_parse(content: &str, base: &str) -> Result<Vec<String>, GrammarError> {
    if let Some(trimmed_content) = content.trim().strip_prefix(base) {
        Ok(trimmed_content
            .trim_end_matches('-')
            .split('-')
            .map(String::from)
            .collect())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "FauxUrl".to_string(),
        })
    }
}
