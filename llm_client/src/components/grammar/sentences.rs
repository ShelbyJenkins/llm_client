use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone)]
pub struct SentencesGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub sentence_token_length: u32,
    pub capitalize_first: bool,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub concatenator: String,
    grammar_string: RefCell<Option<String>>,
}

impl Default for SentencesGrammar {
    fn default() -> Self {
        Self {
            min_count: 1,
            max_count: 1,
            sentence_token_length: 75,
            capitalize_first: true,
            stop_word_done: None,
            stop_word_no_result: None,
            concatenator: " ".to_string(),
            grammar_string: RefCell::new(None),
        }
    }
}

impl SentencesGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Sentences(self)
    }

    pub fn min_count(mut self, min_count: u8) -> Self {
        self.min_count = min_count;
        self
    }

    pub fn max_count(mut self, max_count: u8) -> Self {
        self.max_count = max_count;
        self
    }

    pub fn capitalize_first(mut self, capitalize_first: bool) -> Self {
        self.capitalize_first = capitalize_first;
        self
    }

    pub fn concatenator(mut self, concatenator: &str) -> Self {
        self.concatenator = concatenator.to_string();
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(sentences_grammar(
                self.min_count,
                self.max_count,
                self.sentence_token_length,
                self.capitalize_first,
                &self.concatenator,
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        sentences_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        sentences_parse(content)
    }
}

impl GrammarSetterTrait for SentencesGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub const SENTENCE_ITEM: &str = r#"[^\r\n\x0b\x0c\x85\u2028\u2029.?!]"#;

fn quotes() -> String {
    "( \"\\\"\" | \"'\" )".to_string()
}

pub fn sentences_grammar<T: AsRef<str>>(
    min_count: u8,
    max_count: u8,
    sentence_token_length: u32,
    capitalize_first: bool,
    concatenator: &str,
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    let quotes = quotes();
    let first = if capitalize_first { "[A-Z]" } else { "[a-z]" };
    let sentence_item = 
        format!(
            "( {first} | {quotes}) {SENTENCE_ITEM}{{1,{}}} [a-z] (\".\" | \"?\" | \"!\" | \".\" {quotes} | \"?\" {quotes} | \"!\" {quotes})",
            (sentence_token_length as f32 * 4.5).floor() as u32
        );
    
    let range = create_range(min_count, max_count, stop_word_done);
    let item = format!("item ::= {sentence_item} \"{concatenator}\"",);
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= ( {range} | \"{}\" ) \" {}\"\n{item}",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= ( {range} | \"{}\" )\n{item}",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= {range} \" {}\"\n{item}",
                stop_word_done.as_ref()
            )
        }
        (None, None) => format!("root ::= {range}\n{item}"),
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

pub fn sentences_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content
        .trim_start_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation())
        .trim_end_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation());

    if sentences_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        })
    }
}

pub fn sentences_parse(content: &str) -> Result<String, GrammarError> {
    if content.is_empty() {
        return Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        });
    }
    Ok(content.to_string())
}
