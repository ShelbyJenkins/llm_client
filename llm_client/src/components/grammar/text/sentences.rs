use super::{
    build_disallowed, build_quotes, create_range, Grammar, GrammarError, GrammarSetterTrait,
    RefCell, NEWLINE_CHARS,
};

#[derive(Clone, PartialEq)]
pub struct SentencesGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub sentence_token_length: u32,
    pub capitalize_first: bool,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub concatenator: String,
    pub disallowed_chars: Vec<char>,
    grammar_string: RefCell<Option<String>>,
}

impl Default for SentencesGrammar {
    fn default() -> Self {
        let mut disallowed_chars = NEWLINE_CHARS.to_vec();
        disallowed_chars.push('.');
        disallowed_chars.push('!');
        disallowed_chars.push('?');
        Self {
            min_count: 1,
            max_count: 1,
            sentence_token_length: 50,
            capitalize_first: true,
            stop_word_done: None,
            stop_word_no_result: None,
            concatenator: " ".to_string(),
            disallowed_chars,
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

    pub fn disallowed_char(mut self, disallowed_char: char) -> Self {
        self.disallowed_chars.push(disallowed_char);
        self
    }

    pub fn disallowed_chars(mut self, disallowed_chars: Vec<char>) -> Self {
        self.disallowed_chars.extend(disallowed_chars);
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
                &self.disallowed_chars,
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

pub fn sentences_grammar<T: AsRef<str>>(
    min_count: u8,
    max_count: u8,
    sentence_token_length: u32,
    capitalize_first: bool,
    concatenator: &str,
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
    disallowed_chars: &Vec<char>,
) -> String {
    let char_count = (sentence_token_length as f32 * 4.5).floor() as u32;

    let disallowed = build_disallowed(disallowed_chars);
    let quotes = build_quotes(disallowed_chars);

    if capitalize_first {
        let range = create_range(false, min_count, max_count, stop_word_done);
        let sentence_item = format!(
            "item ::= {} \"{concatenator}\"",
            build_sentence_item(char_count, true, &disallowed, &quotes)
        );
        match (stop_word_done, stop_word_no_result) {
            (Some(stop_word_done), Some(stop_word_no_result)) => format!(
                "root ::= ( {range} | \"{}\" ) \" {}\"\n\n{sentence_item}",
                stop_word_no_result.as_ref(),
                stop_word_done.as_ref()
            ),
            (None, Some(stop_word_no_result)) => {
                format!(
                    "root ::= ( {range} | \"{}\" )\n\n{sentence_item}",
                    stop_word_no_result.as_ref()
                )
            }
            (Some(stop_word_done), None) => {
                format!(
                    "root ::= {range} \" {}\"\n\n{sentence_item}",
                    stop_word_done.as_ref()
                )
            }
            (None, None) => format!("root ::= {range}\n\n{sentence_item}"),
        }
    } else {
        let first_item = format!(
            "first ::= {} \"{concatenator}\"",
            build_sentence_item(char_count, false, &disallowed, &quotes)
        );
        let range = create_range(true, min_count, max_count, stop_word_done);
        let sentence_item = format!(
            "item ::= {} \"{concatenator}\"",
            build_sentence_item(char_count, true, &disallowed, &quotes)
        );
        match (stop_word_done, stop_word_no_result) {
            (Some(stop_word_done), Some(stop_word_no_result)) => format!(
                "root ::= ( {range} | \"{}\" ) \" {}\"\n\n{first_item}\n\n{sentence_item}",
                stop_word_no_result.as_ref(),
                stop_word_done.as_ref()
            ),
            (None, Some(stop_word_no_result)) => {
                format!(
                    "root ::= ( {range} | \"{}\" )\n\n{first_item}\n\n{sentence_item}",
                    stop_word_no_result.as_ref()
                )
            }
            (Some(stop_word_done), None) => {
                format!(
                    "root ::= {range} \" {}\"\n\n{first_item}\n\n{sentence_item}",
                    stop_word_done.as_ref()
                )
            }
            (None, None) => format!("root ::= {range}\n\n{first_item}\n\n{sentence_item}"),
        }
    }
}

fn build_sentence_item(
    char_count: u32,
    capitalize_start: bool,
    disallowed: &str,
    quotes: &Option<String>,
) -> String {
    let first = if capitalize_start { "[A-Z]" } else { "[a-z]" };
    let sentence_item = if let Some(quotes) = quotes {
        format!(
            "({quotes} | {first}) {disallowed}{{1,{char_count}}} [a-z] (\".\" | \"?\" | \"!\" | \".\" {quotes} | \"?\" {quotes} | \"!\" {quotes})"
        )
    } else {
        format!("{first} {disallowed}{{1,{char_count}}} [a-z] (\".\" | \"?\" | \"!\")")
    };
    sentence_item
}

pub fn sentences_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content.trim();
    // .trim_start_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation())
    // .trim_end_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation());

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
