use super::{build_disallowed, Grammar, GrammarError, GrammarSetterTrait, RefCell, NEWLINE_CHARS};

#[derive(Clone, PartialEq, Debug)]
pub struct TextGrammar {
    pub item_token_length: u32,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub disallowed_chars: Vec<char>,
    pub allow_newline: bool,
    grammar_string: RefCell<Option<String>>,
}

impl Default for TextGrammar {
    fn default() -> Self {
        Self {
            item_token_length: 200,
            stop_word_done: None,
            stop_word_no_result: None,
            disallowed_chars: vec![],
            allow_newline: false,
            grammar_string: RefCell::new(None),
        }
    }
}

impl TextGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::Text(self)
    }

    pub fn item_token_length(mut self, item_token_length: u32) -> Self {
        self.item_token_length = item_token_length;
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

    pub fn allow_newline(mut self, allow_newline: bool) -> Self {
        self.allow_newline = allow_newline;
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(text_grammar(
                self.item_token_length,
                &self.stop_word_done,
                &self.stop_word_no_result,
                self.allow_newline,
                &self.disallowed_chars,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        text_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<String, GrammarError> {
        text_parse(content)
    }
}

impl GrammarSetterTrait for TextGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn text_grammar(
    item_token_length: u32,
    stop_word_done: &Option<String>,
    stop_word_no_result: &Option<String>,
    allow_newline: bool,
    disallowed_chars: &Vec<char>,
) -> String {
    let disallowed = if allow_newline {
        build_disallowed(disallowed_chars)
    } else {
        let mut disallowed = disallowed_chars.to_vec();
        disallowed.extend(NEWLINE_CHARS.iter());
        build_disallowed(&disallowed)
    };
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => {
            format!(
                "root ::= ( item{{1,{}}} | \"{stop_word_no_result}\" ) \" {stop_word_done}\"\nitem ::= {disallowed}",
                (item_token_length as f32 * 4.5).floor() as u32, 
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= item{{1,{}}} \" {stop_word_done}\"\nitem ::= {disallowed}",
                (item_token_length as f32 * 4.5).floor() as u32,
      
            )
        }
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= ( item{{1,{}}} | \"{stop_word_no_result}\" )\nitem ::= {disallowed}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
        (None, None) => {
            format!(
                "root ::= item{{0,{}}}\n\nitem ::= {disallowed}",
                (item_token_length as f32 * 4.5).floor() as u32
            )
        }
    }
}

pub fn text_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content
        .trim_start_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation())
        .trim_end_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii_punctuation());

    if text_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        })
    }
}

pub fn text_parse(content: &str) -> Result<String, GrammarError> {
    if content.is_empty() {
        return Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        });
    }
    Ok(content.to_string())
}
