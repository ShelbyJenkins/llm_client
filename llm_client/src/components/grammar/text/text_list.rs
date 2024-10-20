use super::{
    build_disallowed, create_range, Grammar, GrammarError, GrammarSetterTrait, RefCell,
    NEWLINE_CHARS,
};

#[derive(Clone, PartialEq)]
pub struct TextListGrammar {
    pub item_token_length: u32,
    pub min_count: u8,
    pub max_count: u8,
    pub item_prefix: Option<String>,
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub disallowed_chars: Vec<char>,
    grammar_string: RefCell<Option<String>>,
}

impl Default for TextListGrammar {
    fn default() -> Self {
        Self {
            min_count: 1,
            max_count: 5,
            item_token_length: 50,
            item_prefix: None,
            stop_word_done: None,
            stop_word_no_result: None,
            disallowed_chars: vec![],
            grammar_string: RefCell::new(None),
        }
    }
}

impl TextListGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::TextList(self)
    }
    pub fn min_count(mut self, min_count: u8) -> Self {
        self.min_count = min_count;
        self
    }

    pub fn max_count(mut self, max_count: u8) -> Self {
        self.max_count = max_count;
        self
    }

    pub fn item_token_length(mut self, item_token_length: u32) -> Self {
        self.item_token_length = item_token_length;
        self
    }

    pub fn item_prefix<S: Into<String>>(mut self, item_prefix: S) -> Self {
        self.item_prefix = Some(item_prefix.into());
        self
    }

    pub fn item_prefix_option<S: Into<Option<String>>>(mut self, item_prefix: S) -> Self {
        self.item_prefix = item_prefix.into();
        self
    }

    pub fn set_stop_word_done<S: Into<String>>(mut self, stop_word_done: S) -> Self {
        self.stop_word_done = Some(stop_word_done.into());
        self
    }

    pub fn set_stop_word_no_result<S: Into<String>>(mut self, stop_word_no_result: S) -> Self {
        self.stop_word_no_result = Some(stop_word_no_result.into());
        self
    }

    pub fn disallowed_chars(mut self, disallowed_chars: Vec<char>) -> Self {
        self.disallowed_chars.extend(disallowed_chars);
        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(list_grammar(
                self.min_count,
                self.max_count,
                self.item_token_length,
                &self.item_prefix,
                &self.stop_word_done,
                &self.stop_word_no_result,
                &self.disallowed_chars,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        list_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<Vec<String>, GrammarError> {
        list_parse(content)
    }
}

impl GrammarSetterTrait for TextListGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn list_grammar(
    min_count: u8,
    max_count: u8,
    item_token_length: u32,
    item_prefix: &Option<String>,
    stop_word_done: &Option<String>,
    stop_word_no_result: &Option<String>,
    disallowed_chars: &Vec<char>,
) -> String {
    let mut disallowed = disallowed_chars.to_vec();
    disallowed.extend(NEWLINE_CHARS.iter());
    disallowed.push('•');
    let disallowed = build_disallowed(&disallowed);
    let range = create_range(false, min_count, max_count, stop_word_done);

    let list_item = match item_prefix {
        Some(item_prefix) => format!(
            "item ::= \"• \" \"{}\" {disallowed}{{1,{}}} \"\\n\"",
            item_prefix,
            (item_token_length as f32 * 4.5).floor() as u32,
        ),
        None => format!(
            "item ::= \"• \" {disallowed}{{1,{}}} \"\\n\"",
            (item_token_length as f32 * 4.5).floor() as u32,
        ),
    };

    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= ( {range} | \"{}\" ) \" {}\"\n\n{list_item}",
            stop_word_no_result, stop_word_done
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= ( {range} | \"{}\" )\n\n{list_item}",
                stop_word_no_result
            )
        }
        (Some(stop_word_done), None) => {
            format!("root ::= {range} \" {}\"\n\n{list_item}", stop_word_done)
        }
        (None, None) => format!("root ::= {range}\n\n{list_item}"),
    }
}

pub fn list_validate_clean(content: &str) -> Result<String, GrammarError> {
    let trimmed_content = content.trim();
    if list_parse(trimmed_content).is_ok() {
        Ok(trimmed_content.to_owned())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "String".to_string(),
        })
    }
}

pub fn list_parse(content: &str) -> Result<Vec<String>, GrammarError> {
    let trimmed_content = content.trim();
    let mut items = Vec::new();
    for item in content.trim().split('\n') {
        let trimmed_sentence = item.trim().trim_start_matches("• ");

        if !trimmed_sentence.is_empty() {
            items.push(trimmed_sentence.to_owned());
        }
    }

    if items.is_empty() {
        return Err(GrammarError::ParseValueError {
            content: trimmed_content.to_string(),
            parse_type: "List".to_string(),
        });
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_string_with_stop_word_done_and_stop_word_no_result() {
        let grammar = TextListGrammar::default()
            .set_stop_word_done("done")
            .set_stop_word_no_result("null");

        let expected =
            "root ::= ( item{1} ( item | \"done\" ){0,4} | \"null\" ) \" done\"\n\nitem ::= [^\n\u{b}\u{c}\r\u{85}\u{2028}\u{2029}]{1,225} \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_with_stop_word_no_result() {
        let grammar = TextListGrammar::default().set_stop_word_no_result("null");

        let expected = "root ::= ( item{1,225} | \"null\" )\n\nitem ::= [^\n\u{b}\u{c}\r\u{85}\u{2028}\u{2029}] \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_with_stop_word_done() {
        let grammar = TextListGrammar::default().set_stop_word_done("done");

        let expected = "root ::= item{1,225} \" done\"\n\nitem ::= [^\n\u{b}\u{c}\r\u{85}\u{2028}\u{2029}] \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_without_stop_sequences() {
        let grammar = TextListGrammar::default();

        let expected =
            "root ::= item{0,225}\n\nitem ::= [^\n\u{b}\u{c}\r\u{85}\u{2028}\u{2029}] \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_parse_with_item_prefix() {
        let grammar = TextListGrammar::default()
            .set_stop_word_done("done")
            .set_stop_word_no_result("null")
            .item_prefix("Test");

        let expected =
        "root ::= ( item{1} ( item | \"done\" ){0,1} | \"null\" ) \" done\"\nitem ::= \"Test: \" [A-Z] [^\n\u{b}\u{c}\r\u{85}\u{2028}\u{2029}]+  \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    // #[test]
    // fn test_parse() {
    //     let grammar = TextListGrammar::default();
    //     let content = "  Hello, world!  ";
    //     let parsed_content = grammar.grammar_parse(content).unwrap();
    //     assert_eq!(parsed_content, vec!["Hello, world!"]);
    //     let content = "  Hello, world!\n Test.  ";
    //     let parsed_content = grammar.grammar_parse(content).unwrap();
    //     assert_eq!(parsed_content, vec!["Hello, world!", "Test."]);
    //     grammar.item_prefix("Test: ");
    //     let content = "  Test: Hello, world!\n Test: Test.  ";
    //     let parsed_content = grammar.grammar_parse(content).unwrap();
    //     assert_eq!(parsed_content, vec!["Hello, world!", "Test."]);
    // }
}
