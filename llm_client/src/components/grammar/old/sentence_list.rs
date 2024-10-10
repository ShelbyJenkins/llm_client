use crate::grammar::shared_consts::*;

#[derive(Clone, Default)]
pub struct SentenceListGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub item_prefix: Option<String>,
    pub stop_word_done: Option<String>,
    pub no_result_stop_word: Option<String>,
    pub requires_build: bool,
}

impl SentenceListGrammar {
    pub fn new() -> Self {
        Self {
            min_count: 1,
            max_count: 2,
            item_prefix: None,
            stop_word_done: None,
            no_result_stop_word: None,
            requires_build: true,
        }
    }

    pub fn set_stop_word_done<S: Into<String>>(&mut self, stop_word_done: S) -> &mut Self {
        self.stop_word_done = Some(stop_word_done.into());
        self.requires_build = true;
        self
    }

    pub fn set_no_result_stop_word<S: Into<String>>(
        &mut self,
        no_result_stop_word: S,
    ) -> &mut Self {
        self.no_result_stop_word = Some(no_result_stop_word.into());
        self.requires_build = true;
        self
    }

    pub fn min_count(&mut self, min_count: u8) -> &mut Self {
        self.min_count = min_count;
        self.requires_build = true;
        self
    }

    pub fn max_count(&mut self, max_count: u8) -> &mut Self {
        self.max_count = max_count;
        self.requires_build = true;
        self
    }

    pub fn item_prefix<S: Into<String>>(&mut self, item_prefix: S) -> &mut Self {
        self.item_prefix = Some(item_prefix.into());
        self.requires_build = true;
        self
    }

    pub fn grammar_string(&self) -> String {
        grammar_string(
            self.min_count,
            self.max_count,
            &self.item_prefix,
            &self.stop_word_done,
            &self.no_result_stop_word,
        )
    }

    pub fn parse(&self, response_content: &str) -> Vec<String> {
        parse(response_content, &self.item_prefix)
    }
}

pub fn grammar_string<T: AsRef<str>>(
    min_count: u8,
    max_count: u8,
    item_prefix: &Option<T>,
    stop_word_done: &Option<T>,
    no_result_stop_word: &Option<T>,
) -> String {
    let range = create_range(min_count, max_count, stop_word_done);
    let list_item = match item_prefix {
        Some(item_prefix) => format!(
            "item ::= \"{}: \" {SENTENCE_ITEM} \"\\n\"",
            item_prefix.as_ref()
        ),
        None => format!("item ::= {SENTENCE_ITEM} \"\\n\""),
    };

    match (stop_word_done, no_result_stop_word) {
        (Some(stop_word_done), Some(no_result_stop_word)) => format!(
            "root ::= \" \" ( {range} | \"{}\" ) \" {}\"\n{list_item}",
            no_result_stop_word.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(no_result_stop_word)) => {
            format!(
                "root ::= \" \" ( {range} | \"{}\" )\n{list_item}",
                no_result_stop_word.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" {range} \" {}\"\n{list_item}",
                stop_word_done.as_ref()
            )
        }
        (None, None) => format!("root ::= \" \" {range}\n{list_item}"),
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

pub fn parse<T: AsRef<str>>(response_content: &str, item_prefix: &Option<T>) -> Vec<String> {
    let mut sentences = Vec::new();
    for sentence in response_content.split('\n') {
        let trimmed_sentence = if let Some(item_prefix) = item_prefix {
            sentence.trim().trim_start_matches(item_prefix.as_ref())
        } else {
            sentence.trim()
        };

        let trimmed_sentence = trimmed_sentence
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .trim_end_matches(|c: char| !(c.is_alphanumeric() || c.is_ascii_punctuation()))
            .to_owned();
        if !trimmed_sentence.is_empty() {
            sentences.push(trimmed_sentence.to_owned());
        }
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_string_with_stop_word_done_and_no_result_stop_word() {
        let mut grammar = SentenceListGrammar::new();
        grammar
            .min_count(1)
            .max_count(2)
            .set_stop_word_done("done")
            .set_no_result_stop_word("null");
        let expected =
            "root ::= \" \" ( item{1} ( item | \"done\" ){0,1} | \"null\" ) \" done\"\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);

        grammar
            .min_count(3)
            .max_count(5)
            .set_stop_word_done("done")
            .set_no_result_stop_word("null");
        let expected =
            "root ::= \" \" ( item{3} ( item | \"done\" ){0,2} | \"null\" ) \" done\"\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);

        grammar
            .min_count(0)
            .max_count(5)
            .set_stop_word_done("done")
            .set_no_result_stop_word("null");
        let expected =
            "root ::= \" \" ( ( item | \"done\" ){0,5} | \"null\" ) \" done\"\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);

        grammar
            .min_count(0)
            .max_count(0)
            .set_stop_word_done("done")
            .set_no_result_stop_word("null");
        let expected =
            "root ::= \" \" ( item{0,1} | \"null\" ) \" done\"\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_with_no_result_stop_word() {
        let mut grammar = SentenceListGrammar::new();
        grammar
            .min_count(1)
            .max_count(2)
            .set_no_result_stop_word("null");
        let expected = "root ::= \" \" ( item{1} item{0,1} | \"null\" )\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_with_stop_word_done() {
        let mut grammar = SentenceListGrammar::new();
        grammar.min_count(1).max_count(2).set_stop_word_done("done");
        let expected = "root ::= \" \" item{1} ( item | \"done\" ){0,1} \" done\"\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_grammar_string_without_stop_sequences() {
        let mut grammar = SentenceListGrammar::new();
        grammar.min_count(1).max_count(2);
        let expected = "root ::= \" \" item{1} item{0,1}\nitem ::= [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_parse_with_item_prefix() {
        let mut grammar = SentenceListGrammar::new();

        grammar
            .min_count(1)
            .max_count(2)
            .set_stop_word_done("done")
            .set_no_result_stop_word("null")
            .item_prefix("Test");
        let expected =
        "root ::= \" \" ( item{1} ( item | \"done\" ){0,1} | \"null\" ) \" done\"\nitem ::= \"Test: \" [A-Z] [^\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029.?!]+ [a-z] (\". \" | \"? \" | \"! \") \"\\n\"";
        assert_eq!(grammar.grammar_string(), expected);
    }

    #[test]
    fn test_parse() {
        let mut grammar = SentenceListGrammar::new();
        let response_content = "  Hello, world!  ";
        let parsed_content = grammar.parse(response_content);
        assert_eq!(parsed_content, vec!["Hello, world!"]);
        let response_content = "  Hello, world!\n Test.  ";
        let parsed_content = grammar.parse(response_content);
        assert_eq!(parsed_content, vec!["Hello, world!", "Test."]);
        grammar.item_prefix("Test: ");
        let response_content = "  Test: Hello, world!\n Test: Test.  ";
        let parsed_content = grammar.parse(response_content);
        assert_eq!(parsed_content, vec!["Hello, world!", "Test."]);
    }
}
