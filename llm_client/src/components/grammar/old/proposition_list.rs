pub const STOP_WORD: &str = "This list is complete.";
pub const PREFIX: &str = "- ";

#[derive(Clone, Default)]
pub struct PropositionListGrammar {
    pub min_count: u8,
    pub max_count: u8,
    pub max_sentences: u8,
    pub stop_word: String,
    pub item_prefix_start: String,
    pub item_prefix_end: String,
    pub starting_point: u8,
    pub requires_build: bool,
}

impl PropositionListGrammar {
    pub fn new() -> Self {
        Self {
            min_count: 1,
            max_count: 3,
            max_sentences: 3,
            stop_word: STOP_WORD.to_owned(),
            item_prefix_start: PREFIX.to_owned(),
            item_prefix_end: ".".to_owned(),
            starting_point: 1,
            requires_build: true,
        }
    }
}

impl PropositionListGrammar {
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

    pub fn max_sentences(&mut self, max_sentences: u8) -> &mut Self {
        self.max_sentences = max_sentences;
        self.requires_build = true;
        self
    }

    pub fn stop_word<S: Into<String>>(&mut self, stop_word: S) -> &mut Self {
        self.stop_word = stop_word.into();
        self.requires_build = true;
        self
    }

    pub fn item_prefix_start<S: Into<String>>(&mut self, item_prefix_start: S) -> &mut Self {
        self.item_prefix_start = item_prefix_start.into();
        self.requires_build = true;
        self
    }

    pub fn item_prefix_end<S: Into<String>>(&mut self, item_prefix_end: S) -> &mut Self {
        self.item_prefix_end = item_prefix_end.into();
        self.requires_build = true;
        self
    }

    pub fn starting_point(&mut self, starting_point: u8) -> &mut Self {
        self.starting_point = starting_point;
        self.requires_build = true;
        self
    }

    pub fn grammar_string(&self) -> String {
        grammar_string(
            self.min_count,
            self.max_count,
            self.max_sentences,
            &self.stop_word,
            &self.item_prefix_start,
            &self.item_prefix_end,
            self.starting_point,
        )
    }

    pub fn parse(&self, response_content: &str) -> Vec<String> {
        parse(
            response_content,
            &self.item_prefix_start,
            &self.item_prefix_end,
        )
    }
}

const ITEM_PATTERN: &str =
    r#"item ::= [A-Z] [^\r\n\x0b\x0c\x85\u2028\u2029.?!]+ [a-z] (". " | "? " | "! ")"#;
pub fn grammar_string(
    min_count: u8,
    max_count: u8,
    max_sentences: u8,
    stop_word: &str,
    prefix_start: &str,
    prefix_end: &str,
    starting_point: u8,
) -> String {
    let starting_point = starting_point.max(1);
    let min_count = starting_point + min_count;
    let max_count = max_count.max(min_count);

    let max_count = match max_count.cmp(&min_count) {
        std::cmp::Ordering::Less => {
            eprintln!("Max count must be greater than or equal to min count. Setting max count to min count.");
            min_count
        }
        _ => max_count,
    };
    let mut pattern = "root ::=".to_string();

    for i in starting_point..min_count {
        pattern.push_str(&format!(
            " \" {prefix_start} {i} {prefix_end} \" item{{1,{max_sentences}}} \"\n\""
        ));
    }
    for i in min_count..=max_count {
        pattern.push_str(&format!(
            " ( \" {prefix_start} {i} {prefix_end} \" item{{1,{max_sentences}}} | \" {stop_word}\" ) \"\n\""
        ));
    }
    pattern.push_str(&format!(" \" {stop_word}\"\n{ITEM_PATTERN}"));
    pattern
}

pub fn parse(response_content: &str, prefix_start: &str, prefix_end: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for sentence in response_content.split('\n') {
        let trimmed_sentence = sentence
            .trim()
            .trim_start_matches(prefix_start)
            .trim_start_matches(|c: char| c.is_numeric() || c.is_whitespace())
            .trim_start_matches(prefix_end)
            .trim_start_matches(|c: char| !(c.is_alphanumeric()))
            .trim_end_matches(|c: char| !(c.is_alphanumeric() || c.is_ascii_punctuation()))
            .to_owned();
        if !trimmed_sentence.is_empty() {
            sentences.push(trimmed_sentence.to_owned());
        }
    }
    sentences
}
