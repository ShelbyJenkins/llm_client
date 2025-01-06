use regex::Regex;
use std::sync::LazyLock;

#[derive(Default)]
pub enum Newlines {
    Space,
    Single,
    #[default]
    TwoPlus,
    None,
}
#[derive(Default)]
pub struct TextCleaner {
    pub newlines: Newlines,
    pub remove_non_basic_ascii: bool,
    pub remve_citations: bool,
}
impl TextCleaner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn do_not_reduce_newlines(mut self) -> Self {
        self.newlines = Newlines::None;
        self
    }

    pub fn reduce_newlines_to_single_space(mut self) -> Self {
        self.newlines = Newlines::Space;
        self
    }

    pub fn reduce_newlines_to_single_newline(mut self) -> Self {
        self.newlines = Newlines::Single;
        self
    }

    pub fn reduce_newlines_to_double_newline(mut self) -> Self {
        self.newlines = Newlines::TwoPlus;
        self
    }

    pub fn remove_non_basic_ascii(mut self) -> Self {
        self.remove_non_basic_ascii = true;
        self
    }

    pub fn remove_citations(mut self) -> Self {
        self.remve_citations = true;
        self
    }

    pub fn run(&self, text: &str) -> String {
        let text = END_OF_LINE_REGEX.replace_all(text, "\n");
        let text = END_OF_PARAGRAPH_REGEX.replace_all(&text, "\n\n");
        let text = WHITE_SPACE_REGEX.replace_all(&text, " ");

        let text = match self.newlines {
            Newlines::Space => SINGLE_NEWLINE_REGEX.replace_all(&text, " "),
            Newlines::Single => SINGLE_NEWLINE_REGEX.replace_all(&text, "\n"),
            Newlines::TwoPlus => TWO_PLUS_NEWLINE_REGEX.replace_all(&text, "\n\n"),
            Newlines::None => text,
        };

        let text = if self.remove_non_basic_ascii {
            UNWANTED_CHARS_REGEX.replace_all(&text, "")
        } else {
            text
        };

        let text = if self.remve_citations {
            CITATIONS_REGEX.replace_all(&text, "")
        } else {
            text
        };

        SINGLE_SPACE_REGEX
            .replace_all(&text, " ")
            .trim()
            .to_string()
    }
}

pub fn normalize_whitespace(text: &str) -> String {
    let text = END_OF_LINE_REGEX.replace_all(text, "\n");
    let text = END_OF_PARAGRAPH_REGEX.replace_all(&text, "\n\n");
    WHITE_SPACE_REGEX.replace_all(&text, " ").to_string()
}

pub fn strip_unwanted_chars(text: &str) -> String {
    UNWANTED_CHARS_REGEX
        .replace_all(text, "")
        .trim()
        .to_string()
}

pub fn reduce_to_single_whitespace(text: &str) -> String {
    let text = SINGLE_SPACE_REGEX.replace_all(text, " ");
    SINGLE_NEWLINE_REGEX
        .replace_all(&text, "\n")
        .trim()
        .to_string()
}

//
// Newlines
//
pub static END_OF_LINE_SEQUENCES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        // Ascii
        r"(\\r\\n|\r\n)", // Windows // This must be first to avoid matching \r
        r"(\\r|\r)",      // MacOS
        r"(\\v|\v)",      // Vertical tab
        r"(\\f|\f)",      // Form feed
        r"\\n",           // Literal
        // Unicode
        r"\u{2028}",
    ]
});
pub static END_OF_LINE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&END_OF_LINE_SEQUENCES.join("|")).unwrap());
pub static SINGLE_NEWLINE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{1,}").unwrap());

//
// Paragraphs
//
pub static END_OF_PARAGRAPH_SEQUENCES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        // Unicode
        r"\u{2029}",
    ]
});
pub static END_OF_PARAGRAPH_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&END_OF_PARAGRAPH_SEQUENCES.join("|")).unwrap());
pub static TWO_PLUS_NEWLINE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{2,}").unwrap());

//
// White space
//
pub static WHITE_SPACE_SEQUENCES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        // Ascii
        r"\\s",
        r"(\\t|\t)",
        // Unicode
        r"\u{0020}",
        r"\u{00A0}",
        r"\u{1680}",
        r"\u{2000}",
        r"\u{2001}",
        r"\u{2002}",
        r"\u{2003}",
        r"\u{2004}",
        r"\u{2005}",
        r"\u{2006}",
        r"\u{2007}",
        r"\u{2008}",
        r"\u{2009}",
        r"\u{200A}",
        r"\u{2028}",
        r"\u{202F}",
        r"\u{205F}",
        r"\u{3000}",
        r"\u{0009}",
    ]
});

pub static WHITE_SPACE_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&WHITE_SPACE_SEQUENCES.join("|")).unwrap());
pub static SINGLE_SPACE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" {1,}").unwrap());

//
// Unwanted characters
//
pub static UNWANTED_CHARS_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"[^a-zA-Z0-9.,?!:;'\"\-\(\)\[\]\{\}$&@#%^*()\s]+"#).unwrap());
pub static CITATIONS_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\d{1,3}\]").unwrap());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        let ascii_text = "Ascii\tspaces here. Unicode\u{00A0}spaces here. Literal\\sspaces\\t.";
        let ascii_result = "Ascii spaces here. Unicode spaces here. Literal spaces .";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
        let ascii_text = "Ascii\nnewlines
. Unicode\u{2028}newlines. . Literal\\nnewlines.\\n";
        let ascii_result = "Ascii\nnewlines\n. Unicode\nnewlines.\n. Literal\nnewlines.\n";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
        let ascii_text = "Ascii\n\nparagraphs\r\n\r\n.Unicode\u{2029}paragraphs.  Literal\\n\\nparagraphs.\\r\\n\\r\\n";
        let ascii_result =
            "Ascii\n\nparagraphs\n\n.Unicode\n\nparagraphs.\n\n Literal\n\nparagraphs.\n\n";
        assert_eq!(normalize_whitespace(ascii_text), ascii_result);
    }

    #[test]
    fn test_clean_to_single_spaces() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\n And\nof course, newlines.\n\n";
        let ascii_result = "Ascii spaces here. Unicode spaces here. And of course, newlines.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_single_space()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_clean_to_single_newlines() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\nAnd of course, newlines.\n\nCool.";
        let ascii_result =
            "Ascii spaces here. Unicode spaces here.\nAnd of course, newlines.\nCool.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_single_newline()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_clean_to_double_newlines() {
        let ascii_text =
            "Ascii\tspaces here. Unicode\u{00A0}spaces here.\n\nAscii\n\nparagraphs.\r\n\r\nUnicode\u{2029}paragraphs. Cool.";
        let ascii_result =
            "Ascii spaces here. Unicode spaces here.\n\nAscii\n\nparagraphs.\n\nUnicode\n\nparagraphs.\n\nCool.";
        assert_eq!(
            TextCleaner::new()
                .reduce_newlines_to_double_newline()
                .run(ascii_text),
            ascii_result
        );
    }

    #[test]
    fn test_strip_unwanted_chars() {
        let ascii_text = r#"This is a "test" sentence. It include's 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*(). It also includes some advanced punctuation characters that should be removed, such as ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"#;
        let ascii_result = r#"This is a "test" sentence. It include's 'single' and "double" quotes, as well as other basic punctuation characters like commas, periods, question marks?, exclamation marks!, colons:, semicolons;, hyphens-, parentheses(), square brackets[], curly braces{}, and special characters $&@#%^*(). It also includes some advanced punctuation characters that should be removed, such as"#;
        assert_eq!(
            TextCleaner::new()
                .do_not_reduce_newlines()
                .remove_non_basic_ascii()
                .run(ascii_text),
            ascii_result
        );
    }
}
