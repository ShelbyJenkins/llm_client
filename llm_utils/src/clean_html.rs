use anyhow::Result;
use html2text::{config, render::text_renderer::TrivialDecorator};
use readability::extractor;
use std::io::Cursor;
use url::Url;

pub fn clean_html(html: &str) -> Result<String> {
    let mut input = Cursor::new(html);

    let readable = extractor::extract(&mut input, &Url::parse("http://example.com").unwrap())?;
    // Convert html to text with html2text
    // Trivial decorator removes all tags and leaves only text
    let decorator = TrivialDecorator::new();
    let text = config::with_decorator(decorator)
        .allow_width_overflow()
        .string_from_read(readable.content.as_bytes(), 10000)
        .unwrap();

    // Finally, remove excess whitespace
    Ok(super::clean_text::TextCleaner::new()
        .reduce_newlines_to_double_newline()
        .run(&text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_text::*;

    #[test]
    fn test_clean_html() {
        let cleaned = clean_html(&TEXT.html_short.content).unwrap();
        println!("{}", cleaned);
        assert!(
            !cleaned.contains("<p>") && !cleaned.contains("<div>") && !cleaned.contains("<h1>")
        );
    }
}
