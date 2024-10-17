pub mod sentences;
pub mod text;
pub mod text_list;
pub mod words;

use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

const NEWLINE_CHARS: [char; 8] = [
    '\r', '\n', '\u{000C}', '\u{000B}', '\u{000C}', '\u{0085}', '\u{2028}', '\u{2029}',
];

// We'll have to store each char, and then manually allow or disallow each char
fn build_disallowed(disallowed_chars: &[char]) -> String {
    let mut disallowed = disallowed_chars.to_vec();
    disallowed.extend(NEWLINE_CHARS.iter());
    // Now deduplicate
    disallowed.sort();
    disallowed.dedup();

    format!("[^{}]", disallowed.iter().collect::<String>())
}

fn build_quotes(disallowed_chars: &Vec<char>) -> Option<String> {
    if disallowed_chars.contains(&'"') && disallowed_chars.contains(&'\'') {
        None
    } else if disallowed_chars.contains(&'"') {
        Some("\"'\"".to_string())
    } else if disallowed_chars.contains(&'\'') {
        Some("\"\\\"\"".to_string())
    } else {
        Some("( \"\\\"\" | \"'\" )".to_string())
    }
}

fn create_range<T: AsRef<str>>(
    first_item: bool,
    min_count: u8,
    max_count: u8,
    stop_word_done: &Option<T>,
) -> String {
    let max_count = match max_count.cmp(&min_count) {
        std::cmp::Ordering::Less => {
            eprintln!("Max count must be greater than or equal to min count. Setting max count to min count.");
            min_count
        }
        _ => max_count,
    };
    if min_count == 0 && max_count == 0 || min_count == 0 && max_count == 1 {
        if first_item {
            "first{0,1}".to_owned()
        } else {
            "item{0,1}".to_owned()
        }
    } else {
        let mut range = String::new();
        if first_item {
            if min_count > 0 {
                range.push_str(&format!("first{{{min_count}}} "));
                if min_count > 1 {
                    range.push_str(&format!("item{{{}}} ", min_count - 1));
                }
            } else {
                if let Some(stop_word_done) = stop_word_done {
                    range.push_str(&format!(
                        "( first | \"{}\" ){{0,1}}",
                        stop_word_done.as_ref()
                    ))
                } else {
                    range.push_str(&format!("first{{0,1}}"));
                };
            }
        } else {
            if min_count > 0 {
                range.push_str(&format!("item{{{min_count}}} "));
            }
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
