use regex::Regex;
use std::{fs::File, io::Read};
use tiktoken_rs::cl100k_base;

/// Creates a window of text normalized to the specified token size.
///
/// This function takes a string of text and a desired token size, and returns
/// a new string that represents a window of the original text. The window is
/// centered around the middle of the text and its size is determined by the
/// `target_token_size` parameter.
///
/// The function uses the `cl100k_base` tokenizer from the `tiktoken` library
/// to tokenize the input text. If the number of tokens in the input text is
/// less than or equal to `target_token_size`, the function returns the
/// original text as is.
///
/// If the number of tokens exceeds `target_token_size`, the function calculates
/// the start and end indices of the token window based on the desired size and
/// the actual token count. The preserved tokens are then decoded back into a
/// string using the `decode` method of the tokenizer.
///
/// # Arguments
///
/// * `text` - The input text to create a window from.
/// * `target_token_size` - The desired number of tokens in the window.
///
/// # Returns
///
/// A new string that represents the normalized window of text, or the original
/// text if its token count is less than or equal to `target_token_size`.
///
/// # Examples
///
/// ```
/// use llm_client::text_utils::create_text_window;
///
/// let text = "This is a sample text. It will be truncated or returned as is based on the target token size.";
/// let target_token_size = 10;
///
/// let window = create_text_window(text, target_token_size);
/// println!("Normalized window: {}", window);
/// ```
pub fn create_text_window(text: &str, target_token_size: usize) -> String {
    let bpe = cl100k_base().unwrap();
    let tokens = bpe.encode_ordinary(text);

    if tokens.len() <= target_token_size {
        return text.to_string();
    }

    let start_token_index = (tokens.len() - target_token_size) / 2;
    let end_token_index = start_token_index + target_token_size;

    let preserved_tokens = &tokens[start_token_index..end_token_index];
    bpe.decode(preserved_tokens.to_vec()).unwrap()
}

pub fn get_token(text: &str) -> usize {
    let bpe = cl100k_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(text);
    if tokens.len() > 1 {
        panic!("More than one token found in text: {}", text);
    }
    tokens[0]
}
pub fn get_char_tokens(text: &Vec<String>) -> Vec<usize> {
    let bpe = cl100k_base().unwrap();
    let mut tokens = Vec::new();
    for char in text {
        let token = bpe.encode_with_special_tokens(char);
        if token.len() > 1 {
            panic!("More than one token found in text: {}", char);
        }
        tokens.push(token[0])
    }
    tokens
}

pub fn tiktoken_len(text: &str) -> u16 {
    let bpe = cl100k_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(text);
    tokens.len() as u16
}

pub fn tiktoken_len_of_document_list(texts: &Vec<String>) -> u16 {
    let mut token_count = 0;
    for text in texts {
        let tokens = tiktoken_len(text);
        token_count += tokens;
    }
    token_count
}

pub fn filter_drop_words(texts: &[String], drop_words: &[String]) -> Vec<String> {
    texts
        .iter()
        .filter(|s| {
            !s.split_whitespace().any(|w| {
                drop_words
                    .iter()
                    .any(|dw| w.to_lowercase() == dw.to_lowercase())
            })
        })
        .cloned()
        .collect()
}

pub fn split_on_newline(text: &str) -> Vec<String> {
    text.split('\n').map(|s| s.to_string()).collect()
}
pub fn join_with_newline(texts: &[String]) -> String {
    texts.join("\n").trim().to_string()
}
pub fn split_on_double_newline(text: &str) -> Vec<String> {
    text.split("\n\n").map(|s| s.to_string()).collect()
}
pub fn join_with_double_newline(texts: &[String]) -> String {
    texts.join("\n\n").trim().to_string()
}

pub fn clean_text_content(text: &str) -> String {
    let text = convert_white_space(text);
    let text = reduce_excess_whitespace(&text);
    strip_unwanted_chars(&text).trim().to_string()
}

pub fn convert_white_space(text: &str) -> String {
    let patterns = vec![
        (r"\r", "\r"),
        (r"\t", "\t"),
        (r"\n", "\n"),
        (r" ", " "),
        (r"\u{00A0}", " "),
        (r"\u{2009}", " "),
        (r"\u{2002}", " "),
        (r"\u{2003}", " "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }
    reduced_text
}

pub fn reduce_excess_whitespace(text: &str) -> String {
    let patterns = vec![
        (r"\r{1,}", "\r"),
        (r"\t{1,}", "\t"),
        // Preserve double newlines for future splitting
        (r"\n{2,}", "\n\n"),
        (r" {1,}", r" "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }
    // Trim leading and trailing spaces
    reduced_text.trim().to_string()
}

pub fn strip_unwanted_chars(text: &str) -> String {
    // Define which chars can be kept; Alpha-numeric chars, punctuation, and whitespaces.
    let allowed_chars = "a-zA-Z0-9\\p{P}\\s";

    // Create a regex pattern to keep only the allowed characters.
    let regex_pattern = format!("[^{}]", allowed_chars);
    let re = Regex::new(&regex_pattern).unwrap();

    // Remove unwanted chars using regex.
    re.replace_all(text, "").into_owned()
}

pub fn remove_all_white_space_except_space(text: &str) -> String {
    let patterns = vec![
        (r"\r", " "),
        (r"\t", " "),
        (r"\v", " "),
        (r"\f", " "),
        (r"\n", " "),
        // Matches variants of space characters
        (r" ", " "),
        (r"\u00A0", " "),
        (r"\u2009", " "),
        (r"\u2002", " "),
        (r"\u2003", " "),
    ];
    let mut reduced_text = String::from(text);
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, replacement).into_owned();
    }

    reduce_excess_whitespace(&reduced_text)
}

pub fn split_text_with_regex(text: &str, separator: &str, keep_separator: bool) -> Vec<String> {
    let re = match separator {
        r"\d+\.\s" => Regex::new(r"\d+\.\s").expect("Invalid regex pattern"), // Matches "1. ", "2. ", etc.
        r"\d+:\s" => Regex::new(r"\d+:\s").expect("Invalid regex pattern"), // Matches "1: ", "2: ", etc.
        r"\d+\)\s" => Regex::new(r"\d+\)\s").expect("Invalid regex pattern"), // Matches "1) ", "2) ", etc.
        r"\d+\s" => Regex::new(r"\d+\s").expect("Invalid regex pattern"), // Matches "1 ", "2 ", etc.
        "Feature \\d+:" => Regex::new("Feature \\d+:").expect("Invalid regex pattern"), // Matches "1. ", "2. ", etc.
        "feature \\d+:" => Regex::new("feature \\d+:").expect("Invalid regex pattern"), // Matches "1. ", "2. ", etc.
        "n-" => Regex::new(r"\d+-\s").expect("Invalid regex pattern"), // Matches "1- ", "2- ", etc.
        "[n]" => Regex::new(r"\[\d+\]\s").expect("Invalid regex pattern"), // Matches "[1] ", "[2] ", etc.
        _ => {
            let sanitized_separator = regex::escape(separator);
            Regex::new(&sanitized_separator).expect("Invalid regex pattern")
        }
    };

    if keep_separator {
        let mut splits = Vec::new();
        let mut last = 0;

        for mat in re.find_iter(text) {
            splits.push(text[last..mat.start()].to_string());
            splits.push(mat.as_str().to_string());
            last = mat.end();
        }

        if last < text.len() {
            splits.push(text[last..].to_string());
        }

        splits.into_iter().filter(|s| !s.is_empty()).collect()
    } else {
        re.split(text)
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }
}

pub fn load_content(file_path: &str) -> String {
    let path = std::path::Path::new(&file_path);
    match File::open(path) {
        Ok(mut file) => {
            let mut content = String::new();
            match file.read_to_string(&mut content) {
                Ok(_) => {
                    if content.trim().is_empty() {
                        panic!("file_path '{}' is empty.", path.display())
                    }
                }
                Err(e) => panic!("Failed to read file: {}", e),
            }
            content
        }
        Err(e) => panic!("Failed to open file: {}", e),
    }
}
