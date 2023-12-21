use regex::Regex;
use tiktoken_rs::cl100k_base;

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

pub fn clean_text_content(text: &str) -> String {
    let text = strip_unwanted_chars(text);
    reduce_excess_whitespace(&text)
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

pub fn reduce_excess_whitespace(text: &str) -> String {
    // Define patterns for different whitespace characters
    let patterns = vec![
        (r" ", r" {1,}"),
        (r"\t", r"\t{1,}"),
        (r"\n", r"\n{1,}"),
        (r"\r", r"\r{1,}"),
        (r"\v", r"\v{1,}"),
        (r"\f", r"\f{1,}"),
    ];

    let mut reduced_text = String::from(text);

    // Replace any sequential occurrences of each whitespace character
    // greater than one with just one.
    for (char, pattern) in patterns {
        let re = Regex::new(pattern).unwrap();
        reduced_text = re.replace_all(&reduced_text, char).into_owned();
    }
    let pattern = r"\s+";
    let re = Regex::new(pattern).unwrap();
    let reduced_text = re.replace_all(text, " ").into_owned();
    reduced_text.trim().to_string()
}

pub fn remove_all_white_space_except_space(text: &str) -> String {
    // Create a regex to match all whitespace characters except space
    let re = Regex::new(r"[\n\r\t\f\v]+").unwrap();
    let text = re.replace_all(text, "");

    // Create a regex to replace multiple spaces with a single space
    let re = Regex::new(r" +").unwrap();
    let text = re.replace_all(&text, " ");

    // Trim leading and trailing spaces
    text.trim().to_string()
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
