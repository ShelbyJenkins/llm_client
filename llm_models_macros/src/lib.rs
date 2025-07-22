mod api_models;
mod gguf_models;

pub use api_models::generate_api_providers_and_models;
pub use gguf_models::generate_local_organizations_and_models;
use proc_macro2::TokenStream;
use syn::Ident;

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Error, Result};

fn open_file(file_name: &str, file_path: &std::path::PathBuf) -> crate::Result<String> {
    match std::fs::read_to_string(&file_path) {
        Ok(contents) => Ok(contents),
        Err(e) => crate::bail!("Failed to read {file_name}: {}", e),
    }
}

fn open_and_parse<T>(file_name: &str, file_path: &std::path::PathBuf) -> crate::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let contents = open_file(file_name, file_path)?;
    parse(file_name, &contents)
}

fn parse<T>(file_name: &str, contents: &str) -> crate::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    match serde_json::from_str(&contents) {
        Ok(content) => Ok(content),
        Err(e) => crate::bail!("Failed to parse {file_name}: {}", e),
    }
}

fn format_and_write(output_path: &std::path::PathBuf, code: TokenStream) {
    let syntax_tree = match syn::parse_file(&code.to_string()) {
        Ok(syntax_tree) => syntax_tree,
        Err(e) => panic!("Failed to parse generated code: {}", e),
    };
    let code = prettyplease::unparse(&syntax_tree);

    match std::fs::write(&output_path, code.to_string()) {
        Ok(_) => (),
        Err(e) => panic!("Failed to write to ({:?}) {}", output_path, e),
    }
}

fn to_enum(name: &str) -> String {
    if name.is_empty() {
        panic!("Invalid enum variant name: empty string");
    }

    let mut chars = name.chars();
    let first_char = loop {
        match chars.next() {
            Some(c) if c.is_alphabetic() => break c.to_uppercase().next().unwrap(),
            Some(_) => continue,
            None => panic!("Invalid enum variant name: {name} (no alphabetic characters)"),
        }
    };

    let mut result = String::new();
    result.push(first_char);

    // Convert the rest while preserving camelCase word boundaries
    let mut capitalize_next = false;
    for c in chars {
        if c.is_alphanumeric() {
            if capitalize_next {
                // Convert to uppercase if it follows a special character
                result.extend(c.to_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        } else {
            // Mark the next character for capitalization
            capitalize_next = true;
        }
    }

    let enum_name = match result.as_str() {
        "Self" => result + "Enum",
        _ => result,
    };

    syn::parse_str::<Ident>(&enum_name).unwrap();
    enum_name
}

fn to_func(name: &str) -> String {
    let lowercase = name.to_lowercase();

    let mut result = String::new();

    let mut chars = lowercase.chars();
    let first_char = loop {
        match chars.next() {
            Some(c) if c.is_alphabetic() => break c,
            Some(_) => continue,
            None => panic!("Invalid fn name: {name} (no alphabetic characters)"),
        }
    };
    result.push(first_char);

    for c in chars {
        if c.is_alphanumeric() {
            result.push(c);
        } else {
            if !result.ends_with('_') {
                result.push('_');
            }
        }
    }
    if result.ends_with('_') {
        result.pop();
    }
    let fn_name = match result.as_str() {
        "as" | "break" | "const" | "continue" | "crate" | "else" | "enum" | "extern" | "false"
        | "fn" | "for" | "if" | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move"
        | "mut" | "pub" | "ref" | "return" | "self" | "static" | "struct" | "super" | "trait"
        | "true" | "type" | "unsafe" | "use" | "where" | "while" | "async" | "await" | "dyn" => {
            result + "_fn"
        }
        _ => result,
    };

    syn::parse_str::<Ident>(&fn_name).unwrap();

    fn_name
}
