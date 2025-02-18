mod api_models;
mod gguf_models;

pub use api_models::generate_api_providers_and_models;
pub use gguf_models::generate_local_organizations_and_models;
use proc_macro2::TokenStream;

fn open_file(file_name: &str, file_path: &std::path::PathBuf) -> String {
    match std::fs::read_to_string(&file_path) {
        Ok(contents) => contents,
        Err(e) => panic!("Failed to read {file_name}: {}", e),
    }
}

fn open_and_parse<T>(file_name: &str, file_path: &std::path::PathBuf) -> T
where
    T: serde::de::DeserializeOwned,
{
    let contents = open_file(file_name, file_path);
    parse(file_name, &contents)
}

fn parse<T>(file_name: &str, contents: &str) -> T
where
    T: serde::de::DeserializeOwned,
{
    match serde_json::from_str(&contents) {
        Ok(content) => content,
        Err(e) => panic!("Failed to parse {file_name}: {}", e),
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
