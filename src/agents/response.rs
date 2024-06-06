use anyhow::Result;

pub fn parse_text_generation_response(response: &str) -> Result<String> {
    if response.is_empty() {
        let error =
            anyhow::format_err!("parse_text_generation_response error: response.is_empty()");
        Err(error)
    } else {
        let response = if response.starts_with("assistant\n\n") {
            response.strip_prefix("assistant\n\n").unwrap().to_string()
        } else if response.starts_with("assistant\n") {
            response.strip_prefix("assistant\n").unwrap().to_string()
        } else {
            response.to_owned()
        };
        Ok(response)
    }
}
