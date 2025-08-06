use std::collections::HashMap;

use minijinja::{
    Environment, Error, ErrorKind, context,
    value::{Value, ValueKind, from_args},
};

use crate::gguf::metadata::tokenizer::EmbeddedTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    #[error("Unsupported tokenizer format: {0}")]
    UnsupportedTokenizerFormat(String),

    #[error("Missing token ID for {token_name} in the GGML tokenizer metadata")]
    MissingTokenId { token_name: &'static str },

    #[error("Chat template is missing in the GGML tokenizer metadata")]
    MissingChatTemplate,

    #[error("Token not found for ID {id}: {token_name}")]
    TokenNotFound { id: usize, token_name: &'static str },

    #[error("Error setting base generation prefix: {0}")]
    BaseGenerationPrefixError(String),
}

#[derive(Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ChatTemplate {
    pub chat_template: String,
    pub bos: Option<String>,
    pub eos: String,
    pub unk: Option<String>,
    pub base_generation_prefix: String,
}

impl ChatTemplate {
    pub fn from_embedded_tokenizer(
        embedded_tokenizer: &EmbeddedTokenizer,
    ) -> Result<Self, ChatTemplateError> {
        match embedded_tokenizer {
            EmbeddedTokenizer::HuggingFace { .. } => {
                Err(ChatTemplateError::UnsupportedTokenizerFormat(
                    "HuggingFace tokenizer is not supported".to_string(),
                ))
            }
            EmbeddedTokenizer::RwkvWorld { .. } => {
                Err(ChatTemplateError::UnsupportedTokenizerFormat(
                    "RWKV-World tokenizer is not supported".to_string(),
                ))
            }
            EmbeddedTokenizer::Other { .. } => Err(ChatTemplateError::UnsupportedTokenizerFormat(
                "Other tokenizer is not supported".to_string(),
            )),
            EmbeddedTokenizer::Ggml {
                tokens,
                special,
                chat_template,
                ..
            } => {
                let chat_template = chat_template
                    .as_ref()
                    .ok_or_else(|| ChatTemplateError::MissingChatTemplate)?
                    .to_owned();
                let bos = special
                    .bos
                    .and_then(|bos| tokens.get(bos as usize))
                    .map(ToOwned::to_owned);
                let eos = special
                    .eos
                    .and_then(|eos| tokens.get(eos as usize))
                    .ok_or_else(|| ChatTemplateError::MissingTokenId { token_name: "EOS" })?
                    .to_owned();
                let unk = special
                    .unknown
                    .and_then(|unk| tokens.get(unk as usize))
                    .map(ToOwned::to_owned);

                let base_generation_prefix =
                    Self::base_generation_prefix(&chat_template, &bos, &eos, &unk)?;

                Ok(ChatTemplate {
                    chat_template,
                    bos,
                    eos,
                    unk,
                    base_generation_prefix,
                })
            }
        }
    }

    fn base_generation_prefix(
        chat_template: &str,
        bos: &Option<String>,
        eos: &str,
        unk: &Option<String>,
    ) -> Result<String, ChatTemplateError> {
        let user_message_1 = std::collections::HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), "test_user_message_1".to_string()),
        ]);
        let assistant_message_1 = std::collections::HashMap::from([
            ("role".to_string(), "assistant".to_string()),
            (
                "content".to_string(),
                "test_assistant_message_1".to_string(),
            ),
        ]);

        let message_1 = apply_chat_template(
            &vec![user_message_1.clone()],
            &chat_template,
            bos.as_deref(),
            &eos,
            unk.as_deref(),
        );
        let message_1 = message_1.trim_end_matches(eos).to_owned();
        let message_2 = apply_chat_template(
            &vec![user_message_1, assistant_message_1],
            &chat_template,
            bos.as_deref(),
            &eos,
            unk.as_deref(),
        );

        // Find the point where the outputs start to differ
        let diff_index = message_1
            .chars()
            .zip(message_2.chars())
            .position(|(a, b)| a != b)
            .unwrap_or(message_1.len());

        // Extract the differing part
        let diff_part = &message_2[diff_index..];

        // Find the start of the assistant content
        if let Some(content_index) = diff_part.find("test_assistant_message_1") {
            // The prefix is everything before the content
            let base_generation_prefix = diff_part[..content_index]
                .trim_start_matches(eos)
                .to_string();
            if base_generation_prefix.is_empty() {
                return Err(ChatTemplateError::BaseGenerationPrefixError(
                    "Base generation prefix is empty".to_string(),
                ));
            }
            return Ok(base_generation_prefix);
        } else {
            return Err(ChatTemplateError::BaseGenerationPrefixError(
                "Could not find the assistant content in the generated message".to_string(),
            ));
        }
    }
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ChatTemplate:")?;
        writeln!(f, "chat_template: too long to print nicely")?;
        writeln!(f, "bos_token: {:?}", self.bos)?;
        writeln!(f, "eos_token: {}", self.eos)?;
        writeln!(f, "unk_token: {:?}", self.unk)?;
        writeln!(
            f,
            "base_generation_prefix: {:?}",
            self.base_generation_prefix
        )?;
        Ok(())
    }
}

/// Applies a chat template to a message, given a message and a chat template.
///
/// # Arguments
///
/// * `message` - The message as a HashMap.
/// * `chat_template` - The chat template as a String.
///
/// # Returns
///
/// The formatted message as a String.
fn apply_chat_template(
    messages: &Vec<HashMap<String, String>>,
    chat_template: &str,
    bos_token: Option<&str>,
    eos_token: &str,
    unk_token: Option<&str>,
) -> String {
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_template("chat_template", chat_template)
        .expect("Failed to add template");
    env.add_function("raise_exception", raise_exception);

    env.set_unknown_method_callback(|state, value, method, args| match (value.kind(), method) {
        (ValueKind::String, "strip") => {
            let _: () = from_args(args)?;
            Ok(Value::from(value.as_str().unwrap_or("").trim()))
        }
        (ValueKind::Map, "items") => {
            let _: () = from_args(args)?;
            state.apply_filter("items", &[value.clone()])
        }
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("object has no method named {}", method),
        )),
    });

    let tmpl = env
        .get_template("chat_template")
        .expect("Failed to get template");

    let unk_token = unk_token.unwrap_or("");
    let bos_token = bos_token.unwrap_or("");

    tmpl.render(context! {
        messages => messages,
        add_generation_prompt => false,
        bos_token => bos_token,
        eos_token => eos_token,
        unk_token => unk_token,
    })
    .expect("Failed to render template without system prompt")
}

/// This exists specifically for the minijinja template engine to raise an exception.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}
