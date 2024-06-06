use super::*;
use anyhow::Result;
use std::collections::HashMap;

pub mod grammar_text;
pub mod grammar_text_list;
pub mod logit_bias_text;
pub mod unstructured_text;
pub use grammar_text::GrammarText;
pub use grammar_text_list::GrammarTextList;
pub use logit_bias_text::LogitBiasText;
pub use unstructured_text::UnstructuredText;

pub struct TextGenerator<'a> {
    pub llm_client: &'a LlmClient,
}

impl<'a> TextGenerator<'a> {
    pub fn new(llm_client: &'a LlmClient) -> Self {
        Self { llm_client }
    }

    /// Your basic, every day LLM call.
    ///
    /// # Returns
    ///
    /// An `UnstructuredText` instance.
    pub fn basic_text(&self) -> UnstructuredText<'a> {
        UnstructuredText::new(
            self.llm_client,
            self.llm_client.default_request_config.clone(),
        )
    }

    /// Generates text with grammar based restrictions.
    ///
    /// # Returns
    ///
    /// A `GrammarText` instance.
    pub fn grammar_text(&self) -> GrammarText<'a> {
        GrammarText::new(
            self.llm_client,
            self.llm_client.default_request_config.clone(),
        )
    }

    /// Generates list of texts. Returns a vector of strings.
    ///
    /// # Returns
    ///
    /// A `GrammarTextList` instance.
    pub fn grammar_list(&self) -> GrammarTextList<'a> {
        match &self.llm_client.backend {
            LlmBackend::Llama(_) => GrammarTextList::new(
                self.llm_client,
                self.llm_client.default_request_config.clone(),
            ),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(_) => GrammarTextList::new(
                self.llm_client,
                self.llm_client.default_request_config.clone(),
            ),
            LlmBackend::OpenAi(_) => {
                panic!("OpenAI backend is not supported for grammar based calls.")
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for grammar based calls.")
            }
        }
    }

    /// Generates text with logit bias restrictions or modifications.
    ///
    /// # Returns
    ///
    /// A `LogitBiasText` instance.
    pub fn logit_bias_text(&self) -> LogitBiasText<'a> {
        LogitBiasText::new(
            self.llm_client,
            self.llm_client.default_request_config.clone(),
        )
    }
}
