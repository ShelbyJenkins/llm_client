use llm_interface::{llms::LlmBackend, requests::completion::CompletionRequest};
use subject_of_text::ClassifySubjectOfText;

pub mod hierarchical_classification;
pub mod subject_of_text;

pub struct Classify {
    backend: std::sync::Arc<LlmBackend>,
}

impl Classify {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self { backend }
    }

    pub fn subject_of_text<T: AsRef<str>>(self, content: T) -> ClassifySubjectOfText {
        ClassifySubjectOfText::new(CompletionRequest::new(self.backend), content)
    }
}
