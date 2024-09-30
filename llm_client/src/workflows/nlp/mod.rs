// pub mod classify;
pub mod extract;
use extract::Extract;
use llm_interface::{llms::LlmBackend, requests::completion::CompletionRequest};

pub struct Nlp {
    pub base_req: CompletionRequest,
}

impl Nlp {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    pub fn extract(self) -> Extract {
        Extract::new(self.base_req)
    }

    // pub fn classify(self) -> classify::Classify {
    //     classify::Classify::new(self.base_req)
    // }
}
