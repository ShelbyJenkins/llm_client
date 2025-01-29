pub mod extract;

use super::*;
use extract::Extract;

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
}
