pub mod extract;

use crate::{components::base_request::BaseLlmRequest, llm_backends::LlmBackend};
use extract::Extract;
use std::rc::Rc;

pub struct Nlp {
    pub base_req: BaseLlmRequest,
}

impl Nlp {
    pub fn new(backend: &Rc<LlmBackend>) -> Self {
        Self {
            base_req: BaseLlmRequest::new_from_backend(backend),
        }
    }

    pub fn extract(self) -> Extract {
        Extract::new(self.base_req)
    }
}
