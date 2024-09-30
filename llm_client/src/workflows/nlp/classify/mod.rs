use llm_interface::requests::completion::CompletionRequest;

pub mod hierarchy;

pub struct Classify {
    pub base_req: CompletionRequest,
}

impl Classify {
    pub fn new(base_req: CompletionRequest) -> Self {
        Self { base_req }
    }
}
