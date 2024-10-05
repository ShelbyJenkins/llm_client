use llm_interface::requests::completion::CompletionRequest;
pub mod entity;
pub mod hierarchy;
pub mod label;

pub struct Classify {
    pub base_req: CompletionRequest,
}

impl Classify {
    pub fn new(base_req: CompletionRequest) -> Self {
        Self { base_req }
    }

    pub fn entity(self, content: &str) -> entity::ClassifyEntity {
        entity::ClassifyEntity::new(self.base_req, content)
    }
}
