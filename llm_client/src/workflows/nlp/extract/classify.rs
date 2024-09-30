use llm_interface::requests::completion::CompletionRequest;

use crate::components::instruct_prompt::InstructPrompt;

#[derive(Clone)]
pub struct ClassifyEntities {
    pub max_entity_count: u8,
    pub entity_count: Option<u32>,
    pub entity_type: String,
    pub base_req: CompletionRequest,
    pub instruct_prompt: InstructPrompt,
}

impl ClassifyEntities {
    pub fn new(base_req: CompletionRequest) -> Self {
        Self {
            max_entity_count: 5,
            entity_count: None,
            entity_type: "topic".to_string(),
            instruct_prompt: InstructPrompt::new(),
            base_req,
        }
        // let mut flow = CascadeFlow::new("ClassifyEntities");
    }
}
