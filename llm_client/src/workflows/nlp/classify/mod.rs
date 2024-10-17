use llm_interface::requests::completion::CompletionRequest;
pub mod hierarchy;
pub mod label;
pub mod subject_of_text;

pub struct Classify {
    pub base_req: CompletionRequest,
}

impl Classify {
    pub fn new(base_req: CompletionRequest) -> Self {
        Self { base_req }
    }

    pub fn entity(self, content: &str) -> subject_of_text::ClassifySubjectOfText {
        subject_of_text::ClassifySubjectOfText::new(self.base_req, content)
    }

}
