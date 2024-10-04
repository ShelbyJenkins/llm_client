use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};
use llm_interface::requests::completion::CompletionRequest;

#[derive(Clone)]
pub struct ClassifyEntity {
    pub base_req: CompletionRequest,
    pub content: String,
    pub entity_label: Option<String>,
    pub tags: Vec<String>,
    pub duration: std::time::Duration,
    pub flow: CascadeFlow,
}

impl ClassifyEntity {
    pub fn new(base_req: CompletionRequest, content: &str) -> Self {
        Self {
            base_req,
            content: content.to_owned(),
            entity_label: None,
            tags: vec![],
            duration: std::time::Duration::default(),
            flow: CascadeFlow::new("ClassifyEntity"),
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        self.run_cascade().await?;
        println!("{}", self.flow);
        self.duration = self.flow.duration;
        Ok(self)
    }

    async fn run_cascade(&mut self) -> crate::Result<()> {
        self.flow.new_round(
            format!("In the following text, identify the primary entity that serves as the main focus. This entity should be a concrete, tangible entity, such as a physical object, creature, or any specific item mentioned.\nGuidelines:\nIdentify the Single Primary Entity: Determine the most significant concrete entity that the text is mainly about.\nFocus on Concrete Entities Only: The primary entity should be tangible and physically existing (e.g., a thing, object, noun).\nIgnore Descriptive Attributes: disambiguate by ignoring adjective and prepositions.\nConsider Nested Entities: Be attentive if the primary entity includes or is part of another entity.\nAfter analyzing the text, state your answer as: \"The primary entity in the text is:\": ... \nThe text is: '{}'", self.content)
        );

        let step_config = StepConfig {
            step_prefix: Some(
                "Thinking out loud about the entity discussed in the text...".to_owned(),
            ),
            stop_word_done: "The primary entity in the text is".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);

        let step_config = StepConfig {
            step_prefix: Some("The primary entity in the text is:".to_owned()),
            grammar: WordsPrimitive::default().max_count(3).grammar(),
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.flow
            .last_round()?
            .run_all_steps(&mut self.base_req)
            .await?;
        match self.flow.primitive_result() {
            Some(entity_label) => {
                self.entity_label = Some(entity_label);
            }
            None => {
                crate::bail!("No entity identified.");
            }
        };
        self.flow.close_cascade()?;
        Ok(())
    }
}

impl std::fmt::Display for ClassifyEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ClassifyEntity:")?;
        crate::i_nln(f, format_args!("content: {}", self.content))?;
        crate::i_nln(f, format_args!("entity_label: {:?}", self.entity_label))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.duration))?;
        for tag in &self.tags {
            crate::i_nln(f, format_args!("tag: {}", tag))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::LlmClient;

    #[tokio::test]
    #[ignore]
    pub async fn test() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;

        let req = llm_client
            .nlp()
            .classify()
            .entity("A green turtle on a log in a mountain lake.");
        let entity = req.run().await?;
        println!("{}", entity);
        Ok(())
    }
}
