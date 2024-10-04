use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};
use hierarchy::TagSystem;
use llm_interface::requests::{
    completion::CompletionRequest,
    req_components::{RequestConfig, RequestConfigTrait},
};
pub mod entity;
pub mod hierarchy;


pub struct LabelEntity {
    pub base_req: CompletionRequest,
    pub entity: ClassifyEntity,
    pub tags: TagSystem,
}

impl LabelEntity {
    pub fn new(base_req: CompletionRequest, content: &str, tags: &TagSystem) -> Self {
        Self {
            base_req,
            entity: ClassifyEntity::new(content),
            tags: tags.clone(),
        }
    }

    pub async fn run(mut self) -> crate::Result<ClassifyEntity> {
        let flow = self.run_backend().await?;
        println!("{}", flow);
        self.entity.duration = flow.duration;
        Ok(self.entity.clone())
    }

    async fn run_backend(&mut self) -> crate::Result<CascadeFlow> {
        self.run_cascade().await
    }

    fn build_tag_list(&self) -> String {
        let mut list = String::new();
        for name in self.tags.get_parent_tag_names() {
            if list.is_empty() {
                list.push('"');
                list.push_str(name);
            } else {
                list.push_str(", ");
                list.push('"');
                list.push_str(name);
            }
            list.push('"');
        }
        list
    }

    async fn run_cascade(&mut self) -> crate::Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("LabelEntity");
        flow.open_cascade();

        flow.new_round(
            format!("In the following text, identify the primary entity that serves as the main focus. This entity should be a concrete, tangible entity, such as a physical object, creature, or any specific item mentioned.\nGuidelines:\nIdentify the Single Primary Entity: Determine the most significant concrete entity that the text is mainly about.\nFocus on Concrete Entities Only: The primary entity should be tangible and physically existing (e.g., a thing, object, noun).\nIgnore Descriptive Attributes: disambiguate by ignoring adjective and prepositions.\nConsider Nested Entities: Be attentive if the primary entity includes or is part of another entity.\nAfter analyzing the text, state your answer as: \"The primary entity in the text is:\": ... \nThe text is: '{}'", self.entity.content)
        );

        let step_config = StepConfig {
            step_prefix: Some(
                "Thinking out loud about the entity discussed in the text...".to_owned(),
            ),
            stop_word_done: "The primary entity in the text is".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);

        let step_config = StepConfig {
            step_prefix: Some("The primary entity in the text is:".to_owned()),
            grammar: TextPrimitive::default().text_token_length(10).grammar(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_all_steps(&mut self.base_req).await?;
        match flow.primitive_result() {
            Some(entity_name) => {
                self.entity.entity = Some(entity_name);
            }
            None => {
                crate::bail!("No entity identified.");
            }
        };

        flow.new_round(
                format!("Pontificate about the previously identified entity and its relation to one or more of the listed categories.\nDiscuss which category or categories most accurately and comprehensively describe the entity.\nWhen talking about multiple categories, go in order of relevance, with the most applicable category first.\nIf none of the categories are relevant, state \"None of the above.\"\nCategories: {}.", self.build_tag_list())
            );
        let step_config = StepConfig {
            step_prefix: Some(
                "Thinking about the entity and and how it might be classified...".to_owned(),
            ),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };

        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.open_round(&mut self.base_req)?;
        flow.last_round()?.run_next_step(&mut self.base_req).await?;

        for i in 1..=self.tags.get_parent_tag_names().len() {
            let step_config = if i == 1 {
                StepConfig {
                    step_prefix: Some("The best classification is:".to_owned()),
                    stop_word_no_result: Some("None of the above.".to_owned()),
                    grammar: self.tags.grammar(),
                    ..StepConfig::default()
                }
            } else {
                StepConfig {
                    step_prefix: Some("The next most accurate classification is:".to_owned()),
                    stop_word_no_result: Some("None of the above.".to_owned()),
                    grammar: self.tags.grammar(),
                    ..StepConfig::default()
                }
            };

            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut self.base_req).await?;

            match flow.primitive_result() {
                Some(tag_name) => {
                    self.tags.remove_parent_tag(&tag_name)?;
                    self.entity.tags.push(tag_name);
                }
                None => {
                    break;
                }
            };
        }
        flow.last_round()?.close_round(&mut self.base_req)?;
        flow.close_cascade()?;

        Ok(flow)
    }
}

impl RequestConfigTrait for LabelEntity {
    fn config(&mut self) -> &mut RequestConfig {
        &mut self.base_req.config
    }

    fn reset_request(&mut self) {
        self.tags = TagSystem::new();
        self.base_req.reset_completion_request();
    }
}

#[derive(Clone)]
pub struct ClassifyEntity {
    pub content: String,
    pub entity: Option<String>,
    pub tags: Vec<String>,
    pub duration: std::time::Duration,
}

impl ClassifyEntity {
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_owned(),
            entity: None,
            tags: vec![],
            duration: std::time::Duration::default(),
        }
    }
}

impl std::fmt::Display for ClassifyEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ClassifyEntity:")?;
        crate::i_nln(f, format_args!("content: {}", self.content))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.duration))?;
        for tag in &self.tags {
            crate::i_nln(f, format_args!("tag: {}", tag))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::LlmClient;

    #[tokio::test]
    #[ignore]
    pub async fn test() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;
        let input = "\
        terrestrial
        aquatic
        host-associated
        salinity
        age group
        space
        human";

        let tag_system = TagSystem::create_from_string(input);

        let req = llm_client
            .nlp()
            .classify("A green turtle on a log in a mountain lake.", &tag_system);
        let entity = req.run().await?;
        println!("{}", entity);
        Ok(())
    }
}
