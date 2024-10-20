use crate::components::cascade::CascadeFlow;
use crate::LlmClient;
use crate::{components::cascade::step::StepConfig, primitives::*};

use anyhow::Ok;
use llm_interface::requests::completion::CompletionRequest;
use serde::{Deserialize, Serialize};

use super::{Tag, TagCollection};

pub struct TagCollectionDescriber {
    pub base_req: CompletionRequest,
    pub flow: CascadeFlow,
    pub criteria: String,
}

impl TagCollectionDescriber {
    pub async fn run(
        llm_client: &LlmClient,
        criteria: &str,
        tag_collection: &mut TagCollection,
    ) -> crate::Result<()> {
        let mut describer = Self {
            base_req: CompletionRequest::new(llm_client.backend.clone()),
            criteria: criteria.to_owned(),
            flow: CascadeFlow::new("TagDescription"),
        };
        let mut root_tag = tag_collection.get_root_tag()?;
        for tag in root_tag.tags.values_mut() {
            describer.describe_tag(tag, 0).await?;
        }
        tag_collection.root_tag = Some(root_tag);
        Ok(())
    }

    const DESCRIPTION_MIN_COUNT: u8 = 3;
    const DESCRIPTION_MAX_COUNT: u8 = 4;
    const INSTRUCTIONS_MIN_COUNT: u8 = 3;
    const INSTRUCTIONS_MAX_COUNT: u8 = 5;
    fn describe_tag<'a>(
        &'a mut self,
        parent_tag: &'a mut Tag,
        level: u8,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + 'a>> {
        let description_min_count = Self::DESCRIPTION_MIN_COUNT.saturating_sub(level).max(1);
        let description_max_count = Self::DESCRIPTION_MAX_COUNT.saturating_sub(level).max(2);
        let instructions_min_count = Self::INSTRUCTIONS_MIN_COUNT.saturating_sub(level).max(1);
        let instructions_max_count = Self::INSTRUCTIONS_MAX_COUNT.saturating_sub(level).max(2);
        Box::pin(async move {
            if parent_tag.description.is_none() {
                self.flow = CascadeFlow::new("TagDescription");
                self.base_req.reset_completion_request();
                self.flow.open_cascade();

                let round = self.flow.new_round(self.describe_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                for child_tag in parent_tag.get_tag_names() {
                    let step_config = StepConfig {
                        step_prefix: Some(format!(
                            "Immediate Child Classification: '{}' ",
                            child_tag
                        )),
                        stop_word_done: "\n".to_owned(),
                        grammar: SentencesPrimitive::default()
                            .min_count(description_min_count)
                            .max_count(description_max_count)
                            .grammar(),
                        ..StepConfig::default()
                    };
                    round.add_inference_step(&step_config);
                    round.run_next_step(&mut self.base_req).await?;
                    round.last_step()?.set_dynamic_suffix("\n");
                }
                let description = round.display_outcome()?;
                round.close_round(&mut self.base_req)?;

                let round = self.flow.new_round(self.instruction_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some("Classification criteria: ".to_owned()),
                    grammar: SentencesPrimitive::default()
                        .min_count(instructions_min_count)
                        .max_count(instructions_max_count)
                        .grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
                let instructions = round.last_step()?.display_step_outcome()?;
                round.close_round(&mut self.base_req)?;

                let round = self.flow.new_round(self.is_applicable_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some(
                        "This classification is applicable if: The entity ".to_owned(),
                    ),
                    stop_word_done: "Therefore, it is ap".to_owned(),
                    grammar: TextPrimitive::default().text_token_length(300).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
                let is_applicable = round
                    .last_step()?
                    .primitive_result()
                    .ok_or_else(|| anyhow::anyhow!("is_applicable was None"))?;
                round.close_round(&mut self.base_req)?;

                self.flow.close_cascade()?;
                parent_tag.description = Some(TagDescription {
                    description,
                    instructions,
                    is_applicable,
                    is_parent_tag: true,
                });
            };
            for child_tag in parent_tag.tags.values_mut() {
                // if child_tag.tags.is_empty() {
                //     child_tag.description = parent_tag.description.clone();
                //     child_tag.description.as_mut().unwrap().is_parent_tag = false;
                // }
                self.describe_tag(child_tag, level + 1).await?;
            }
            Ok(())
        })
    }

    fn describe_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Describe the parent classification category '{}' by examining its relationship to its immediate child classifications.

        Parent Classification Category: '{}'

        Immediate Child Classifications: {}
    
        Using natural English, describe each immediate child classification in relation to the parent category, '{}'. For each immediate child classification, provide a single, newline seperated list item that explains its significance and how it relates to or specifies the parent category.

        Nested classifications within immediate children:

        {}

        ",
        tag.name.as_ref().unwrap(),
        tag.name.as_ref().unwrap(),
        tag.display_child_tags_comma(),
        tag.name.as_ref().unwrap(),
        tag.display_all_tags_with_nested_paths(),
        }
    }

    fn instruction_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        The criteria is instructions for how the parent classification category, '{}', should be applied.

        Criteria:

        {}
       
        Reword the criteria into specific instructions for applying the parent classification category, '{}' to an 'entity'. Focus on the key characteristics or conditions that would make this classification appropriate. Distill the essence of the child classifications, expressing them as a general trait or set of traits that would lead to the application of the parent classification category. The 'context text' provides additional details to aspects of the 'entity'.
 
        This should be a brief, 1-3 sentence, criteria that will be used to determine if the parent classification category applies to the entity. Use natural language.
        ",
        tag.name.as_ref().unwrap(),
        self.criteria,
        tag.name.as_ref().unwrap(),
        }
    }

    fn is_applicable_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Use the criteria to craft an 'is applicable if' sentence for the classification category '{}'. The statement should be a clear and concise sentence that explains the specific conditions under which this classification is applicable. Generalize using English natural language. No yapping. Be as brief as possible.
    
        Criteria:

        {}
    
        Your statement should follow this format:

        ```
        This classification is applicable if <specific or general conditions that meet the criteria>. Therefore, it is applicable.
        ```

        ",
        self.criteria,
        tag.name.as_ref().unwrap(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDescription {
    pub description: String,
    pub instructions: String,
    pub is_applicable: String,
    pub is_parent_tag: bool,
}

#[cfg(test)]
mod tests {
    use llm_models::local_model::GgufPresetTrait;

    use super::*;

    fn criteria() -> String {
        indoc::formatdoc! {"
            # Microbial Source Classification
            We have a hierarchical classification system used to categorize, filter, and sort the where a microbial organism was collected from.
            The 'entity' is the primary indicator of the collection source.
            The 'context text' provides additional details like the environment, location, or other aspects of the collection source.
            'Classifications' pertain to aspects of the collection source—what or where the microbial organism was collected from.
            We are interested in the source of a microbial organism. Classifications should only apply to direct aspects of the source or any details specified in the context text. An entity can have multiple classifications.
            "}
    }

    #[tokio::test]
    #[ignore]
    async fn test_tag_with_descriptions() -> crate::Result<()> {
        let mut tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()?;
        let llm_client = crate::LlmClient::llama_cpp()
            .llama3_1_70b_nemotron_instruct()
            .init()
            .await?;
        tag_collection
            .populate_descriptions(&llm_client, &criteria())
            .await?;
        let tags = tag_collection.get_root_tag()?;
        for tag in tags.get_tags() {
            println!("{:?}", tag.description);
        }
        // println!("{}", tags.display_all_tags());
        // println!("{}", tags.display_child_tags());
        // println!("{}", tags.display_all_tags_with_nested_paths());
        Ok(())
    }
}
