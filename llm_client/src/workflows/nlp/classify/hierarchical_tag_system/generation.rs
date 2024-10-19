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

    const DESCRIPTION_MIN_COUNT: u8 = 5;
    const DESCRIPTION_MAX_COUNT: u8 = 7;
    const SUMMARY_MIN_COUNT: u8 = 4;
    const SUMMARY_MAX_COUNT: u8 = 5;
    fn describe_tag<'a>(
        &'a mut self,
        parent_tag: &'a mut Tag,
        level: u8,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + 'a>> {
        let description_min_count = Self::DESCRIPTION_MIN_COUNT.saturating_sub(level).max(2);
        let description_max_count = Self::DESCRIPTION_MAX_COUNT.saturating_sub(level).max(3);
        let summary_min_count = Self::SUMMARY_MIN_COUNT.saturating_sub(level).max(1);
        let summary_max_count = Self::SUMMARY_MAX_COUNT.saturating_sub(level).max(2);
        Box::pin(async move {
            if parent_tag.description.is_none() {
                self.flow = CascadeFlow::new("TagDescription");
                self.base_req.reset_completion_request();
                self.flow.open_cascade();

                let round = self.flow.new_round(self.describe_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some("The parent classification category ".to_owned()),
                    grammar: SentencesPrimitive::default()
                        .capitalize_first(false)
                        .min_count(description_min_count)
                        .max_count(description_max_count)
                        .grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
                let description = round.last_step()?.display_step_outcome()?;
                round.close_round(&mut self.base_req)?;

                let round = self.flow.new_round(self.instruction_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some("Classification criteria: ".to_owned()),
                    grammar: SentencesPrimitive::default()
                        .min_count(summary_min_count)
                        .max_count(summary_max_count)
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
                    step_prefix: Some("This classification is applicable if: The ".to_owned()),
                    stop_word_done: "Therefore, it is ap".to_owned(),
                    grammar: TextPrimitive::default().text_token_length(300).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
                let is_applicable = round.last_step()?.display_step_outcome()?;
                round.close_round(&mut self.base_req)?;

                self.flow.close_cascade()?;
                parent_tag.description = Some(TagDescription {
                    description,
                    instructions,
                    is_applicable,
                });
            };
            for child_tag in parent_tag.tags.values_mut() {
                if child_tag.tags.is_empty() {
                    continue;
                }
                self.describe_tag(child_tag, level + 1).await?;
            }
            Ok(())
        })
    }

    fn describe_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        In a single paragraph, describe the parent classification category and it's relationship to it's child classifications.
    
        Parent Classification Category:

        {}
    
        Child Classifications:

        {}
    
        Use natural language to describe the following the relationship between the parent classification category, '{}', and the child classifications.
        ",
        tag.name.as_ref().unwrap(),
        tag.display_all_tags(),
        tag.name.as_ref().unwrap(),
        }
    }

    fn instruction_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        The criteria is instructions for how the parent classification category, '{}', should be applied.

        Criteria:

        {}
        
        Reword the criteria so that it is specific instructions for applying the parent classification category, '{}' to an entity. If any of the child classifictions apply, then the parent classification category should be applied. The context text provides additional details to aspects of the entity.
 
        This should be a brief, 1-3 sentence, criteria that will be used to determine if the parent classification category applies to the entity. Use natural language.
        ",
        tag.name.as_ref().unwrap(),
        self.criteria,
        tag.name.as_ref().unwrap(),
        }
    }

    fn is_applicable_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Use the criteria to craft an 'is applicable if' sentence for the classification category '{}'. The statement should be a clear and concise sentence that explains the specific conditions under which this classification is applicable. Use natural language.
    
        Criteria:

        {}
    
        Your statement should follow this format:

        ```
        This classification is applicable if <specific conditions that meet the criteria>. Therefore, it is applicable.
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
            .mistral_nemo_instruct2407()
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
        // println!("{}", tags.display_all_tags_with_paths());
        Ok(())
    }
}
