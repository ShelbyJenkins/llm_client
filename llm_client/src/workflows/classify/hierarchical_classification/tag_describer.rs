use crate::components::cascade::step::StepConfig;
use crate::components::cascade::CascadeFlow;
use crate::components::grammar::NoneGrammar;
use crate::LlmClient;

use super::*;
use serde::{Deserialize, Serialize};

use super::{tag::Tag, tag_collection::TagCollection};

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
            describer.describe_tag(tag).await?;
        }
        tag_collection.root_tag = Some(root_tag);
        Ok(())
    }

    fn describe_tag<'a>(
        &'a mut self,
        parent_tag: &'a mut Tag,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + 'a>> {
        Box::pin(async move {
            if parent_tag.description.is_none() {
                self.flow = CascadeFlow::new("TagDescription");
                self.base_req.reset_completion_request();
                self.flow.open_cascade();

                let description = if !parent_tag.get_tags().is_empty() {
                    let round = self.flow.new_round(self.describe_prompt(&parent_tag));
                    round.open_round(&mut self.base_req)?;
                    let step_config = StepConfig {
                        stop_word_done: "\n".to_owned(),
                        grammar: NoneGrammar::default().wrap(),
                        ..StepConfig::default()
                    };
                    round.add_inference_step(&step_config);
                    round.run_next_step(&mut self.base_req).await?;
                    round.last_step()?.set_dynamic_suffix("\n");

                    round.close_round(&mut self.base_req)?;
                    Some(round.display_outcome()?)
                } else {
                    None
                };

                let round = self.flow.new_round(self.instruction_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some(format!(
                        "'{}' application criteria:",
                        parent_tag.name.as_ref().unwrap()
                    )),
                    stop_word_done: "\n".to_owned(),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
                let instructions = round.last_step()?.display_step_outcome()?;
                round.close_round(&mut self.base_req)?;

                let round = self.flow.new_round(self.is_applicable_prompt(&parent_tag));
                round.open_round(&mut self.base_req)?;
                let step_config = StepConfig {
                    step_prefix: Some(format!(
                        "'{}' is applicable if: The entity",
                        parent_tag.name.as_ref().unwrap()
                    )),
                    stop_word_done: "\n".to_owned(),
                    grammar: NoneGrammar::default().wrap(),
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
                });
            }
            // for child_tag in parent_tag.tags.values_mut() {
            //     self.describe_tag(child_tag).await?;
            // }

            Ok(())
        })
    }

    fn describe_prompt(&self, tag: &Tag) -> String {
        let tag_name = tag.name.as_ref().unwrap();
        indoc::formatdoc! {"
        Describe the parent classification category '{tag_name}' by examining its relationship to its immediate child classifications.

        Parent Classification Category: 
        
        {tag_name}

        Immediate Child Classifications: 

        {}
    
        Your description should distill the child classifications of '{tag_name}', expressing them as a general trait or set of traits. The essence of the parent classificaiton category and it's children will serve as the primary description of the parent classification category. Be brief and concise. Use MLA, APA, or Chicago style natural english language. This should be a single sentence.
        ",
        tag.display_all_tags_with_nested_paths(),
        }
    }

    fn instruction_prompt(&self, tag: &Tag) -> String {
        let tag_name = tag.name.as_ref().unwrap();
        indoc::formatdoc! {"
        Reword the criteria into specific instructions for applying the classification category, '{tag_name}' to the given 'entity'.

        Criteria:

        {}
       
        A single sentence rule that will be used to determine if the '{tag_name}' classification category applies to the 'entity'. Focus on the key characteristics, aspect, or conditions that would make this classification applicable for the 'entity'.  
        ",
        self.criteria,
        }
    }

    fn is_applicable_prompt(&self, tag: &Tag) -> String {
        let tag_name = tag.name.as_ref().unwrap();
        indoc::formatdoc! {"
        Use the criteria to craft an 'is applicable if' sentence for the classification category '{tag_name}'. 
    
        Criteria:

        {}
    
        The statement should be a clear and concise single sentence that explains if '{tag_name}' is applicable to the 'entity'.

        Your statement should follow this format:
        ```
        '{tag_name}' is applicable if: <description of conditions that would satisfy the criteria>.
        ```
        ",
        self.criteria,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDescription {
    pub description: Option<String>,
    pub instructions: String,
    pub is_applicable: String,
}
