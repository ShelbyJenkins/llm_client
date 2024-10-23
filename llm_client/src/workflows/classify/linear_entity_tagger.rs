use crate::components::cascade::step::StepConfig;
use crate::components::cascade::CascadeFlow;

use crate::components::grammar::*;
use crate::LlmClient;

use llm_interface::requests::completion::CompletionRequest;

use super::hierarchical_tag_system::Tag;

#[derive(Debug, Clone)]
pub struct Critera {
    pub entity_definition: String,
    pub instructions: String,
}

impl Critera {
    pub fn new(entity_definition: &str, instructions: &str) -> Self {
        Self {
            entity_definition: entity_definition.to_owned(),
            instructions: instructions.to_owned(),
        }
    }
}

pub struct LinearEntityTagger<'a> {
    pub llm_client: &'a LlmClient,
    pub entity: String,
    pub context_text: String,
    pub criteria: Critera,
    pub refined_instructions: String,
    pub tag_collection: Tag,
    pub assigned_terminal_tags: Vec<Tag>,
    pub start_time: std::time::Instant,
    pub duration: std::time::Duration,
}

impl<'a> LinearEntityTagger<'a> {
    pub fn new(
        llm_client: &'a LlmClient,
        entity: &str,
        context_text: &str,
        criteria: &Critera,
        tag_collection: Tag,
    ) -> Self {
        Self {
            llm_client,
            entity: entity.to_owned(),
            context_text: context_text.to_owned(),
            criteria: criteria.clone(),
            refined_instructions: String::new(),
            start_time: std::time::Instant::now(),
            duration: std::time::Duration::default(),
            tag_collection,
            assigned_terminal_tags: Vec::new(),
        }
    }

    pub async fn run(&mut self) -> crate::Result<()> {
        self.start_time = std::time::Instant::now();
        let tag_collection = self.tag_collection.clone();
        let mut flow = CascadeFlow::new("LinearEntityTagger");
        let mut base_req = CompletionRequest::new(self.llm_client.backend.clone());

        // refined_instructions
        flow.open_cascade();
        flow.new_round(self.refine_instructions_prompt());
        flow.last_round()?.open_round(&mut base_req)?;

        flow.last_round()?.add_guidance_step(
            &StepConfig::default(),
            format!(
                "1. Classifying: We are classifying the enitity, '{}', from the text.\n",
                self.entity
            ),
        );
        flow.last_round()?.run_next_step(&mut base_req).await?;
        flow.last_round()?.add_guidance_step(
            &StepConfig::default(),
            format!(
                "2. '{}' Definition: {}\n",
                self.entity, self.criteria.entity_definition,
            ),
        );
        flow.last_round()?.run_next_step(&mut base_req).await?;
        let step_config = StepConfig {
            step_prefix: Some("3. Relevant details:".to_owned()),
            stop_word_done: "4.".to_owned(),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(&mut base_req).await?;
        let step_config = StepConfig {
            step_prefix: Some("4. Specialized instructions:".to_owned()),
            stop_word_done: "5.".to_owned(),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(&mut base_req).await?;
        if let Some(refined_instructions) = flow.last_round()?.last_step()?.primitive_result() {
            self.refined_instructions = refined_instructions;
        } else {
            crate::bail!("No refined instructions");
        }
        flow.last_round()?.close_round(&mut base_req)?;

        self.recursive_tag_evaluation(tag_collection, flow, base_req)
            .await?;
        self.duration = self.start_time.elapsed();
        Ok(())
    }

    fn recursive_tag_evaluation(
        &mut self,
        parent_tag: Tag,
        initial_flow: CascadeFlow,
        initial_base_req: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + '_>> {
        Box::pin(async move {
            for potential_tag in parent_tag.get_tags() {
                let mut flow = initial_flow.clone();
                let mut base_req = initial_base_req.clone();
                flow.new_round(self.reason_prompt(&potential_tag));
                flow.last_round()?.open_round(&mut base_req)?;

                let step_config = StepConfig {
                    step_prefix: Some("1. Is applicable if:".to_owned()),
                    stop_word_done: "2.".to_owned(),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                flow.last_round()?.add_inference_step(&step_config);
                flow.last_round()?.run_next_step(&mut base_req).await?;

                let step_config = StepConfig {
                    step_prefix: Some("2. Relevant Aspects and Details:".to_owned()),
                    stop_word_done: "3.".to_owned(),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                flow.last_round()?.add_inference_step(&step_config);
                flow.last_round()?.run_next_step(&mut base_req).await?;

                let step_config = StepConfig {
                    step_prefix: Some("3. Category applicability:".to_owned()),
                    stop_word_done: "4.".to_owned(),
                    // stop_word_no_result: Some("none".to_owned()),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                flow.last_round()?.add_inference_step(&step_config);
                flow.last_round()?.run_next_step(&mut base_req).await?;

                if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                    if response.contains("none") || response.contains("None") {
                        continue; // Conintue to next tag
                    }
                } else {
                    continue; // Conintue to next tag
                }

                let step_config = StepConfig {
                    step_prefix: Some("4. Applicable categories:".to_owned()),
                    stop_word_done: "::".to_owned(),
                    stop_word_no_result: Some("none".to_owned()),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                flow.last_round()?.add_inference_step(&step_config);
                flow.last_round()?.run_next_step(&mut base_req).await?;
                flow.last_round()?.close_round(&mut base_req)?;

                if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                    if !response.contains("none") && !response.contains("None") {
                        if potential_tag.tags.is_empty() {
                            self.assigned_terminal_tags.push(potential_tag.clone());
                        } else {
                            let new_flow = flow.clone();
                            let new_base_req = base_req.clone();
                            self.list(potential_tag.clone(), new_flow, new_base_req)
                                .await?;
                        }
                    }
                }
                // if !self.assigned_terminal_tags.is_empty() {
                //     break;
                // }
            }

            Ok(())
        })
    }

    fn list(
        &mut self,
        parent_tag: Tag,
        mut flow: CascadeFlow,
        mut base_req: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<Tag>>> + '_>> {
        Box::pin(async move {
            let mut assigned_terminal_tags = Vec::new();
            flow.new_round(self.list_prompt(&parent_tag));
            flow.last_round()?.open_round(&mut base_req)?;

            let step_config = StepConfig {
                step_prefix: Some("0. ".to_owned()),
                stop_word_done: "1.".to_owned(),
                stop_word_no_result: Some("None".to_owned()),
                grammar: NoneGrammar::default().wrap(),
                ..StepConfig::default()
            };
            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut base_req).await?;
            if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                if response.contains("none") || response.contains("None") {
                    return Ok(assigned_terminal_tags);
                }
            } else {
                return Ok(assigned_terminal_tags);
            }

            let mut immediate_child_tags = parent_tag.get_tags();
            for i in 1..parent_tag.tags.len() {
                let stop_word: Option<String> = if i == 1 {
                    None
                } else {
                    Some("No additional categories".to_string())
                };
                let step_config = StepConfig {
                    stop_word_no_result: stop_word.clone(),
                    stop_word_done: "\n".to_owned(),
                    grammar: Self::list_grammar(i, &immediate_child_tags, stop_word),
                    ..StepConfig::default()
                };
                flow.last_round()?.add_inference_step(&step_config);
                flow.last_round()?.run_next_step(&mut base_req).await?;
                match flow.last_round()?.last_step()?.primitive_result() {
                    Some(result_string) => {
                        if let Some(potential_tag) = parent_tag.get_tag(&result_string) {
                            immediate_child_tags
                                .retain(|x| x.tag_name() != potential_tag.tag_name());

                            flow.last_round()?.drop_last_step()?;
                            flow.last_round()?.add_guidance_step(
                                &StepConfig::default(),
                                format!("{i}. {}", potential_tag.tag_path()),
                            );
                            flow.last_round()?.run_next_step(&mut base_req).await?;
                            if potential_tag.tags.is_empty() {
                                self.assigned_terminal_tags.push(potential_tag.clone());
                                assigned_terminal_tags.push(potential_tag.clone());
                            } else {
                                let mut new_flow = flow.clone();
                                let mut new_base_req = base_req.clone();
                                new_flow.last_round()?.close_round(&mut new_base_req)?;

                                let tags = self
                                    .list(potential_tag.clone(), new_flow, new_base_req)
                                    .await?;

                                flow.last_round()?.drop_last_step()?;

                                let tag_names = if tags.is_empty() {
                                    format!("{i}. {}::other\n", potential_tag.tag_path())
                                } else if tags.len() == 1 {
                                    format!("{i}. {}\n", tags[0].tag_path())
                                } else {
                                    format!(
                                        "{i}. {}::{{{}}}\n",
                                        potential_tag.tag_path(),
                                        tags.iter()
                                            .map(|tag| tag.tag_name())
                                            .collect::<Vec<String>>()
                                            .join(", ")
                                    )
                                };
                                flow.last_round()?
                                    .add_guidance_step(&StepConfig::default(), tag_names);
                                flow.last_round()?.run_next_step(&mut base_req).await?;
                            }
                        } else {
                            crate::bail!("Tag not found: {}", result_string);
                        }
                    }
                    None => {
                        break;
                    }
                };
                // if !self.assigned_terminal_tags.is_empty() {
                //     break;
                // }
            }

            Ok(assigned_terminal_tags)
        })
    }

    fn list_grammar(i: usize, child_tags: &Vec<&Tag>, stop_word: Option<String>) -> Grammar {
        let paths = child_tags
            .iter()
            .map(|tag| tag.tag_name())
            .collect::<Vec<String>>();
        let mut range = String::new();
        for path in paths {
            if range.is_empty() {
                range.push_str(&format!("( "));
            } else {
                range.push_str(&format!(" | "));
            }
            range.push_str(&format!("\"{i}. {path}\""));
        }
        if let Some(stop_word) = stop_word {
            range.push_str(&format!(" | \"{stop_word}\" )"));
        } else {
            range.push_str(" )");
        }
        let list_grammar_string = indoc::formatdoc! {"
        root ::= {range} \"\n\"
        ",
        };
        CustomGrammar::default()
            .custom_grammar(list_grammar_string)
            .wrap()
    }

    fn refine_instructions_prompt(&self) -> String {
        indoc::formatdoc! {"
        {}
        
        Text:
        '{}'

        1. Classifying: State the entity from the 'text' that is being classified.

        2. '{}' Definition: Explain what is being classified.

        3. Relevant details: List the elements of the input 'text' that are useful and relevant for classification.

        4. Specialized instructions: Refine the instructions into a single sentence guide specific to the input 'text'.

        5. Discuss potential categories that apply to the classification.
        ",
        self.criteria.instructions,
        self.context_text,
        self.entity,
        }
    }

    fn reason_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Determine if the '{}' classification category applies to the 'text' using the instructions:
        {}

        Category Criteria:
        {}

        Child Category Categories:
        {}

        1. Distill the instructions and {}'s critera into a an 'Is applicable if' sentence specialized for this classification.
        
        2. State any aspects, characteristics, and details that are relevent.
        
        3. Discuss if any or none of '{}'s' children categories apply.
        
        4. List which children categories, if any, apply. If none apply, say 'Applicable children categories: none'
        ",
        tag.tag_name(),
        self.refined_instructions,
        tag.format_tag_criteria(&self.entity),
        tag.display_all_tags_with_nested_paths(),
        tag.tag_name(),
        tag.tag_name(),
        }
    }

    fn list_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        '{}' Immediate Child Categories:
        {}

        0. Discuss which, if any, categories apply. If after discussing the options, no categories apply, say 'None applicable'.

        1. The most applicable immediate child category.

        2. The next most applicable immediate child category. If no others apply say 'No additional categories' to stop.
        ",
        tag.tag_name(),
        tag.display_child_tags(),
        }
    }
}

impl<'a> std::fmt::Display for LinearEntityTagger<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LinearEntityTagger:")?;
        crate::i_nln(f, format_args!("entity: {}", self.entity))?;
        crate::i_nln(f, format_args!("context_text: {}", self.context_text))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.duration))?;
        for tag in &self.assigned_terminal_tags {
            crate::i_ln(f, format_args!("{}", tag))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::workflows::classify::hierarchical_tag_system::TagCollection;

    use super::*;
    use crate::*;

    #[tokio::test]
    #[ignore]
    pub async fn test_one() -> crate::Result<()> {
        let llm_client: LlmClient = LlmClient::llama_cpp().stable_lm2_12b_chat().init().await?;

        let tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        let critera = Critera {
            entity_definition: "The source - the place or thing where the sample was collected from and what is to be classified.".to_owned(),
            instructions: "Apply classfication labels to the source described in the 'text'. The source is the place or thing where a sample of a microbial organism was collected from. Our goal is to apply labels from our classifiction system using specifically mentioned details from the 'text'.".to_owned(),
        };

        let mut req = LinearEntityTagger::new(
            &llm_client,
            "Gryllus bimaculatus",
            "Edible insect Gryllus bimaculatus (Pet Feed Store)",
            &critera,
            tag_collection.get_root_tag()?,
        );
        req.run().await?;
        println!("{}", req);

        Ok(())
    }
}
