use crate::components::cascade::step::StepConfig;
use crate::components::cascade::CascadeFlow;

use crate::components::grammar::*;
use crate::LlmClient;

use super::*;

use super::tag::Tag;

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

pub struct HierarchicalEntityTagger {
    backend: std::sync::Arc<LlmBackend>,
    pub entity: String,
    pub input_text: String,
    pub criteria: Critera,
    pub refined_instructions: String,
    pub tag_collection: Tag,
    pub assigned_terminal_tags: Vec<Tag>,
    pub start_time: std::time::Instant,
    pub duration: std::time::Duration,
}

impl HierarchicalEntityTagger {
    pub fn new(
        llm_client: &LlmClient,
        entity: &str,
        input_text: &str,
        criteria: &Critera,
        tag_collection: Tag,
    ) -> Self {
        Self {
            backend: llm_client.backend.clone(),
            entity: entity.to_owned(),
            input_text: input_text.to_owned(),
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
        let mut base_req = CompletionRequest::new(self.backend.clone());
        let flow: CascadeFlow = self.refine_instructions(&mut base_req).await?;

        self.evaluate_root_tags(tag_collection, flow, base_req)
            .await?;
        self.duration = self.start_time.elapsed();
        Ok(())
    }

    fn refine_instructions_prompt(&self) -> String {
        indoc::formatdoc! {"
        Instructions:
        '{}'
        
        Input Text:
        '{}'

        1. Identify entity: State the entity being classified.

        2. Entity definition: Describe what is being classified.

        3. Evaluate relevant details: Enumerate the elements of the 'input text' that may be useful information for classification.

        4. Refine instructions: Distill into a single sentence criteria specific to applying classification labels to the entity.

        5. Discuss potential categories that apply to the classification.
        ",
        self.criteria.instructions,
        self.input_text,
        }
    }

    async fn refine_instructions(
        &mut self,
        base_req: &mut CompletionRequest,
    ) -> crate::Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("HierarchicalEntityTagger");
        flow.open_cascade();
        flow.new_round(self.refine_instructions_prompt());
        flow.last_round()?.open_round(base_req)?;

        flow.last_round()?.add_guidance_step(
            &StepConfig::default(),
            format!("1.Identify entity: We are classifying '{}'.\n", self.entity),
        );
        flow.last_round()?.run_next_step(base_req).await?;
        flow.last_round()?.add_guidance_step(
            &StepConfig::default(),
            format!(
                "2. Entity definition: {} '{}'\n",
                self.criteria.entity_definition, self.entity,
            ),
        );
        flow.last_round()?.run_next_step(base_req).await?;
        let step_config = StepConfig {
            step_prefix: Some("3. Evaluate relevant details:".to_owned()),
            stop_word_done: "4.".to_owned(),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(base_req).await?;
        let step_config = StepConfig {
            step_prefix: Some("4. Refine instructions:".to_owned()),
            stop_word_done: "5.".to_owned(),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(base_req).await?;
        if let Some(refined_instructions) = flow.last_round()?.last_step()?.primitive_result() {
            self.refined_instructions = refined_instructions;
        } else {
            crate::bail!("No refined instructions");
        }
        flow.last_round()?.close_round(base_req)?;
        Ok(flow)
    }

    fn evaluate_root_tags_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Determine if the '{}' classification category applies to the 'input text' using the instructions:
        {} {}

        '{}' Criteria:
        '{}'

        '{}' Child Category Paths:
        {}

        1. Distill the instructions and {}'s critera into a an 'Is applicable if' sentence specialized for this classification.
        
        2. State any aspects, characteristics, or details from the 'input text' that are directly relevant to this category.
        
        3. Discuss if any of '{}'s' children categories apply. If none apply, say 'None apply'.
        
        4. List which children categories, if any, apply using their full path. If none apply, say 'None apply'.
        ",
        tag.tag_name(),
        self.criteria.instructions,
        self.refined_instructions,
        tag.tag_name(),
        tag.display_tag_criteria(&self.entity),
        tag.tag_name(),
        tag.display_all_tags_with_nested_paths(),
        tag.tag_name(),
        tag.tag_name(),
        }
    }

    async fn evaluate_root_tags(
        &mut self,
        parent_tag: Tag,
        initial_flow: CascadeFlow,
        initial_base_req: CompletionRequest,
    ) -> crate::Result<()> {
        for potential_tag in parent_tag.get_tags() {
            let mut flow = initial_flow.clone();
            let mut base_req = initial_base_req.clone();
            flow.new_round(self.evaluate_root_tags_prompt(&potential_tag));
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
                step_prefix: Some("2. Relevant aspects and details:".to_owned()),
                stop_word_done: "3.".to_owned(),
                grammar: NoneGrammar::default().wrap(),
                ..StepConfig::default()
            };
            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut base_req).await?;

            let step_config = StepConfig {
                step_prefix: Some("3. Category applicability:".to_owned()),
                stop_word_done: "4.".to_owned(),
                stop_word_no_result: Some("one apply".to_owned()),
                grammar: NoneGrammar::default().wrap(),
                ..StepConfig::default()
            };
            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut base_req).await?;

            if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                if response.contains("none") || response.contains("None") {
                    continue;
                }
            } else {
                continue;
            }

            let step_config = StepConfig {
                step_prefix: Some(format!(
                    "Therefore, the category '{}' or one if it's children ",
                    potential_tag.tag_name()
                )),
                grammar: ExactStringGrammar::default()
                    .add_exact_string("does apply")
                    .add_exact_string("does not apply")
                    .wrap(),
                ..StepConfig::default()
            };
            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut base_req).await?;

            if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                if response.contains("does not apply") {
                    continue;
                }
            } else {
                crate::bail!("No response in validate_terminal");
            }

            let step_config = StepConfig {
                step_prefix: Some("4. Applicable categories:".to_owned()),
                stop_word_done: "::".to_owned(),
                stop_word_no_result: Some("one apply".to_owned()),
                grammar: NoneGrammar::default().wrap(),
                ..StepConfig::default()
            };
            flow.last_round()?.add_inference_step(&step_config);
            flow.last_round()?.run_next_step(&mut base_req).await?;
            flow.last_round()?.drop_last_step()?;
            flow.last_round()?.close_round(&mut base_req)?;

            if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
                if !response.contains("none") && !response.contains("None") {
                    if potential_tag.tags.is_empty() {
                        self.assigned_terminal_tags.push(potential_tag.clone());
                    } else {
                        let new_flow = flow.clone();
                        let new_base_req = base_req.clone();
                        self.list_tags_recursive(potential_tag.clone(), new_flow, new_base_req)
                            .await?;
                    }
                }
            }
            // For testing purposes, we will end the loop after the first terminal tag is found.
            // if !self.assigned_terminal_tags.is_empty() {
            //     break;
            // }
        }

        Ok(())
    }

    fn list_tags_recursive_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        List immediate child categories of '{}' that apply to '{}' using the instructions:
        {}

        '{}' Immediate Child Categories:
        {}

        List applicable categories. If no categories apply, say 'None apply'. Start with the most applicable category. Say 'No additional categories' to stop. 
        ",
        tag.tag_name(),
        self.entity,
        self.refined_instructions,
        tag.tag_name(),
        tag.display_immediate_child_paths(),
        }
    }

    fn list_tags_recursive(
        &mut self,
        parent_tag: Tag,
        mut flow: CascadeFlow,
        mut base_req: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<Tag>>> + '_>> {
        Box::pin(async move {
            let mut failed_evaluations = 0;
            let mut assigned_terminal_tags = Vec::new();
            flow.new_round(self.list_tags_recursive_prompt(&parent_tag));
            flow.last_round()?.open_round(&mut base_req)?;

            let mut immediate_child_tags = parent_tag.get_tags();
            for i in 1..parent_tag.tags.len() {
                let i = i - failed_evaluations;
                let stop_word: Option<String> = if i == 1 && failed_evaluations == 0 {
                    None
                } else {
                    Some("No additional categories".to_string())
                };
                let step_config = StepConfig {
                    stop_word_no_result: stop_word.clone(),
                    stop_word_done: "\n".to_owned(),
                    grammar: Self::list_tags_recursive_grammar(
                        i,
                        &parent_tag.tag_name(),
                        &immediate_child_tags,
                        stop_word,
                    ),
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
                                let mut new_flow = flow.clone();
                                let mut new_base_req = base_req.clone();
                                new_flow.last_round()?.close_round(&mut new_base_req)?;

                                if self
                                    .validate_terminal(&potential_tag, new_flow, new_base_req)
                                    .await?
                                {
                                    self.assigned_terminal_tags.push(potential_tag.clone());
                                    assigned_terminal_tags.push(potential_tag.clone());
                                } else {
                                    flow.last_round()?.drop_last_step()?;
                                    failed_evaluations += 1;
                                }
                            } else {
                                let mut new_flow = flow.clone();
                                let mut new_base_req = base_req.clone();
                                new_flow.last_round()?.close_round(&mut new_base_req)?;

                                let tags = self
                                    .list_tags_recursive(
                                        potential_tag.clone(),
                                        new_flow,
                                        new_base_req,
                                    )
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

    fn list_tags_recursive_grammar(
        i: usize,
        parent_tag_name: &str,
        child_tags: &Vec<&Tag>,
        stop_word: Option<String>,
    ) -> Grammar {
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
            range.push_str(&format!("\"{i}. {parent_tag_name}::{path}\""));
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

    fn validate_terminal_prompt(&self, tag: &Tag) -> String {
        indoc::formatdoc! {"
        Determine if the category '{}' applies to the '{}' using the instructions:
        {}

        1. Explore which details of the category apply. If after discussing, it does not apply, say 'Does not apply'. 

        2. State your determination of the category's applicability. Say 'Does not apply' or 'Category applies.'

        3. Share any relevant aspects or details from the 'input text' that are directly relevant to this category.
        ",
        tag.tag_name(),
        self.entity,
        self.refined_instructions,
        }
    }

    async fn validate_terminal(
        &mut self,
        terminal_tag: &Tag,
        mut flow: CascadeFlow,
        mut base_req: CompletionRequest,
    ) -> crate::Result<bool> {
        flow.new_round(self.validate_terminal_prompt(&terminal_tag));
        flow.last_round()?.open_round(&mut base_req)?;

        let step_config = StepConfig {
            step_prefix: Some("1. ".to_owned()),
            stop_word_done: "2.".to_owned(),
            stop_word_no_result: Some("oes not apply".to_owned()),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(&mut base_req).await?;

        if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
            if response.contains("oes not")
                || response.contains("not appl")
                || response.contains("do not")
            {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }

        let step_config = StepConfig {
            step_prefix: Some("2. Relevant aspects and details:".to_owned()),
            stop_word_done: "3.".to_owned(),
            stop_word_no_result: Some("oes not apply".to_owned()),
            grammar: NoneGrammar::default().wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(&mut base_req).await?;

        if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
            if response.contains("oes not")
                || response.contains("not appl")
                || response.contains("do not")
            {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "Therefore, the category '{}' ",
                terminal_tag.tag_name()
            )),
            grammar: ExactStringGrammar::default()
                .add_exact_string("does apply")
                .add_exact_string("does not apply")
                .wrap(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);
        flow.last_round()?.run_next_step(&mut base_req).await?;
        flow.last_round()?.close_round(&mut base_req)?;

        if let Some(response) = flow.last_round()?.last_step()?.primitive_result() {
            if response.contains("does not apply") {
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            crate::bail!("No response in validate_terminal");
        }
    }
}

impl std::fmt::Display for HierarchicalEntityTagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "HierarchicalEntityTagger:")?;
        crate::i_nln(f, format_args!("entity: {}", self.entity))?;
        crate::i_nln(f, format_args!("input_text: {}", self.input_text))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.duration))?;
        for tag in &self.assigned_terminal_tags {
            crate::i_ln(f, format_args!("{}", tag))?;
        }
        Ok(())
    }
}
