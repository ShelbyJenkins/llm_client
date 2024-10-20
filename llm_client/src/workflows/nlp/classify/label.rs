use crate::components::cascade::{CascadeFlow, CascadeRound};
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;

use super::hierarchical_tag_system::{tag, Tag};

pub struct LabelEntity {
    pub base_req: CompletionRequest,
    pub entity: String,
    pub context_text: String,
    pub tags: Tag,
    pub assigned_tags: Tag,
    pub flow: CascadeFlow,
    pub criteria: String,
}

impl LabelEntity {
    pub fn new(
        base_req: CompletionRequest,
        entity: String,
        context_text: String,
        criteria: String,
        tags: Tag,
    ) -> Self {
        Self {
            base_req,
            entity,
            context_text,
            criteria,
            tags,
            assigned_tags: Tag::new(),
            flow: CascadeFlow::new("LabelEntity"),
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        self.run_cascade().await?;
        // let initial_tags = self.tags.clone();
        // let mut count = 1;
        // while count <= self.base_req.config.retry_after_fail_n_times {
        //     match self.run_cascade().await {
        //         Ok(_) => break,
        //         Err(e) => {
        //             self.base_req.reset_completion_request();
        //             self.tags = initial_tags.clone();
        //             self.assigned_tags = Tag::new();
        //             self.flow = CascadeFlow::new("LabelEntity");
        //             count += 1;
        //             if count == self.base_req.config.retry_after_fail_n_times {
        //                 crate::bail!("Failed to classify entity after {} attempts: {}", count, e);
        //             }
        //         }
        //     }
        // }
        Ok(self)
    }

    async fn run_cascade(&mut self) -> crate::Result<()> {
        self.flow.open_cascade();

        let initial_tags = self.list_potential_tags().await?;
        let initial_tags = Self::deduplicate_tags(initial_tags);
        let (mut initial_tags, mut potential_exact_tags) =
            Self::seperate_parent_and_exact_tags(initial_tags, Vec::new());

        let mut potential_parent_tags: Vec<Tag> = Vec::new();
        for tag in &mut initial_tags {
            let additional_tags = self.list_additional_tags(tag).await?;
            let (new_potential_parent_tags, new_potential_exact_tags) =
                Self::seperate_parent_and_exact_tags(additional_tags, Vec::new());
            let new_potential_exact_tags = Self::deduplicate_tags(new_potential_exact_tags);
            let new_potential_parent_tags = Self::deduplicate_tags(new_potential_parent_tags);
            potential_exact_tags.extend(new_potential_exact_tags);
            potential_parent_tags.extend(new_potential_parent_tags);
        }

        let potential_exact_tags = Self::deduplicate_tags(potential_exact_tags);
        let potential_parent_tags = Self::deduplicate_tags(potential_parent_tags);

        for tag in potential_parent_tags {
            println!("Parent Tag childred: {:?}", tag.get_tag_names());
        }
        for tag in &potential_exact_tags {
            println!("Exact Tags: {}", tag.tag_name());
        }
        self.evaluate_exact_tags(potential_exact_tags).await?;

        self.flow.close_cascade()?;
        Ok(())
    }

    async fn list_potential_tags(&mut self) -> crate::Result<Vec<Tag>> {
        self.flow
            .new_round(self.consider_potential_prompt(&self.tags));
        self.flow.last_round()?.open_round(&mut self.base_req)?;
        let step_config = StepConfig {
            step_prefix: Some(format!(
                "Classification Categories applicable to '{}': ",
                self.entity
            )),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow.last_round()?.close_round(&mut self.base_req)?;

        self.flow.new_round(self.list_potential_prompt(&self.tags));
        self.flow.last_round()?.open_round(&mut self.base_req)?;
        let mut grammar: TextListPrimitive = TextListPrimitive::default();
        grammar.max_count(5).item_prefix("root::");
        let step_config = StepConfig {
            stop_word_done: "No additional classifications".to_owned(),
            grammar: grammar.grammar(),
            ..StepConfig::default()
        };

        let mut first_potential_tags = self.list_tags(&step_config, &grammar).await?;
        let potentential_tags = if first_potential_tags.len() < 2 {
            first_potential_tags.extend(self.list_tags(&step_config, &grammar).await?);
            first_potential_tags
        } else {
            first_potential_tags
        };

        self.flow.last_round()?.close_round(&mut self.base_req)?;
        Ok(potentential_tags)
    }

    async fn list_additional_tags(&mut self, parent_tag: &Tag) -> crate::Result<Vec<Tag>> {
        self.flow
            .new_round(self.list_additional_tags_prompt(&parent_tag));
        self.flow.last_round()?.open_round(&mut self.base_req)?;
        let mut grammar = TextListPrimitive::default();
        grammar
            .max_count(4)
            .item_prefix(format!("root::{}", parent_tag.full_path.as_ref().unwrap()));

        let step_config = StepConfig {
            stop_word_done: "No additional classifications".to_owned(),
            grammar: grammar.grammar(),
            ..StepConfig::default()
        };
        let potentential_tags = self.list_tags(&step_config, &grammar).await?;
        self.flow.last_round()?.close_round(&mut self.base_req)?;
        Ok(potentential_tags)
    }

    async fn list_tags(
        &mut self,
        step_config: &StepConfig,
        grammar: &TextListPrimitive,
    ) -> crate::Result<Vec<Tag>> {
        let mut potentential_tags: Vec<Tag> = Vec::new();
        let round = self.flow.last_round()?;
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;
        match round.last_step()?.primitive_result() {
            Some(result_string) => {
                let tag_names = grammar.parse_to_primitive(&result_string)?;
                for tag_name_result in tag_names {
                    if let Some(tag) = self.tags.get_tag(&tag_name_result) {
                        potentential_tags.push(tag.clone());
                    } else {
                        println!("{tag_name_result} not found")
                    }
                }
            }
            None => {
                crate::bail!("No tags provided.");
            }
        };
        Ok(potentential_tags)
    }

    async fn evaluate_exact_tags(&mut self, potential_tags: Vec<Tag>) -> crate::Result<()> {
        let mut not_applicable_tag_names: Vec<String> = Vec::new();
        self.dummy_evaluate_exact_tags(&potential_tags).await?;
        let round = self.flow.last_round()?;

        for tag in &potential_tags {
            let guidance_content = format!("{}", tag.format_tag_criteria(&self.entity));
            round.add_guidance_step(&StepConfig::default(), guidance_content);
            round.cache_next_step(&mut self.base_req).await?;

            let step_config = StepConfig {
                step_prefix: Some(format!(
                    "Explicit aspects of '{}' applicable to '{}': ",
                    self.context_text,
                    tag.tag_name()
                )),
                stop_word_done: "Applicability of classification".to_owned(),
                stop_word_no_result: Some("is not applicable".to_owned()),
                grammar: SentencesPrimitive::default().max_count(1).grammar(),
                ..StepConfig::default()
            };
            round.add_inference_step(&step_config);
            round.run_next_step(&mut self.base_req).await?;
            match round.primitive_result() {
                Some(_) => (),
                None => {
                    not_applicable_tag_names.push(tag.tag_name());
                    continue;
                }
            };
            let step_config = StepConfig {
                step_prefix: Some(format!(
                    "Applicability of classification '{}' to '{}': ",
                    tag.tag_name(),
                    self.entity,
                )),
                stop_word_no_result: Some("is not applicable".to_owned()),
                stop_word_done: "In conclusion".to_owned(),
                grammar: SentencesPrimitive::default().max_count(1).grammar(),
                ..StepConfig::default()
            };
            round.add_inference_step(&step_config);
            round.run_next_step(&mut self.base_req).await?;
            match round.primitive_result() {
                Some(_) => (),
                None => {
                    not_applicable_tag_names.push(tag.tag_name());
                    continue;
                }
            };

            let step_config = StepConfig {
                step_prefix: Some(format!("In conclusion, '{}' ", tag.tag_name(),)),
                grammar: ExactStringPrimitive::default()
                    .add_strings_to_allowed(&["is applicable", "is not applicable"])
                    .grammar(),
                cache_prompt: false,
                ..StepConfig::default()
            };
            round.add_inference_step(&step_config);
            round.run_next_step(&mut self.base_req).await?;
            let guidance_content = match round.primitive_result().as_deref() {
                Some("is applicable") => {
                    format!(
                        "In conclusion, '{}' is applicable to '{}'.",
                        tag.tag_name(),
                        self.entity
                    )
                }
                Some("is not applicable") => {
                    not_applicable_tag_names.push(tag.tag_name());
                    format!(
                        "In conclusion, '{}' is not applicable to '{}'.",
                        tag.tag_name(),
                        self.entity
                    )
                }
                Some(other) => {
                    crate::bail!("Unexpected result: {}", other);
                }
                None => {
                    crate::bail!("No result for tag: {}", tag.tag_name());
                }
            };
            round.drop_last_step()?;
            round.add_guidance_step(&StepConfig::default(), guidance_content);
            round.cache_next_step(&mut self.base_req).await?;
        }
        round.close_round(&mut self.base_req)?;
        for tag in potential_tags {
            if not_applicable_tag_names.contains(&tag.tag_name()) {
                continue;
            }
            self.assigned_tags.add_tag(tag);
        }

        Ok(())
    }

    async fn dummy_evaluate_exact_tags(&mut self, potential_tags: &Vec<Tag>) -> crate::Result<()> {
        let round = self
            .flow
            .new_round(self.evaluate_exact_tags_prompt(&potential_tags));
        round.open_round(&mut self.base_req)?;
        round.step_separator = Some('\n');

        let guidance_content = format!("Classification 'example' is applicable if '{}' meets the criteria the specific or general conditions.", self.context_text);
        round.add_guidance_step(&StepConfig::default(), guidance_content);
        round.cache_next_step(&mut self.base_req).await?;

        let guidance_content = format!("Explicit aspects of '{}' applicable to 'example': A single sentence detailing the explicitly stated aspects of the entity that apply to the classification. Or, 'The classification is not applicable.'", self.entity);
        round.add_guidance_step(&StepConfig::default(), guidance_content);
        round.cache_next_step(&mut self.base_req).await?;

        let guidance_content = format!("Applicability of classification 'example' to '{}: ': A single sentence reasoning why the classification applies to the entity, or 'The classification is not applicable.'", self.entity);
        round.add_guidance_step(&StepConfig::default(), guidance_content);
        round.cache_next_step(&mut self.base_req).await?;

        let guidance_content = format!(
            "In conclusion, 'example' is not applicable to '{}'.",
            self.entity
        );
        round.add_guidance_step(&StepConfig::default(), guidance_content);
        round.cache_next_step(&mut self.base_req).await?;

        Ok(())
    }

    fn consider_potential_prompt(&self, root_tag: &Tag) -> String {
        indoc::formatdoc! {"
        Classification Categories:
        {}
        
        Entity:
        {}

        Context Text:
        {}

        Criteria:
        {}
        Format: 'root::path::path::tag_name' for each category selection.
        ",
        root_tag.display_all_tags_with_paths(),
        self.entity,
        self.context_text,
        self.criteria,
        }
    }

    fn list_potential_prompt(&self, _root_tag: &Tag) -> String {
        indoc::formatdoc! {"
        Entity:
        {}

        Context Text:
        {}

        Create a list of all classifications that apply to the entity. Say 'No additional classifications' when complete. List the best fitting classifications first.
        Format: 'root::path::path::tag_name' for each classification.
        ",

        self.entity,
        self.context_text,
        }
    }

    fn list_additional_tags_prompt(&self, parent_tag: &Tag) -> String {
        indoc::formatdoc! {"
        '{}' Classification Child Categories:
        {}
        
        Criteria:
        {}

        Entity:
        {}

        Context Text:
        {}

        Which, if any, child classifications of the '{}' classification category apply to the entity? Use the criteria to determine if the classification is applicable. The entity should directly relate to the classification or the context text should provide details that connect the entity to the classification. Return at least 1 or 'No additional classifications'.
        Format: 'root::path::path::tag_name' for each classification.
        ",
        parent_tag.tag_name(),
        parent_tag.display_all_tags_with_paths(),
        parent_tag.format_tag_criteria(&self.entity),
        self.entity,
        self.context_text,
        parent_tag.tag_name(),
        }
    }

    fn evaluate_exact_tags_prompt(&self, potential_tags: &Vec<Tag>) -> String {
        let mut potential_tag_names = String::new();
        for tag in potential_tags {
            potential_tag_names.push_str(&format!("'{}'\n", tag.tag_path()));
        }
        indoc::formatdoc! {"
        Potential Classification Categories:
        
        {}
        
        What classifications should be applied to the entity, '{}'? What explicit detail in the context text, '{}', or specific aspect of the entity, directly connects it to those classifications? Evaulate each classification and determine if it or any of it's children meet the criteria.

        Task Criteria:
        
        {}
        ",
        potential_tag_names,
        self.entity,
        self.context_text,
        self.criteria,
        }
    }

    fn deduplicate_tags(tags: Vec<Tag>) -> Vec<Tag> {
        let mut deduplicated_tags: Vec<Tag> = Vec::new();
        let mut seen_names: Vec<String> = Vec::new();
        for tag in tags {
            if seen_names.contains(&tag.tag_name()) {
                continue;
            }
            seen_names.push(tag.tag_name());
            deduplicated_tags.push(tag);
        }
        deduplicated_tags
    }

    fn seperate_parent_and_exact_tags(
        parent_tags: Vec<Tag>,
        exact_tags: Vec<Tag>,
    ) -> (Vec<Tag>, Vec<Tag>) {
        let mut new_parent_tags: Vec<Tag> = Vec::new();
        let mut new_exact_tags: Vec<Tag> = Vec::new();
        for tag in parent_tags {
            if tag.get_tags().len() > 0 {
                new_parent_tags.push(tag);
            } else {
                new_exact_tags.push(tag);
            }
        }
        for tag in exact_tags {
            if tag.get_tags().len() > 0 {
                new_parent_tags.push(tag);
            } else {
                new_exact_tags.push(tag);
            }
        }
        (new_parent_tags, new_exact_tags)
    }
}

impl std::fmt::Display for LabelEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LabelEntity:")?;
        crate::i_nln(f, format_args!("entity: {}", self.entity))?;
        crate::i_nln(f, format_args!("context_text: {}", self.context_text))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        crate::i_nln(f, format_args!("assigned_tags: {}", self.assigned_tags))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use workflows::nlp::classify::{
        hierarchical_tag_system::tag, subject_of_text::ClassifySubjectOfText,
    };

    use crate::workflows::nlp::classify::hierarchical_tag_system::TagCollection;

    use super::*;
    use crate::*;

    fn criteria() -> String {
        indoc::formatdoc! {"
            Apply classification labels to the source of a microbial organism.
            The entity is where or what the sample was collected from.
            The context text provides additional details like the environment, location, or other aspects of the collection source.
            "}
    }

    #[tokio::test]
    #[ignore]
    pub async fn test_one() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().llama3_1_8b_instruct().init().await?;

        let subject = "Gryllus bimaculatus".to_owned();
        let context_text = "Edible insect Gryllus bimaculatus (Pet Feed Store)".to_owned();
        let tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        let req = LabelEntity::new(
            CompletionRequest::new(llm_client.backend.clone()),
            subject,
            context_text,
            criteria(),
            tag_collection.get_root_tag()?,
        );
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    pub async fn test_full_workflow() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp()
            .mistral_nemo_instruct2407()
            .init()
            .await?;

        let mut tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();

        // Run this once!
        tag_collection
            .populate_descriptions(&llm_client, &criteria())
            .await?;

        let entity_classification = ClassifySubjectOfText::new(
            CompletionRequest::new(llm_client.backend.clone()),
            "Edible insect Gryllus bimaculatus (Pet Feed Store)",
        )
        .run()
        .await?;
        let entity = entity_classification.subject.unwrap().to_owned();
        let context_text = entity_classification.content.to_owned();

        let req = LabelEntity::new(
            CompletionRequest::new(llm_client.backend.clone()),
            entity,
            context_text,
            criteria(),
            tag_collection.get_root_tag()?,
        );
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);

        Ok(())
    }

    const CASES: &[(&str, &str)] = &[
        ("Ciliate: Metopus sp. strain SALT15A", "ciliate"),
        ("Coastal soil sample", "soil"),
        (
            "Edible insect Gryllus bimaculatus (Pet Feed Store)",
            "insect",
        ),
        ("Public spring water", "water"),
        ("River snow from South Saskatchewan River", "snow"),
        ("A green turtle on a log in a mountain lake.", "turtle"),
        (
            "Green turtle on log\nSunlight warms her emerald shell\nStillness all around",
            "turtle",
        ),
    ];

    #[tokio::test]
    #[ignore]
    pub async fn test_cases() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp()
            .mistral_nemo_instruct2407()
            .init()
            .await?;
        let tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()
            .unwrap();
        for (case, _) in CASES {
            let entity = llm_client.nlp().classify().entity(case).run().await?;
            let subject = entity.subject.as_ref().unwrap().to_owned();
            let context_text = entity.content.to_owned();

            let req = LabelEntity::new(
                entity.base_req.clone(),
                subject,
                context_text,
                criteria(),
                tag_collection.get_root_tag()?,
            );
            let entity = req.run().await?;
            println!("{}", entity.flow);
            println!("{}", entity);
            break;
        }

        Ok(())
    }
}
