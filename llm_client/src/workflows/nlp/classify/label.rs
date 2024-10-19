use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;

use super::hierarchical_tag_system::Tag;

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
        let tag_collection = self.tags.clone();
        let assigned_tag = self.evaluate_tag_set(tag_collection).await?;

        self.assigned_tags = assigned_tag;
        self.flow.close_cascade()?;
        Ok(())
    }

    fn evaluate_tag_set<'a>(
        &'a mut self,
        mut parent_tag: Tag,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Tag>> + 'a>> {
        Box::pin(async move {
            let mut assigned_child_tags: Vec<Tag> = Vec::new();

            let round = self.flow.new_round(self.cot_prompt(&parent_tag));
            round.open_round(&mut self.base_req)?;
            round.step_separator = None;
            let is_root = parent_tag.name.is_none();
            if is_root {
                let step_config = StepConfig {
                    step_prefix: Some("The entity ".to_owned()),
                    grammar: TextPrimitive::default().text_token_length(800).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
            } else {
                let step_config = StepConfig {
                    step_prefix: Some("The entity ".to_owned()),
                    grammar: TextPrimitive::default().text_token_length(100).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
            };
            round.close_round(&mut self.base_req)?;

            let round = self.flow.new_round(self.list_prompt(&parent_tag));
            round.open_round(&mut self.base_req)?;
            round.step_separator = None;
            let length = parent_tag.get_tags().len();
            for i in 1..=length {
                let mut step_config = StepConfig {
                    grammar: ExactStringPrimitive::default()
                        .add_strings_to_allowed(&parent_tag.get_tag_names())
                        .grammar(),
                    stop_word_done: "\n".to_owned(),
                    ..StepConfig::default()
                };
                if i == 1 {
                    step_config.step_prefix =
                        Some("The applicable classifications are:\n".to_owned());
                    step_config.stop_word_no_result = Some("None of the above.".to_owned())
                } else {
                    // step_config.step_prefix = None;
                    step_config.step_prefix = Some("\n".to_owned());
                    step_config.stop_word_no_result =
                        Some("No additional classifications.".to_owned());
                };

                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;

                match round.primitive_result() {
                    Some(tag_name) => {
                        let tag = parent_tag.remove_tag(&tag_name)?;
                        assigned_child_tags.push(tag);
                    }
                    None => {
                        break;
                    }
                };
            }
            round.close_round(&mut self.base_req)?;

            parent_tag.clear_tags();
            for tag in assigned_child_tags {
                if tag.get_tags().len() > 0 {
                    let tag = self.evaluate_tag_set(tag).await?;
                    parent_tag.add_tag(tag);
                } else {
                    parent_tag.add_tag(tag);
                }
            }
            Ok(parent_tag)
        })
    }

    fn cot_prompt(&self, parent_tag: &Tag) -> String {
        if let Some(name) = &parent_tag.name {
            indoc::formatdoc! {"
            In a single paragraph, explain if any child classifications of the '{name}' classification apply to the entity '{}'.

            Classifications:
            ```
            {}
            ```

            The entity is, '{}' and additional details are provided in the context text, '{}'. Are any of the parent classifications applicable? If so, which?
            ",
            self.entity,
            parent_tag.display_child_tag_descriptions(),
            self.entity,
            self.context_text,
            }
        } else {
            indoc::formatdoc! {"
            List all relevant root classifications whose children classifications apply to the entity '{}'.

            Criteria: 
            '{}'
     
            Classifications:
            ```
            {}
            ```

            The entity is, '{}' and additional details are provided in the context text, '{}'. Are any of the child classifications of the root classifications applicable? If so, which? What are their parent classifications.
            ",
            self.entity,
            self.criteria,
            parent_tag.display_child_tag_descriptions(),
            self.entity,
            self.context_text,
            }
        }
    }

    fn list_prompt(&self, parent_tag: &Tag) -> String {
        if let Some(name) = &parent_tag.name {
            indoc::formatdoc! {"
                Build a newline seperated list of the most relevant classifications for the entity '{}'.
                Use the child classifications of the '{name}' classification:
                ```
                {}
                ```
                Format:
                \"
                The applicable classifications are:
                classification1
                classification2
                etc.
                No additional classifications.
                \"

                Or if none of the classifications apply:
                \"
                The applicable classifications are:
                None of the above.
                \"
                ",
                self.entity,
                parent_tag.display_child_tags()
            }
        } else {
            indoc::formatdoc! {"
                Build a newline seperated list of the root classifications for the entity '{}'. Include root classifications whose child classifications apply.
                Use the root classifications:
                ```
                {}
                ```
                Format:
                \"
                The applicable classifications are:
                classification1
                classification2
                etc.
                No additional classifications.
                \"
                
                Or if none of the classifications apply:
                \"
                The applicable classifications are:
                None of the above.
                \"
                ",
                self.entity,
                parent_tag.display_child_tags()
            }
        }
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
    pub async fn test_one() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp()
            .mistral_nemo_instruct2407()
            .init()
            .await?;

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
