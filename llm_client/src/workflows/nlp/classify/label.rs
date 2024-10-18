use core::panic;

use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;

use super::hierarchy::Tag;

pub struct LabelEntity {
    pub base_req: CompletionRequest,
    pub subject: String,
    pub content: String,
    pub tags: Tag,
    pub assigned_tags: Tag,
    pub flow: CascadeFlow,
    pub criteria: String,
}

impl LabelEntity {
    pub fn new(
        base_req: CompletionRequest,
        subject: String,
        content: String,
        criteria: String,
        tags: Tag,
    ) -> Self {
        Self {
            base_req,
            subject,
            content,
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
            let task = self.prompt(&parent_tag);
            let round = self.flow.new_round(task);
            round.open_round(&mut self.base_req)?;
            round.step_separator = None;
            let is_root = parent_tag.name.is_none();
            if is_root {
                let step_config = StepConfig {
                    step_prefix: Some(
                        "Thinking out loud about which root category(ies) apply... ".to_owned(),
                    ),
                    stop_word_done: " No additional relevant categories.".to_owned(),
                    grammar: TextPrimitive::default().text_token_length(200).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
            } else {
                let step_config = StepConfig {
                    step_prefix: Some(
                        "Thinking out loud about which category(ies) apply... ".to_owned(),
                    ),
                    stop_word_done: "The relevant  ".to_owned(),
                    grammar: TextPrimitive::default().text_token_length(100).grammar(),
                    ..StepConfig::default()
                };
                round.add_inference_step(&step_config);
                round.run_next_step(&mut self.base_req).await?;
            };

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
                    step_config.step_prefix = Some("The relevant category(ies) are:\n".to_owned());
                    step_config.stop_word_no_result = Some("None of the above.".to_owned())
                } else {
                    step_config.step_prefix = None;
                    step_config.step_prefix = Some("\n".to_owned());
                    step_config.stop_word_no_result =
                        Some("No additional relevant categories.".to_owned());
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

    fn prompt(&self, parent_tag: &Tag) -> String {
        if let Some(name) = &parent_tag.name {
            indoc::formatdoc! {"
            To determine which categories to apply use this criteria:
            '{}'
            The sample text containing the entity: '{}'
            The entity: '{}'
            Of these categories from the '{name}' category:
            ```
            {}
            ```
            Start by discussing which categor(ies) apply, and conclude with a newline seperate list of the best choices like:
            \"
            The relevant category(ies) are:
            category1
            category2
            etc.
            \"
            If none of the categories apply, say 'None of the above.' After selecting all relevant categories, conclude with 'No additional relevant categories.'
            ",
            self.criteria,
            self.content,
            self.subject,
            parent_tag.display_child_tags()
            }
        } else {
            indoc::formatdoc! {"
            To determine which categories to apply use this criteria:
            '{}'
            All available categories and their categories:
            ```
            {}
            ```
            The sample text containing the entity: '{}'
            The entity: '{}'
            Which root categories apply or have sub-categories that apply?
            Root categories:
            ```
            {}
            ```
            Briefly discuss which sub-categories(ies) apply, and conclude with a newline seperate list of the applicable root categories like:
            \"
            The relevant category(ies) are:
            category1
            category2
            etc.
            \"
            If none of the categories apply, say 'None of the above.' After selecting all relevant categories, conclude with 'No additional relevant categories.'
            ",
                self.criteria,
                parent_tag.display_all_tags_with_paths(),
                self.subject,
                self.content,
                parent_tag.display_child_tags()
            }
        }
    }
}

impl std::fmt::Display for LabelEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LabelEntity:")?;
        crate::i_nln(f, format_args!("subject: {}", self.subject))?;
        crate::i_nln(f, format_args!("content: {}", self.content))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        crate::i_nln(f, format_args!("assigned_tags: {}", self.assigned_tags))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        workflows::nlp::classify::{hierarchy::Tag, label::LabelEntity},
        LlmClient,
    };

    #[tokio::test]
    #[ignore]
    pub async fn test_one() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp()
            .mistral_nemo_instruct2407()
            .init()
            .await?;

        let subject = "Gryllus bimaculatus".to_owned();
        let content = "Edible insect Gryllus bimaculatus (Pet Feed Store)".to_owned();
        let criteria = "We are applying labels used to filter where a microbial organism was collected from. The entity indicates the object or location from where it was collected, and the content may provide additional information. Only apply labels that apply to what or where the sample was collected from.".to_owned();
        let entity = llm_client.nlp().classify().entity(&content);
        let req = LabelEntity::new(
            entity.base_req.clone(),
            subject,
            content,
            criteria,
            Tag::new_collection_from_text_file("/workspaces/test/bacdive_hierarchy.txt", ":"),
        );
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);

        Ok(())
    }

    const CASES: &[(&str, &str)] = &[
        ("Ciliate: Metopus sp. strain SALT15A", "ciliate"),
        ("Coastal soil sample", "soil"),
        ("Edible insect Gryllus bimaculatus (Pet Feed Store)", "insect"),
        ("Public spring water", "water"),
        ("River snow from South Saskatchewan River", "snow"),
        ("Tara packed so many boxes that she ran out of tape, and had to go to the store to buy more. Then she made grilled cheese sandwiches for lunch. She did a lot of things. She did too much.", "tara"),
        ("A green turtle on a log in a mountain lake.", "turtle"),
        (
            "Green turtle on log\nSunlight warms her emerald shell\nStillness all around",
            "turtle",
        ),
    ];
    use crate::prelude::*;

    #[tokio::test]
    #[ignore]
    pub async fn test_cases() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().llama3_2_3b_instruct().init().await?;

        for (case, _) in CASES {
            let entity = llm_client.nlp().classify().entity(case).run().await?;
            let subject = entity.subject.as_ref().unwrap().to_owned();
            let content = entity.content.to_owned();
            let criteria = "We are applying labels used to filter where a sample was collected from. The entity indicates the object or location from where it was collected, and the content may provide additional information.".to_owned();
            let req = LabelEntity::new(
                entity.base_req.clone(),
                subject,
                content,
                criteria,
                Tag::new_collection_from_text_file("/workspaces/test/bacdive_hierarchy.txt", ":"),
            );
            let entity = req.run().await?;
            println!("{}", entity.flow);
            println!("{}", entity);
            break;
        }

        Ok(())
    }
}
