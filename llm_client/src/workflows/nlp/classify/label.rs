use crate::components::cascade::step::StepConfig;
use crate::components::cascade::CascadeFlow;
use crate::components::grammar::{CustomGrammar, Grammar, NoneGrammar};

use llm_interface::requests::completion::CompletionRequest;

use super::hierarchical_tag_system::Tag;

pub struct LabelEntity {
    pub base_req: CompletionRequest,
    pub entity: String,
    pub context_text: String,
    pub tags: Tag,
    pub assigned_tags: Vec<Tag>,
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
            assigned_tags: Vec::new(),
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
        let inital_test = self.list_initial(&self.tags.clone()).await?;

        self.assigned_tags = inital_test;
        self.flow.close_cascade()?;
        Ok(())
    }

    fn list_initial<'a>(
        &'a mut self,
        parent_tag: &'a Tag,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<Tag>>> + 'a>> {
        Box::pin(async move {
            let mut exact_tags: Vec<Tag> = Vec::new();
            let mut potential_parent_tags: Vec<Tag> = Vec::new();
            let prompt = indoc::formatdoc! {"
            Criteria:
            {}
            
            Classification Categories:
            {}
            
            Entity:
            {}
    
            Context Text:
            {}
    
            Given the criteria, which classification categories apply to the entity described in the context text? Discuss why each classification may apply.

            1.  Classification '<category>' <reason why the category applies> 
    
            2.  Classification '<category>' <reason why the category applies> 
    
            3.  No additional classifications
            ",
            self.criteria,
            parent_tag.display_immediate_child_descriptions(&self.entity),
            self.entity,
            self.context_text,
            };

            self.flow.new_round(prompt);
            self.flow.last_round()?.open_round(&mut self.base_req)?;

            let mut immediate_child_tags = parent_tag.get_tags();
            for i in 1..parent_tag.tags.len() {
                let step_config = StepConfig {
                    step_prefix: Some(format!("{i}. ",)),
                    stop_word_done: format!("Classification",),
                    stop_word_no_result: Some("No additional classifications".to_owned()),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                self.flow.last_round()?.add_inference_step(&step_config);
                self.flow
                    .last_round()?
                    .run_next_step(&mut self.base_req)
                    .await?;
                match self.flow.last_round()?.last_step()?.primitive_result() {
                    Some(_) => (),
                    None => {
                        break;
                    }
                };
                self.flow
                    .last_round()?
                    .last_step()?
                    .set_dynamic_suffix("Classification");

                let step_config = StepConfig {
                    step_prefix: Some(format!("'",)),
                    stop_word_done: format!("'",),
                    grammar: Self::list_grammar(&immediate_child_tags),
                    ..StepConfig::default()
                };
                self.flow.last_round()?.add_inference_step(&step_config);
                self.flow
                    .last_round()?
                    .run_next_step(&mut self.base_req)
                    .await?;
                self.flow.last_round()?.last_step()?.set_dynamic_suffix("'");
                let tag_name = match self.flow.last_round()?.last_step()?.primitive_result() {
                    Some(result_string) => result_string,
                    None => {
                        break;
                    }
                };

                let step_config = StepConfig {
                    step_prefix: Some(format!("is",)),
                    stop_word_no_result: Some("not applicable".to_owned()),
                    stop_word_done: format!("\n",),
                    grammar: NoneGrammar::default().wrap(),
                    ..StepConfig::default()
                };
                self.flow.last_round()?.add_inference_step(&step_config);
                self.flow
                    .last_round()?
                    .run_next_step(&mut self.base_req)
                    .await?;

                match self.flow.last_round()?.last_step()?.primitive_result() {
                    Some(_) => {
                        if let Some(tag) = parent_tag.get_tag(&tag_name) {
                            if tag.tags.is_empty() {
                                exact_tags.push(tag.clone());
                            } else {
                                potential_parent_tags.push(tag.clone());
                            }
                            immediate_child_tags.retain(|x| x.tag_name() != tag.tag_name());
                        } else {
                            println!("{tag_name} not found")
                        }
                        self.flow
                            .last_round()?
                            .last_step()?
                            .set_dynamic_suffix("\n\n");
                    }
                    None => {
                        self.flow
                            .last_round()?
                            .last_step()?
                            .set_dynamic_suffix(".\n\n");
                        break;
                    }
                };
            }
            self.flow.last_round()?.close_round(&mut self.base_req)?;
            self.flow.rounds.clear();
            self.base_req.reset_completion_request();
            for tag in potential_parent_tags {
                let res = self.list_initial(&tag).await?;
                exact_tags.extend(res);
            }
            Ok(exact_tags)
        })
    }

    fn list_grammar(child_tags: &Vec<&Tag>) -> Grammar {
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
            range.push_str(&format!("\"{path}\""));
        }
        range.push_str(&format!(" )"));
        let list_grammar_string = indoc::formatdoc! {"
        root ::= {range} \"'\"
        ",
        };
        CustomGrammar::default()
            .custom_grammar(list_grammar_string)
            .wrap()
    }
}

impl std::fmt::Display for LabelEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LabelEntity:")?;
        crate::i_nln(f, format_args!("entity: {}", self.entity))?;
        crate::i_nln(f, format_args!("context_text: {}", self.context_text))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        for tag in &self.assigned_tags {
            crate::i_ln(f, format_args!("{}", tag))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use workflows::nlp::classify::subject_of_text::ClassifySubjectOfText;

    use crate::workflows::nlp::classify::hierarchical_tag_system::TagCollection;

    use super::*;
    use crate::*;

    fn criteria() -> String {
        indoc::formatdoc! {"
            Label the entity with classification categories.
            The entity is where or what the sample was collected from.
            The context text provides additional details like the environment, location, or other aspects of the collection source.
            "}
    }

    #[tokio::test]
    #[ignore]
    pub async fn test_one() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp()
            .llama3_1_70b_nemotron_instruct()
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
        // println!("{}", entity.flow);
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
