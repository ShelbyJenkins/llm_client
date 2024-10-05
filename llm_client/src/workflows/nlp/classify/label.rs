use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;

use super::entity::ClassifyEntity;
use super::hierarchy::TagCollection;

pub struct LabelEntity {
    pub base_req: CompletionRequest,
    pub entity: ClassifyEntity,
    pub tags: TagCollection,
    pub assigned_tags: TagCollection,
    pub flow: CascadeFlow,
}

impl LabelEntity {
    pub fn new(base_req: CompletionRequest, entity: ClassifyEntity, tags: TagCollection) -> Self {
        Self {
            base_req,
            entity,
            tags,
            assigned_tags: TagCollection::new(),
            flow: CascadeFlow::new("LabelEntity"),
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        let mut count = 1;
        let initial_tags = self.tags.clone();
        while count <= self.base_req.config.retry_after_fail_n_times {
            match self.run_cascade().await {
                Ok(_) => break,
                Err(e) => {
                    self.base_req.reset_completion_request();
                    self.tags = initial_tags.clone();
                    self.assigned_tags = TagCollection::new();
                    self.flow = CascadeFlow::new("LabelEntity");
                    count += 1;
                    if count == self.base_req.config.retry_after_fail_n_times {
                        crate::bail!("Failed to classify entity after {} attempts: {}", count, e);
                    }
                }
            }
        }
        // println!("{}", self.flow);
        Ok(self)
    }

    fn build_tag_list(&self) -> String {
        let mut list = String::new();
        for name in self.tags.get_tag_names() {
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

    async fn run_cascade(&mut self) -> crate::Result<()> {
        let task = indoc::formatdoc! {"
            Assign the most accurate category or categories to the an entity.
            Guidelines:
            Consider each of the provided categories and determine which ones most accurately and comprehensively describe the entity.
            When multiple categories are applicable, list them in order of relevance, starting with the most applicable.
            If none of the categories are relevant to the entity, state \"None of the above.\"
            Process:
            Start by analyzing each relevant category in relation to the entity: \"Thinking out loud about which categories apply...\".
            After discussing all options, propose the best candidate(s): \"Therefore, the most relevant category/categories are... category_1 because reasons. Also, category_2 because reasons. Also, category_3 because reasons.\".
            Finally, state restate the best category or categories as a list: \"Assigned category/categories: category_1, category_2, category_3, no additional categories apply.\".
            Categories: {}. 
            Entity type: '{}'
            ",
            self.build_tag_list(),
            self.entity.entity_type.as_deref().unwrap(),
        };

        let round = self.flow.new_round(task);
        round.open_round(&mut self.base_req)?;

        let step_config = StepConfig {
            step_prefix: Some("Thinking out loud about which categories apply...".to_owned()),
            stop_word_done: "Therefore, the most relevant category/categories are".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;

        let step_config = StepConfig {
            step_prefix: Some("Therefore, the most relevant category/categories are...".to_owned()),
            stop_word_done: "Assigned category/categories:".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;

        for i in 1..=self.tags.get_tag_names().len() {
            let step_config = if i == 1 {
                StepConfig {
                    step_prefix: Some("Assigned category/categories:".to_owned()),
                    stop_word_no_result: Some("None of the above.".to_owned()),
                    grammar: self.tags.grammar(),
                    ..StepConfig::default()
                }
            } else {
                StepConfig {
                    step_prefix: None,
                    stop_word_no_result: Some("no additional categories apply".to_owned()),
                    grammar: self.tags.grammar(),
                    ..StepConfig::default()
                }
            };

            round.add_inference_step(&step_config);
            round.run_next_step(&mut self.base_req).await?;

            match round.primitive_result() {
                Some(tag_name) => {
                    let tag = self.tags.remove_tag(&tag_name)?;
                    self.assigned_tags.add_tag(tag);
                    round.last_step()?.set_dynamic_suffix(",");
                }
                None => {
                    round.last_step()?.set_dynamic_suffix(".");
                    break;
                }
            };
        }
        round.close_round(&mut self.base_req)?;
        self.flow.close_cascade()?;
        Ok(())
    }
}

impl std::fmt::Display for LabelEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "LabelEntity:")?;
        crate::i_nln(f, format_args!("entity: {}", self.entity))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        for tag in &self.assigned_tags.get_tag_names() {
            crate::i_nln(f, format_args!("tag: {}", tag))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        workflows::nlp::classify::{hierarchy::TagCollection, label::LabelEntity},
        LlmClient,
    };

    fn create_sample_tag_collection() -> TagCollection {
        let input = "\
terrestrial
terrestrial:soil
terrestrial:arid
terrestrial:grassland
terrestrial:cropland
terrestrial:forest
terrestrial:tundra
terrestrial:permafrost
terrestrial:wetland
terrestrial:wetland:mangrove
terrestrial:subsurface
aquatic
aquatic:lentic
aquatic:lotic
aquatic:fresh water
aquatic:fresh water:lacustrine
aquatic:fresh water:lake bed
aquatic:fresh water:stream
aquatic:fresh water:stream bed
aquatic:brackish water
aquatic:brackish water:estuary
aquatic:saline water
aquatic:briny water
aquatic:marine
aquatic:marine:sea floor
aquatic:littoral
aquatic:pelagic
aquatic:benthic
aquatic:sediment
aquatic:groundwater
aquatic:spring
aquatic:spring:hot spring
aquatic:spring:hydrothermal vent
aquatic:spring:seep
aquatic:ice
host-associated
host-associated:plant host
host-associated:plant host:rhizosphere
host-associated:plant host:phyllosphere
host-associated:plant host:plant litter
host-associated:animal host
host-associated:animal host:mammalian host
host-associated:animal host:mammalian host:human host
host-associated:animal host:coral reef
host-associated:animal host:insect host
host-associated:animal host:digestive tract
host-associated:animal host:digestive tract:mouth
host-associated:animal host:digestive tract:mouth:saliva
host-associated:ainmal host:digestive tract:stomach
host-associated:ainmal host:digestive tract:rumen
host-associated:animal host:digestive tract:intestine
host-associated:animal host:urogenital tract
host-associated:animal host:airways
host-associated:animal host:skin
air
anthropogenic
anthropogenic:mock community
anthropogenic:built environment
anthropogenic:built environment:hospital
anthropogenic:food
anthropogenic:food:fermented food
anthropogenic:food:dairy product
anthropogenic:agriculture
anthropogenic:mine
anthropogenic:waste
anthropogenic:wastewater
anthropogenic:wastewater:sewage
anthropogenic:sludge
contaminated
salinity:low salinity
salinity:medium salinity
salinity:high salinity
salinity:very high salinity
temperature:very low temperature
temperature:low temperature
temperature:medium temperature
temperature:high temperature
temperature:very high temperature
ph:acidic
ph:neutral ph range
ph:alkaline
oxygen level:aerobic
oxygen level:microaerobic
oxygen level:anaerobic
age group:infant
age group:child
age group:adolescent
age group:adult
age group:elderly
birth term:preterm birth
birth term:full term birth
    ";
        TagCollection::create_from_string(input)
    }

    const CASES: &[&str] = &[
        // "Ciliate: Metopus sp. strain SALT15A",
        // "Coastal soil sample",
        // "Edible insect Gryllus bimaculatus (Pet Feed Store)",
        // "Public spring water",
        "River Snow from South Saskatchewan River",
    ];

    #[tokio::test]
    #[ignore]
    pub async fn test() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;

        let entity = llm_client
            .nlp()
            .classify()
            .entity("A green turtle on a log in a mountain lake.");
        let entity = entity.run().await?;

        let req = LabelEntity::new(
            entity.base_req.clone(),
            entity,
            create_sample_tag_collection(),
        );

        let entity = req.run().await?;
        println!("{}", entity);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    pub async fn test_cases() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;

        for case in CASES {
            let entity = llm_client.nlp().classify().entity(case);
            let entity = entity.run().await?;
            println!("{}", entity.flow);
            let req = LabelEntity::new(
                entity.base_req.clone(),
                entity,
                create_sample_tag_collection(),
            );

            let entity = req.run().await?;
            println!("{}", entity.flow);
            println!("{}", entity);
        }

        Ok(())
    }
}
