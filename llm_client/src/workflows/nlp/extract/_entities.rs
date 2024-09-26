use crate::{
    components::{
        base_request::{BaseLlmRequest, BaseRequestConfig, BaseRequestConfigTrait},
        cascade::{step::StepConfig, CascadeFlow},
        instruct_prompt::{InstructPrompt, InstructPromptTrait},
    },
    primitives::*,
};
use anyhow::Result;
use llm_utils::{grammar::Grammar, text_utils::extract::extract_urls};

#[derive(Clone)]
pub struct ExtractEntities {
    pub max_entity_count: u8,
    pub entity_count: Option<u32>,
    pub entity_type: String,
    pub base_req: BaseLlmRequest,
    pub instruct_prompt: InstructPrompt,
    pub results: Vec<ExtractEntity>,
    pub flow: CascadeFlow,
}

impl ExtractEntities {
    pub fn new(base_req: BaseLlmRequest) -> Self {
        ExtractEntities {
            max_entity_count: 5,
            entity_count: None,
            entity_type: "topic".to_string(),
            instruct_prompt: InstructPrompt::new(&base_req.config),
            base_req,
            results: Vec::new(),
            flow: CascadeFlow::new("ExtractEntities"),
        }
    }

    pub fn entity_type<T: Into<String>>(mut self, entity_type: T) -> Self {
        self.entity_type = entity_type.into();
        self
    }

    pub fn max_entity_count(mut self, max_entity_count: u8) -> Self {
        self.max_entity_count = max_entity_count;
        self
    }

    pub async fn run_return_result(&mut self) -> Result<ExtractEntitiesResult> {
        self.run_backend().await?;
        let flow = self.flow.clone();
        if self.results.is_empty() {
            Ok(ExtractEntitiesResult::new(flow, None, &self.entity_type))
        } else {
            Ok(ExtractEntitiesResult::new(
                flow,
                Some(self.results.clone()),
                &self.entity_type,
            ))
        }
    }

    async fn run_backend(&mut self) -> Result<()> {
        // if !self.count_entities().await? {
        //     return Ok(());
        // }
        self.set_initial().await?;
        self.run_cascade().await?;
        Ok(())
    }

    // async fn count_entities(&mut self) -> Result<bool> {
    //     let mut gen = IntegerPrimitive::default();
    //     gen.instructions().set_content(format!(
    //         "How many unique {}s are discussed in the text?",
    //         self.entity_type,
    //     ));
    //     gen.supporting_material()
    //         .set_content(&self.instruct_prompt.build_supporting_material().unwrap());
    //     gen.result_can_be_none(true);
    //     gen.upper_bound(99);
    //     let mut result = gen
    //         .decision()
    //         .reasoning_sentences(3)
    //         .conclusion_sentences(3)
    //         .best_of_n_votes(1)
    //         .return_result()
    //         .await
    //         .unwrap();
    //     let entity_count = gen.result_index_to_primitive(result.winner_index)?;
    //     self.base_req.prompt = gen.primitive_base.unwrap().base_req.prompt.clone();
    //     self.flow = result.reason_results.pop().unwrap().workflow;
    //     if let Some(entity_count) = entity_count {
    //         if entity_count > self.max_entity_count.into() {
    //             self.entity_count = Some(self.max_entity_count as u32);
    //         } else {
    //             self.entity_count = Some(entity_count);
    //         }
    //         Ok(true)
    //     } else {
    //         Ok(false)
    //     }
    // }

    async fn set_initial(&mut self) -> Result<()> {
        let entity_type = self.entity_type.clone();
        self.flow.new_round(
            format!("A 'entity' in knowledge taxonomy can be any topic, concept, or subject that serves as a focal point for organizing and categorizing information. Entities in the text will be classified. Specifically, the entity we're interested in is: {entity_type}. Analyze the given text and extract information about each '{entity_type}' mentioned. For each {entity_type}, provide:\n{entity_type} description: A concise, single sentence explanation of the {entity_type} from the information in the text.\n{entity_type} keywords: 3-5 relevant terms associated with the {entity_type}.\n{entity_type} name: A single sentence name, title, or heading for the {entity_type}.\nblog url: A unique URL for a potential blog post about this specific {entity_type}. The URL should end with a slug consisting of 2-5 words separated by hyphens, with each word relevant to the {entity_type}. Likely from the {entity_type} name. For example: https://blog.com/your-slug-here\nPlease provide an example of extracting a {entity_type} using these instructions as the input text. With no yapping.")).add_guidance_step(
            &StepConfig::default(),
            format!("{entity_type} description: This text provides instructions for classifying and extracting {entity_type}s from given text, emphasizing the identification of key characteristics and relevant information. The 'no yapping' directive encourages concise, focused responses, ensuring efficient and precise extraction of {entity_type} data.\n{entity_type} keywords: classification, extraction, analysis, {entity_type}s\n{entity_type} name: Instructions for exctraction and classification of {entity_type}. \nblog url: https://blog.com/{entity_type}-extraction-classification\n"),
        );
        self.flow
            .last_round()?
            .run_all_steps(&mut self.base_req)
            .await?;

        Ok(())
    }

    async fn description(&mut self) -> Result<Option<String>> {
        let step_prefix = format!("{} description:", self.entity_type);
        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some(step_prefix),
            grammar: SentencesPrimitive::default()
                .min_count(1)
                .max_count(2)
                .capitalize_first(true)
                .grammar(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("\n".to_owned());
        Ok(self.flow.primitive_result())
    }

    async fn keywords(&mut self) -> Result<Option<String>> {
        let step_prefix = format!("{} keywords:", self.entity_type);
        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some(step_prefix),
            grammar: WordsPrimitive::default()
                .min_count(1)
                .max_count(5)
                .concatenator(", ")
                .grammar(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("\n".to_owned());
        Ok(self.flow.primitive_result())
    }

    async fn name(&mut self) -> Result<Option<String>> {
        let step_prefix = format!("{} name:", self.entity_type);
        let stop_word_done = format!("{} description:", self.entity_type);
        let config = StepConfig {
            cache_prompt: true,
            stop_word_done,
            step_prefix: Some(step_prefix),
            grammar: TextPrimitive::default().text_token_length(75).grammar(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("\n".to_owned());
        Ok(self.flow.primitive_result())
    }

    async fn blog_url(&mut self, potential_name: &str) -> Result<Option<String>> {
        let grammar = Grammar::faux_url()
            .min_count(3)
            .max_count(7)
            .base_url("https://blog.com/");

        let potential = extract_urls(potential_name);
        if let Some(result) = potential.last() {
            return Ok(Some(grammar.grammar_parse(result.as_str())?.join(" ")));
        }

        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some("blog url:".to_owned()),

            grammar: grammar.clone().wrap(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("\n".to_owned());
        if let Some(result) = self.flow.primitive_result() {
            Ok(Some(grammar.grammar_parse(&result)?.join(" ")))
        } else {
            Ok(None)
        }
    }

    async fn check_for_remaining(&mut self) -> Result<bool> {
        self.flow.new_round(format!(
            "True or false: are there any unique {}s in the text that have not yet been extracted?",
            self.entity_type
        ));
        self.flow.last_round()?.open_round(&mut self.base_req);

        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some("The text".to_owned()),
            grammar: TextPrimitive::default().text_token_length(25).grammar(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("...".to_owned());

        let config = StepConfig {
            cache_prompt: true,
            step_prefix: Some("So the answer is:".to_owned()),
            grammar: BooleanPrimitive::default().grammar(),
            ..StepConfig::default()
        };

        self.flow.last_round()?.add_inference_step(&config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        let res: bool = self.flow.primitive_result().unwrap().parse().unwrap();
        self.flow.last_round()?.close_round(&mut self.base_req)?;
        Ok(res)
    }

    async fn run_cascade(&mut self) -> Result<()> {
        self.flow
            .new_round(format!(
                "Of the {} {}s in the text, return the most prevalent {}.",
                self.entity_count.unwrap(),
                self.entity_type,
                self.entity_type
            ))
            .step_separator = None;
        self.flow.last_round()?.open_round(&mut self.base_req);

        for i in 1..=self.entity_count.unwrap() {
            if i > 1 {
                // if !self.check_for_remaining().await? {
                //     break;
                // }
                self.flow
                    .new_round(format!(
                        "Of the {} {}s in the text, return the next most prevalent {}.",
                        self.entity_count.unwrap(),
                        self.entity_type,
                        self.entity_type
                    ))
                    .step_separator = None;
                self.flow.last_round()?.open_round(&mut self.base_req);
            }
            let description = self.description().await?.unwrap();
            let keywords = self.keywords().await?.unwrap();
            let potential_name = self.name().await?.unwrap();
            let name = self.blog_url(&potential_name).await?.unwrap();
            self.results.push(ExtractEntity {
                description,
                keywords,
                name,
            });
            self.flow.last_round()?.close_round(&mut self.base_req)?;
        }

        self.flow.close_cascade(&mut self.base_req)?;
        Ok(())
    }
}

impl BaseRequestConfigTrait for ExtractEntities {
    fn base_config(&mut self) -> &mut BaseRequestConfig {
        &mut self.base_req.config
    }

    fn clear_request(&mut self) {
        self.instruct_prompt.reset_instruct_prompt();
        self.base_req.reset_request();
        self.results.clear();
    }
}

impl InstructPromptTrait for ExtractEntities {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.instruct_prompt
    }
}

#[derive(Clone)]
pub struct ExtractEntity {
    pub description: String,
    pub keywords: String,
    pub name: String,
}

impl std::fmt::Display for ExtractEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\x1b[1m\x1b[38;5;174mName\x1b[0m: '{}'", self.name)?;
        writeln!(
            f,
            "\x1b[38;5;175mDescription\x1b[0m: '{}'",
            self.description
        )?;
        writeln!(f, "\x1b[38;5;176mKeywords\x1b[0m: '{}'", self.keywords)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct ExtractEntitiesResult {
    entity_type: String,
    pub results: Option<Vec<ExtractEntity>>,
    pub duration: std::time::Duration,
    pub workflow: CascadeFlow,
}

impl ExtractEntitiesResult {
    fn new(flow: CascadeFlow, results: Option<Vec<ExtractEntity>>, entity_type: &str) -> Self {
        ExtractEntitiesResult {
            entity_type: entity_type.to_string(),
            results,
            duration: flow.duration,
            workflow: flow,
        }
    }
}

impl std::fmt::Display for ExtractEntitiesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(
            f,
            "\x1b[38;5;172m\x1b[1m{}\x1b[0m - {}",
            self.workflow.cascade_name, self.entity_type
        )?;
        writeln!(f)?;
        for result in self.results.iter().flatten() {
            writeln!(f, "{}", result)?;
        }
        writeln!(f)?;
        writeln!(f, "\x1b[38;5;43mduration\x1b[0m: {:?}", self.duration)?;
        Ok(())
    }
}
