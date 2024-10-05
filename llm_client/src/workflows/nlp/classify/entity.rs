use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;
use std::collections::HashSet;

#[derive(Clone)]
pub struct ClassifyEntity {
    pub base_req: CompletionRequest,
    pub content: String,
    pub entity_type: Option<String>,
    pub flow: CascadeFlow,
}

impl ClassifyEntity {
    pub fn new(base_req: CompletionRequest, content: &str) -> Self {
        Self {
            base_req,
            content: content.to_owned(),
            entity_type: None,
            flow: CascadeFlow::new("ClassifyEntity"),
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        let mut count = 1;
        while count <= self.base_req.config.retry_after_fail_n_times {
            match self.run_cascade().await {
                Ok(_) => break,
                Err(e) => {
                    self.base_req.reset_completion_request();
                    self.flow = CascadeFlow::new("ClassifyEntity");
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

    async fn run_cascade(&mut self) -> crate::Result<()> {
        let task = format!("Identify the primary entity that serves as the main subject of the text.\nGuidelines:\nIdentify the Primary Entity: Determine the most significant entity that acts as the subject of the text. The entity should be the main focus of the discussion.\nFocus on Concrete Entities Only: The primary entity should be tangible such as a physical object, being, or any specific noun mentioned.\nProcess:\nStart by listing the entities mentioned in the text, \"The entities in the text are: ...\"\nAnalyze each potential entity in the text, \"Thinking out loud about which is the primary entity: ...\"\nAfter discussing all options propose the best candidate, \"Therefore, the primary entity is: ...\"\nFinally, state the primary entity as a single word, \"Primary entity: ...\"\nThe text is: '{}'", self.content);
        let round = self.flow.new_round(task);
        round.open_round(&mut self.base_req)?;

        // List entities
        // Need to make a list grammar type
        let step_config = StepConfig {
            step_prefix: Some("The entities in the text are:".to_owned()),
            stop_word_done: "Thinking out loud about which is the primary entity".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;
        let list_content = match round.primitive_result() {
            Some(cot_content) => cot_content,
            None => {
                crate::bail!("No entity identified.");
            }
        };

        // CoT
        let step_config = StepConfig {
            step_prefix: Some("Thinking out loud about which is the primary entity:".to_owned()),
            stop_word_done: "Therefore, the primary entity is:".to_owned(),
            grammar: TextPrimitive::default().text_token_length(200).grammar(),
            ..StepConfig::default()
        };
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;
        let cot_content = match round.primitive_result() {
            Some(cot_content) => cot_content,
            None => {
                crate::bail!("No entity identified.");
            }
        };

        // Extract entity label
        let step_config = StepConfig {
            step_prefix: Some("Therefore, the primary entity is:".to_owned()),
            stop_word_done: "Primary entity:".to_owned(),
            grammar: SentencesPrimitive::default().max_count(2).grammar(),
            ..StepConfig::default()
        };
        round.add_inference_step(&step_config);
        round.run_next_step(&mut self.base_req).await?;
        let entity_type = match round.primitive_result() {
            Some(entity_type) => {
                self.entity_type = Some(entity_type.clone());
                entity_type
            }
            None => {
                crate::bail!("No entity identified.");
            }
        };

        // Extract entity label
        let common_words =
            words_in_all_strings(&[&list_content, &cot_content, &entity_type, &self.content]);
        if !common_words.is_empty() {
            let step_config = StepConfig {
                step_prefix: Some("Primary entity:".to_owned()),
                grammar: ExactStringPrimitive::default()
                    .add_strings_to_allowed(&common_words)
                    .grammar(),
                cache_prompt: false,
                ..StepConfig::default()
            };
            round.add_inference_step(&step_config);
            round.run_next_step(&mut self.base_req).await?;
            match round.primitive_result() {
                Some(entity_type) => {
                    self.entity_type = Some(entity_type);
                }
                None => (),
            };
        }
        round.close_round(&mut self.base_req)?;
        self.flow.close_cascade()?;
        Ok(())
    }
}

fn words_in_all_strings(strings: &[&str]) -> Vec<String> {
    let words: Vec<Vec<String>> = strings
        .iter()
        .map(|&s| {
            s.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .collect();

    if words.is_empty() {
        return Vec::new();
    }

    let mut common_words: HashSet<_> = words[0].iter().cloned().collect();
    for words in &words[1..] {
        let word_set: HashSet<_> = words.iter().collect();
        common_words.retain(|word| word_set.contains(word));
    }

    common_words.into_iter().collect()
}

impl std::fmt::Display for ClassifyEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ClassifyEntity:")?;
        crate::i_nln(f, format_args!("content: \"{}\"", self.content))?;
        crate::i_nln(f, format_args!("entity_type: {:?}", self.entity_type))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::LlmClient;

    #[tokio::test]
    #[ignore]
    pub async fn test() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;

        let req = llm_client
            .nlp()
            .classify()
            .entity("A green turtle on a log in a mountain lake.");
        let entity = req.run().await?;
        println!("{}", entity);
        assert!(entity.entity_type.unwrap().contains("turtle"));

        let req = llm_client.nlp().classify().entity(
            "Mountain lake mirror\nGreen shell gleams on weathered log\nTurtle's calm retreat",
        );
        let entity = req.run().await?;
        println!("{}", entity);
        assert!(entity.entity_type.unwrap().contains("turtle"));

        let req = llm_client
            .nlp()
            .classify()
            .entity("Green turtle on log\nSunlight warms her emerald shell\nStillness all around");
        let entity = req.run().await?;
        println!("{}", entity);
        assert!(entity.entity_type.unwrap().contains("turtle"));

        Ok(())
    }
}
