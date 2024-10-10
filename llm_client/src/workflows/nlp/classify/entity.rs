use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;
use std::collections::HashMap;

#[derive(Clone)]
pub struct ClassifyEntity {
    pub base_req: CompletionRequest,
    pub content: String,
    pub flow: CascadeFlow,
    pub content_strings: Vec<String>,
    pub subject: Option<String>,
    pub common_noun: Option<String>,
    pub specific_identifer: Option<String>,
}

impl ClassifyEntity {
    pub fn new(base_req: CompletionRequest, content: &str) -> Self {
        Self {
            base_req,
            content: content.to_owned(),
            flow: CascadeFlow::new("ClassifyEntity"),
            content_strings: Vec::new(),
            subject: None,
            common_noun: None,
            specific_identifer: None,
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        self.run_cascade().await?;
        // let mut count = 1;
        // while count <= self.base_req.config.retry_after_fail_n_times {
        //     match self.run_cascade().await {
        //         Ok(_) => break,
        //         Err(e) => {
        //             count += 1;
        //             if count == self.base_req.config.retry_after_fail_n_times {
        //                 crate::bail!("Failed to classify entity after {} attempts: {}", count, e);
        //             }
        //             self.base_req.reset_completion_request();
        //             self.flow = CascadeFlow::new("ClassifyEntity");
        //         }
        //     }
        // }
        // println!("{}", self.flow);
        Ok(self)
    }

    async fn run_cascade(&mut self) -> crate::Result<()> {
        self.flow.open_cascade();

        let task = indoc::formatdoc! {"
        Explain like I'm five; what is the subject of the text:
        '{}'",
        self.content
        };
        self.flow.new_round(task).step_separator(' ');
        self.flow.last_round()?.open_round(&mut self.base_req)?;

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "In the text, a five-year-old would find the *interesting* things to be: "
            )),
            stop_word_done: format!("\n"),
            grammar: TextPrimitive::default()
                // .max_count(2)
                // .capitalize_first(false)
                .grammar(), // May need to increase this if input has punctuation
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.run_it().await?;

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "However, the *main* thing in the text, the subject, is "
            )),
            stop_word_done: format!("\n"),
            grammar: TextPrimitive::default()
                // .max_count(2)
                // .capitalize_first(false)
                .grammar(), // May need to increase this if input has punctuation
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.run_it().await?;
        let result = self
            .flow
            .last_round()?
            .last_step()?
            .display_step_outcome()?;
        self.content_strings.push(result);

        // let step_config = StepConfig {
        //     step_prefix: Some(format!(
        //         "The word that represents this in the text, the noun, is: \""
        //     )),
        //     stop_word_done: format!("\n"),
        //     grammar: SentencesPrimitive::default().max_count(1).grammar(), // May need to increase this if input has punctuation
        //     ..StepConfig::default()
        // };
        // self.flow.last_round()?.add_inference_step(&step_config);
        // self.run_it().await?;
        // let result = self
        //     .flow
        //     .last_round()?
        //     .last_step()?
        //     .display_step_outcome()?;
        // self.content_strings.push(result);

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "So, an english teacher would say the subject is: \""
            )),
            stop_word_done: format!("\n"),
            grammar: SentencesPrimitive::default().max_count(1).grammar(), // May need to increase this if input has punctuation
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.run_it().await?;
        let result = self
            .flow
            .last_round()?
            .last_step()?
            .display_step_outcome()?;
        self.content_strings.push(result);

        // println!("{:?}", self.content_strings);
        let possible_subjects = extract_quoted_text(self.content_strings.clone());
        let step_config = StepConfig {
            step_prefix: Some(format!(
                "But, you could simplify the subject with the word: \""
            )),
            grammar: ExactStringPrimitive::default()
                .add_strings_to_allowed(&possible_subjects)
                .grammar(), // May need to increase this if input has punctuation
            ..StepConfig::default()
        };
        self.flow.last_round()?.add_inference_step(&step_config);
        self.run_it().await?;
        self.flow
            .last_round()?
            .last_step()?
            .set_dynamic_suffix("\"");
        let result = self
            .flow
            .last_round()?
            .last_step()?
            .primitive_result()
            .unwrap();
        self.common_noun = Some(result);
        self.subject = most_common_words(&possible_subjects).into();

        self.flow.last_round()?.close_round(&mut self.base_req)?;

        self.flow.close_cascade()?;
        Ok(())
    }

    async fn run_it(&mut self) -> crate::Result<()> {
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;

        Ok(())
    }
}

fn extract_quoted_text(inputs: Vec<String>) -> Vec<String> {
    fn is_likely_contraction(chars: &[char], index: usize) -> bool {
        if index == 0 || index == chars.len() - 1 {
            return false;
        }

        let prev = chars[index - 1];
        let next = chars[index + 1];

        prev.is_alphabetic() && (next.is_alphabetic() || next == 's')
    }

    let mut result = Vec::new();

    for input in inputs {
        let chars: Vec<char> = input.chars().collect();
        let mut start_index = 0;
        let mut current_quote = None;
        for (i, &c) in chars.iter().enumerate() {
            match (current_quote, c) {
                (None, '\'') => {
                    if is_likely_contraction(&chars, i) {
                        continue;
                    }
                    current_quote = Some(c);
                    start_index = i + 1;
                }
                (None, '"') => {
                    current_quote = Some(c);
                    start_index = i + 1;
                }
                (Some(quote), c) if c == quote => {
                    if quote == '\'' && is_likely_contraction(&chars, i) {
                        continue;
                    }
                    result.push(input[start_index..i].to_string());
                    current_quote = None;
                }
                _ => {}
            }
        }
        let mut current_quote = None;
        let mut start_index = 0;
        for (i, c) in input.char_indices() {
            match (current_quote, c) {
                (None, '"') => {
                    current_quote = Some(c);
                    start_index = i + 1;
                }
                (Some(quote), c) if c == quote => {
                    result.push(input[start_index..i].to_string());
                    current_quote = None;
                }
                _ => {}
            }
        }
    }
    let mut cleaned = vec![];
    for res in result.iter_mut() {
        let new_res = res
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .trim_end_matches(|c: char| !c.is_alphanumeric())
            .to_owned();
        cleaned.push(new_res);
    }
    println!("{:?}", cleaned);
    cleaned
}

fn most_common_words(strings: &[String]) -> String {
    let mut word_counts: HashMap<String, usize> = HashMap::new();

    for string in strings {
        // Convert to lowercase and split into words
        let words: Vec<String> = string
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        // Count individual words
        for word in &words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }

        // Count word pairs
        if words.len() > 1 {
            let pair = words.join(" ");
            *word_counts.entry(pair).or_insert(0) += 1;
        }
    }

    // Find the entry with the highest count
    word_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(words, _)| words)
        .unwrap_or_else(|| String::from(""))
}

impl std::fmt::Display for ClassifyEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ClassifyEntity:")?;
        crate::i_nln(f, format_args!("content: \"{}\"", self.content))?;
        crate::i_nln(f, format_args!("subject: {:?}", self.subject))?;
        crate::i_nln(
            f,
            format_args!("specific_identifer: {:?}", self.specific_identifer),
        )?;
        crate::i_nln(f, format_args!("common_noun: {:?}", self.common_noun))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    const CASES: &[&str] = &[
        // "Ciliate: Metopus sp. strain SALT15A",
        // "Coastal soil sample",
        // "Edible insect Gryllus bimaculatus (Pet Feed Store)",
        // "Public spring water",
        "River snow from South Saskatchewan River",
        "Tara packed so many boxes that she ran out of tape, and had to go to the store to buy more. Then she made grilled cheese sandwiches for lunch.",
    ];

    use crate::prelude::*;

    #[tokio::test]
    #[ignore]
    pub async fn test() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().init().await?;

        let req = llm_client
            .nlp()
            .classify()
            .entity("A green turtle on a log in a mountain lake.");
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);
        // assert!(entity.entity_type.unwrap().contains("turtle"));

        let req = llm_client.nlp().classify().entity(
            "Mountain lake mirror\nGreen shell gleams on weathered log\nTurtle's calm retreat",
        );
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);
        // assert!(entity.entity_type.unwrap().contains("turtle"));

        let req = llm_client
            .nlp()
            .classify()
            .entity("Green turtle on log\nSunlight warms her emerald shell\nStillness all around");
        let entity = req.run().await?;
        println!("{}", entity.flow);
        println!("{}", entity);
        // assert!(entity.entity_type.unwrap().contains("turtle"));

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

            println!("{}", entity);
        }

        Ok(())
    }
}
