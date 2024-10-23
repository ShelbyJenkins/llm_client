use crate::components::cascade::CascadeFlow;
use crate::{components::cascade::step::StepConfig, primitives::*};

use llm_interface::requests::completion::CompletionRequest;

#[derive(Clone)]
pub struct ClassifySubjectOfText {
    pub base_req: CompletionRequest,
    pub content: String,
    pub flow: CascadeFlow,
    pub content_strings: Vec<String>,
    pub subject: Option<String>,
    subject_strings: Vec<String>,
    default_grammar: SentencesPrimitive,
    default_step_config: StepConfig,
}

impl ClassifySubjectOfText {
    pub fn new(base_req: CompletionRequest, content: &str) -> Self {
        let mut grammar: SentencesPrimitive = SentencesPrimitive::default();
        grammar
            .min_count(1)
            .max_count(3)
            .disallowed_char('\'')
            .disallowed_char('(')
            .disallowed_char(')')
            .capitalize_first(false);
        let mut step_config: StepConfig = StepConfig::default();
        step_config.stop_word_done("\n").grammar(grammar.grammar());

        Self {
            base_req,
            content: content.to_owned(),
            flow: CascadeFlow::new("ClassifySubjectOfText"),
            content_strings: Vec::new(),
            subject: None,
            subject_strings: Vec::new(),
            default_grammar: grammar,
            default_step_config: step_config,
        }
    }

    pub async fn run(mut self) -> crate::Result<Self> {
        let mut count = 1;
        while count <= self.base_req.config.retry_after_fail_n_times {
            match self.run_cascade().await {
                Ok(_) => break,
                Err(e) => {
                    crate::error!("Failed to classify entity: {}", e);
                    count += 1;
                    if count == self.base_req.config.retry_after_fail_n_times {
                        crate::bail!("Failed to classify entity after {} attempts: {}", count, e);
                    }
                    self.base_req.reset_completion_request();
                    self.flow = CascadeFlow::new("ClassifySubjectOfText");
                }
            }
        }
        Ok(self)
    }

    async fn run_cascade(&mut self) -> crate::Result<()> {
        self.flow.open_cascade();
        let task = indoc::formatdoc! {"
        Explain like I'm five; what is the subject of the text:
        '{}'",
        self.content
        };
        self.flow.new_round(task).step_separator('\n');
        self.flow.last_round()?.open_round(&mut self.base_req)?;

        self.default_step_config
            .step_prefix("In the text, the main thing a five-year-old would see is: \"")
            .grammar(self.default_grammar.max_count(2).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        self.default_step_config
            .step_prefix(
                "An english teacher would clarify that the person or thing that is being discussed, described, or dealt with, is: \"",
            )
            .grammar(self.default_grammar.max_count(2).capitalize_first(false).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        self.ensure_options().await?;

        self.default_step_config
            .step_prefix(format!(
                "So, the primary subject of the text '{}' is: \"",
                self.content
            ))
            .grammar(self.default_grammar.max_count(1).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        let possible_subjects = self.extract_quoted_text();
        if possible_subjects.len() == 1 {
            self.subject = Some(possible_subjects[0].clone());
        } else {
            self.default_step_config
                .step_prefix(
                    "To restate so a five-year-old could understand, the primary subject is: ",
                )
                .grammar(
                    ExactStringPrimitive::default()
                        .add_strings_to_allowed(&possible_subjects)
                        .grammar(),
                );

            self.run_it().await?;
            self.subject = self.flow.last_round()?.last_step()?.primitive_result();
        }

        self.flow.last_round()?.close_round(&mut self.base_req)?;
        self.flow.close_cascade()?;

        Ok(())
    }

    async fn run_it(&mut self) -> crate::Result<String> {
        self.flow
            .last_round()?
            .add_inference_step(&self.default_step_config);
        self.flow
            .last_round()?
            .run_next_step(&mut self.base_req)
            .await?;
        let result = self
            .flow
            .last_round()?
            .last_step()?
            .display_step_outcome()?;
        Ok(result)
    }

    async fn ensure_options(&mut self) -> crate::Result<()> {
        let mut possible_subjects = self.extract_quoted_text();

        if !possible_subjects.is_empty() {
            return Ok(());
        };
        self.default_step_config
            .step_prefix("The nouns in the text are: \"")
            .grammar(self.default_grammar.max_count(1).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        self.default_step_config
            .step_prefix("The proper nouns in the text are: \"")
            .grammar(self.default_grammar.max_count(1).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        self.default_step_config
            .step_prefix("The common nouns in the text are: \"")
            .grammar(self.default_grammar.max_count(1).grammar());
        let result = self.run_it().await?;
        self.subject_strings.push(result);

        possible_subjects = self.extract_quoted_text();
        if possible_subjects.is_empty() {
            crate::bail!("Failed to classify subject: no qouted subject returned");
        }
        Ok(())
    }

    fn extract_quoted_text(&self) -> Vec<String> {
        let mut result = Vec::new();

        for input in &self.subject_strings {
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
                .to_lowercase()
                .split_whitespace()
                .filter(|word| word.len() > 1 && *word != "the")
                .collect::<Vec<_>>()
                .join(" ")
                .to_owned();
            cleaned.push(new_res);
        }
        cleaned.sort();
        cleaned.dedup();
        let lower_content = self.content.to_lowercase();
        cleaned.retain(|x| lower_content.contains(x));
        // println!("{:?}", self.subject_strings);
        // println!("{:?}", cleaned);

        cleaned
    }
}

impl std::fmt::Display for ClassifySubjectOfText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "ClassifySubjectOfText:")?;
        crate::i_nln(f, format_args!("content: \"{}\"", self.content))?;
        crate::i_nln(f, format_args!("subject: {:?}", self.subject))?;
        crate::i_nln(f, format_args!("duration: {:?}", self.flow.duration))?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
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
    use crate::{prelude::*, workflows::classify::subject_of_text::ClassifySubjectOfText};

    #[tokio::test]
    #[ignore]
    pub async fn test_cases() -> crate::Result<()> {
        let llm_client = LlmClient::llama_cpp().llama3_1_8b_instruct().init().await?;

        for (case, answer) in CASES {
            let entity = ClassifySubjectOfText::new(
                CompletionRequest::new(llm_client.backend.clone()),
                case,
            );
            let entity = entity.run().await?;
            println!("{}", entity.flow);
            println!("{}", entity);
            assert!(entity.subject.unwrap().contains(answer));
        }

        Ok(())
    }
}
