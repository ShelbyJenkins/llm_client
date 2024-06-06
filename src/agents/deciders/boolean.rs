use self::decider_backend::*;
use super::*;

impl<'a> RequestConfigTrait for BooleanDecider<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

const BOOLEAN_JUSTIFICATION_PROMPT: &str = r#"The response should contain your justification and end with "The answer is:". You are answering a boolean question. If the answer is true/yes/affirmative, return 'true'. If the answer is false/no/negative, return 'false'. Do not restate the question. Do not annotate. Justify your response succinctly."#;

const BOOLEAN_BASIC_PARSER_PROMPT: &str = r#"The user answered a boolean question in plain spoken english. Do not explain the user's answer. Do not correct the user's answer. The user's answer will be either true or false. Respond only with the the answer."#;

const BOOLEAN_GRAMMER_PARSER_PROMPT: &str = r#"The user answered a multiple choice question. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the multiple choice answer matching the users answer."#;

const BOOLEAN_LOGIT_BIAS_PARSER_PROMPT: &str = r#"The user answered a multiple choice question. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the multiple choice answer matching the users answer."#;

#[derive(Clone)]
pub struct BooleanDecider<'a> {
    pub decider_choices: Vec<DeciderChoice>,
    pub decider_config: DeciderConfig,
    pub decider_result: Option<DeciderResult>,
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
}

impl<'a> BooleanDecider<'a> {
    pub fn new(
        decider_config: DeciderConfig,
        default_request_config: RequestConfig,
        llm_client: &'a LlmClient,
    ) -> Self {
        let true_choice = DeciderChoice {
            choice_value: "true".to_string(),
            parser_key: Self::create_parser_key(&decider_config.decision_parser_type, true),
            choice_index: 0,
        };

        let false_choice = DeciderChoice {
            choice_value: "false".to_string(),
            parser_key: Self::create_parser_key(&decider_config.decision_parser_type, false),
            choice_index: 1,
        };
        Self {
            decider_choices: vec![true_choice, false_choice],
            decider_config,
            decider_result: None,
            req_config: default_request_config,
            llm_client,
        }
    }

    /// Runs the BooleanDecider and returns the DeciderResult object.
    ///
    /// # Returns
    ///
    /// The result of the BooleanDecider.
    pub async fn run_with_result(&mut self) -> Result<DeciderResult> {
        self.decider_result = Some(DeciderResult::default());

        self.create_justification_prompt();

        self.create_parser_prompt();

        match self.decider_config.decision_parser_type {
            DecisionParserType::Basic => {}
            DecisionParserType::Grammar => {
                self.decider_config.grammar = Some(create_parser_grammar(self.decider_choices()));
            }
            DecisionParserType::LogitBias => {
                if self.req_config.logit_bias.is_none() {
                    self.req_config.logit_bias = Some(
                        generate_parser_logit_bias(
                            &self.llm_client.backend,
                            self.decider_choices(),
                        )
                        .await?,
                    );
                    generate_parser_logit_bias_for_backend(
                        &self.llm_client.backend,
                        &mut self.req_config,
                    )
                    .await?;
                }
            }
        }

        self.run_decision().await?;
        Ok(self.get_decider_result().clone())
    }

    /// Runs the BooleanDecider and returns the boolean result.
    ///
    /// # Returns
    ///
    /// The boolean result of the BooleanDecider.
    pub async fn run(&mut self) -> Result<bool> {
        let res = self.run_with_result().await?;
        let bool: bool = res.choice_value.parse()?;
        Ok(bool)
    }

    fn create_parser_key(decision_parser_type: &DecisionParserType, choice_value: bool) -> String {
        match decision_parser_type {
            DecisionParserType::Basic => choice_value.to_string(),
            DecisionParserType::Grammar => {
                format!("The answer is {choice_value}).")
            }
            DecisionParserType::LogitBias => choice_value.to_string(),
        }
    }

    fn create_justification_prompt(&mut self) {
        let mut justification_prompt = BOOLEAN_JUSTIFICATION_PROMPT.to_string();
        if let Some(existing_system_content) = &self.req_config().system_content {
            justification_prompt.push_str(&format!(
                "additional instructions: {}",
                existing_system_content
            ));
        }
        justification_prompt.push_str("\nchoices:");
        for choice in self.decider_choices() {
            justification_prompt.push_str(&format!(" {},", choice.choice_value));
        }
        self.decider_config.justification_prompt = justification_prompt;
    }

    fn create_parser_prompt(&mut self) {
        let mut decision_parser_prompt = match self.decider_config.decision_parser_type {
            DecisionParserType::Grammar => BOOLEAN_GRAMMER_PARSER_PROMPT.to_string(),
            DecisionParserType::Basic => BOOLEAN_BASIC_PARSER_PROMPT.to_string(),
            DecisionParserType::LogitBias => BOOLEAN_LOGIT_BIAS_PARSER_PROMPT.to_string(),
        };
        decision_parser_prompt.push_str("\nchoices:");
        for choice in self.decider_choices() {
            decision_parser_prompt.push_str(&format!(" {},", choice.choice_value));
        }

        self.decider_config.decision_parser_prompt = decision_parser_prompt;
    }
}

impl DecisionThing for BooleanDecider<'_> {
    fn set_decider_result(&mut self) -> &mut DeciderResult {
        self.decider_result.as_mut().unwrap()
    }
    fn get_decider_result(&self) -> &DeciderResult {
        self.decider_result.as_ref().unwrap()
    }
    fn decider_config(&self) -> &DeciderConfig {
        &self.decider_config
    }
    fn decider_choices(&self) -> &Vec<DeciderChoice> {
        &self.decider_choices
    }
    fn req_config(&self) -> &RequestConfig {
        &self.req_config
    }
    fn llm_client(&self) -> &LlmClient {
        self.llm_client
    }
    fn set_temperature(&mut self, value: f32) {
        self.req_config.temperature = value;
    }
    fn validate_basic_parser_response(&self, response: Result<String>) -> Result<u16> {
        match response {
            Ok(response) => {
                let mut matches: HashMap<u16, u8> = HashMap::new();
                let response = response.trim().to_lowercase();
                let mut result = "BooleanDecider validate_basic_parser_response:".to_string();
                for choice in &self.decider_choices {
                    let count = response
                        .matches(&choice.choice_value.to_lowercase())
                        .count() as u8;
                    if count > 0 {
                        result.push_str(&format!(
                            " matches found for choice_value: {} count: {}",
                            choice.choice_value, count
                        ));
                        matches.insert(choice.choice_index, count);
                    }
                }
                if matches.len() > 2 {
                    result.push_str(&format!(
                        " error: more than two matching choice_value in response. Matches should be either 'true' or 'false' matches.len() == {}",
                        matches.len()
                    ));
                    if self.llm_client.backend.logging_enabled() {
                        tracing::info!(?result);
                    }
                    Err(anyhow!("{}", result))
                } else if matches.is_empty() {
                    result.push_str(" error: no matching choice_value in response.");
                    if self.llm_client.backend.logging_enabled() {
                        tracing::info!(?result);
                    }
                    Err(anyhow!("{}", result))
                } else {
                    if self.llm_client.backend.logging_enabled() {
                        tracing::info!(?result);
                    } else {
                        println!("{}", result);
                    }
                    let key_with_highest_count = matches
                        .iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(key, _)| key)
                        .unwrap();
                    Ok(key_with_highest_count.to_owned())
                }
            }
            Err(error) => Err(anyhow!("{}", error)),
        }
    }
}

pub async fn apply_test_questions(mut decider: BooleanDecider<'_>) -> Result<()> {
    let question = "Do scientist believe the moon is made of cheese?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    let res = res.choice_value.parse::<bool>()?;
    if !res {
        println!("🟢 correct boolean response '{res}' to test question: {question}");
    } else {
        println!("🔴 incorrect boolean response '{res}' to test question: {question}");
    }

    let question = "Would most people say the sky is blue?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    res.choice_value.parse::<bool>()?;
    let res = res.choice_value.parse::<bool>()?;
    if res {
        println!("🟢 correct boolean response '{res}' to test question: {question}");
    } else {
        println!("🔴 incorrect boolean response '{res}' to test question: {question}");
    }

    let question = "Would most people say the italy is in asia?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    let res = res.choice_value.parse::<bool>()?;
    if !res {
        println!("🟢 correct boolean response '{res}' to test question: {question}");
    } else {
        println!("🔴 incorrect boolean response '{res}' to test question: {question}");
    }
    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::deciders::boolean::apply_test_questions;
    use crate::*;
    use anyhow::Result;

    #[tokio::test]
    #[serial]
    pub async fn test_basic() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_basic_backend().boolean();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_grammar() -> Result<()> {
        let llm = LlmClient::llama_backend().init().await?;

        let decider = llm.decider().use_grammar_backend().boolean();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_logit_bias() -> Result<()> {
        let llm = LlmClient::llama_backend().init().await?;

        let decider = llm.decider().use_logit_bias_backend().boolean();
        apply_test_questions(decider).await?;

        Ok(())
    }
}
