use self::decider_backend::*;
use super::*;

impl<'a> RequestConfigTrait for CustomDecider<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}
const CUSTOM_JUSTIFICATION_PROMPT: &str = r#"The response should contain your justification and end with "The answer is:". You are answering a multiple choice question. Do not list the choices. Do not restate the question. Do not annotate. Justify your response succinctly."#;

const CUSTOM_BASIC_PARSER_PROMPT: &str = r#"The user answered a multiple choice question in plain spoken english. Do not explain the user's answer. Do not correct the user's answer. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the answer."#;

const CUSTOM_GRAMMER_PARSER_PROMPT: &str = r#"The user answered a multiple choice question in plain spoken english. Do not explain the user's answer. Do not correct the user's answer. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the answer."#;

const CUSTOM_LOGIT_BIAS_PARSER_PROMPT: &str = r#"The user answered a multiple choice question in plain spoken english. Do not explain the user's answer. Do not correct the user's answer. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the answer."#;

#[derive(Clone)]
pub struct CustomDecider<'a> {
    pub decider_choices: Vec<DeciderChoice>,
    pub decider_config: DeciderConfig,
    pub decider_result: Option<DeciderResult>,
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
}

impl<'a> CustomDecider<'a> {
    pub fn new(
        decider_config: DeciderConfig,
        default_request_config: RequestConfig,
        llm_client: &'a LlmClient,
    ) -> Self {
        Self {
            decider_choices: Vec::new(),
            decider_config,
            decider_result: None,
            req_config: default_request_config,
            llm_client,
        }
    }

    /// Add a choice to the list of choices for the decider.
    pub fn add_choice(&mut self, choice: &str) -> &mut Self {
        let choice = DeciderChoice {
            choice_value: choice.to_string(),
            choice_index: self.decider_choices.len() as u16,
            parser_key: self.create_parser_key(choice),
        };
        self.decider_choices.push(choice);
        self
    }

    /// Add multiple choices to the list of choices for the decider.
    pub fn add_choices(&mut self, choices: &Vec<String>) -> &mut Self {
        for choice in choices {
            let choice = DeciderChoice {
                choice_value: choice.to_string(),
                choice_index: self.decider_choices.len() as u16,
                parser_key: self.create_parser_key(choice),
            };
            self.decider_choices.push(choice);
        }
        self
    }

    /// Clear all existing choices. Useful for testing and reusing the same decider for multiple decisions.
    pub fn clear_choices(&mut self) -> &mut Self {
        self.decider_choices.clear();
        self.req_config.logit_bias = None;
        self.decider_config.grammar = None;
        self
    }

    /// Run the decider and return the DeciderResult object. The choice value can be found in the choice_value field.
    pub async fn run_with_result(&mut self) -> Result<DeciderResult> {
        if self.decider_choices.is_empty() {
            panic!("Choices should not be empty")
        }

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

    /// Run the decider and return the choice string value.
    pub async fn run(&mut self) -> Result<String> {
        let res = self.run_with_result().await?;
        Ok(res.choice_value)
    }

    fn create_parser_key(&self, choice_value: &str) -> String {
        match self.decider_config.decision_parser_type {
            DecisionParserType::Basic => choice_value.to_string(),
            DecisionParserType::Grammar => {
                format!("The answer is {choice_value}).")
            }
            DecisionParserType::LogitBias => choice_value.to_string(),
        }
    }

    fn create_justification_prompt(&mut self) {
        let mut justification_prompt = CUSTOM_JUSTIFICATION_PROMPT.to_string();
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
            DecisionParserType::Grammar => CUSTOM_GRAMMER_PARSER_PROMPT.to_string(),
            DecisionParserType::Basic => CUSTOM_BASIC_PARSER_PROMPT.to_string(),
            DecisionParserType::LogitBias => CUSTOM_LOGIT_BIAS_PARSER_PROMPT.to_string(),
        };
        decision_parser_prompt.push_str("\nchoices:");
        for choice in self.decider_choices() {
            decision_parser_prompt.push_str(&format!(" {},", choice.choice_value));
        }

        self.decider_config.decision_parser_prompt = decision_parser_prompt;
    }
}

impl DecisionThing for CustomDecider<'_> {
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
                let mut result = "CustomDecider validate_basic_parser_response:".to_string();
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
                if matches.len() > self.decider_choices.len() {
                    result.push_str(&format!(
                        " error: more matching values in response than self.decider_choices.len() {}",
                        self.decider_choices.len()
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

pub async fn apply_test_questions(mut decider: CustomDecider<'_>) -> Result<()> {
    let question = "Most people would say the color of the sky is what color?";
    decider
        .clear_choices()
        .user_content(question)
        .add_choice("red")
        .add_choice("purple")
        .add_choice("blue")
        .add_choice("green");
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_value == "blue" {
        println!(
            "🟢 correct custom response '{}' to test question: {question}",
            res.choice_value
        );
    } else {
        println!(
            "🔴 incorrect custom response '{}' to test question: {question}",
            res.choice_value
        );
    }

    let question = "Which of these is a food?";
    decider
        .clear_choices()
        .user_content(question)
        .add_choice("pizza")
        .add_choice("cats")
        .add_choice("sky");
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_value == "pizza" {
        println!(
            "🟢 correct custom response '{}' to test question: {question}",
            res.choice_value
        );
    } else {
        println!(
            "🔴 incorrect custom response '{}' to test question: {question}",
            res.choice_value
        );
    }

    let question = "Which of these is not a food?";
    decider
        .clear_choices()
        .user_content(question)
        .add_choice("pizza")
        .add_choice("cats")
        .add_choice("bread")
        .add_choice("apple");
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_value == "cats" {
        println!(
            "🟢 correct custom response '{}' to test question: {question}",
            res.choice_value
        );
    } else {
        println!(
            "🔴 incorrect custom response '{}' to test question: {question}",
            res.choice_value
        );
    }

    let question = "Tomatos! Fruit, vegetable, or both?";
    decider
        .clear_choices()
        .user_content(question)
        .add_choice("fruit")
        .add_choice("both");
    let res = decider.run_with_result().await?;
    if res.choice_value == "both" {
        println!(
            "🟢 correct custom response '{}' to test question: {question}",
            res.choice_value
        );
    } else {
        println!(
            "🔴 incorrect custom response '{}' to test question: {question}",
            res.choice_value
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::deciders::custom::apply_test_questions;
    use crate::*;
    use anyhow::Result;

    #[tokio::test]
    #[serial]
    pub async fn test_basic() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_basic_backend().custom();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_grammar() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_grammar_backend().custom();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_logit_bias() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_logit_bias_backend().custom();
        apply_test_questions(decider).await?;

        Ok(())
    }
}
