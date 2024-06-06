use self::decider_backend::*;
use super::*;
use num2words::Num2Words;

impl<'a> RequestConfigTrait for IntegerDecider<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

const INTEGER_JUSTIFICATION_PROMPT: &str = r#"The response should contain your justification and end with "The answer is:". You are answering a question with a number for an answer. Choose the best choice. Do not list the choices. Do not restate the question. Do not annotate. Justify your response succinctly."#;

const INTEGER_BASIC_PARSER_PROMPT: &str = r#"The user answered a multiple choice question in plain spoken english. Do not explain the user's answer. Do not correct the user's answer. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the multiple choice answer matching the users answer."#;

const INTEGER_GRAMMER_PARSER_PROMPT: &str = r#"The user answered a multiple choice question. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the multiple choice answer matching the users answer."#;

const INTEGER_LOGIT_BIAS_PARSER_PROMPT: &str = r#"The user answered a multiple choice question. The user's answer will match one of the multiple choice answers. Do not list the choices. Respond only with the the multiple choice answer matching the users answer."#;

#[derive(Clone)]
pub struct IntegerDecider<'a> {
    pub lower_bound: u16,
    pub upper_bound: u16,
    pub decider_choices: Vec<DeciderChoice>,
    pub decider_config: DeciderConfig,
    pub decider_result: Option<DeciderResult>,
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
}

impl<'a> IntegerDecider<'a> {
    pub fn new(
        decider_config: DeciderConfig,
        default_request_config: RequestConfig,
        llm_client: &'a LlmClient,
    ) -> Self {
        Self {
            lower_bound: 1,
            upper_bound: 9,
            decider_choices: Vec::new(),
            decider_config,
            decider_result: None,
            req_config: default_request_config,
            llm_client,
        }
    }

    /// Set the lower bound of the integer range. Default is 1.
    pub fn lower_bound(&mut self, lower_bound: u16) -> &mut Self {
        self.lower_bound = lower_bound;
        self.decider_choices = Vec::new();
        self
    }

    /// Set the upper bound of the integer range. Default is 9.
    pub fn upper_bound(&mut self, upper_bound: u16) -> &mut Self {
        self.upper_bound = upper_bound;
        self.decider_choices = Vec::new();
        self
    }

    /// Run the decider and return the complete DeciderResult.
    /// The integer response is available in the DeciderResult.choice_index field.
    pub async fn run_with_result(&mut self) -> Result<DeciderResult> {
        if self.lower_bound > self.upper_bound {
            panic!("Lower bound cannot be greater than upper bound.")
        }
        self.decider_result = Some(DeciderResult::default());

        if self.decider_choices.is_empty() {
            for i in self.lower_bound..=self.upper_bound {
                let number = Num2Words::new(i).to_words().map_err(|e| anyhow!("{}", e))?;
                let choice = DeciderChoice {
                    parser_key: self.create_parser_key(&number, i),
                    choice_value: number,
                    choice_index: i,
                };

                self.decider_choices.push(choice);
            }
            self.req_config.logit_bias = None;
            self.decider_config.grammar = None;
        }

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

    /// Run the decider and return integer response.
    pub async fn run(&mut self) -> Result<u16> {
        let res = self.run_with_result().await?;
        Ok(res.choice_index)
    }

    fn create_parser_key(&self, choice_value: &str, choice_index: u16) -> String {
        match self.decider_config.decision_parser_type {
            DecisionParserType::Basic => {
                format!("{}). {},", choice_index, choice_value)
            }
            DecisionParserType::Grammar => {
                format!("The answer is {}). {}", choice_index, choice_value)
            }
            DecisionParserType::LogitBias => choice_index.to_string(),
        }
    }

    fn create_prompt_choice(&self, choice: &DeciderChoice) -> String {
        match self.decider_config.decision_parser_type {
            DecisionParserType::Grammar => {
                format!(" {}). {},", choice.choice_index, choice.choice_value)
            }
            DecisionParserType::Basic => {
                format!(" {}). {},", choice.choice_index, choice.choice_value)
            }
            DecisionParserType::LogitBias => {
                format!(" {}). {},", choice.choice_index, choice.choice_value)
            }
        }
    }

    fn create_justification_prompt(&mut self) {
        let mut justification_prompt = INTEGER_JUSTIFICATION_PROMPT.to_string();
        if let Some(existing_system_content) = &self.req_config().system_content {
            justification_prompt.push_str(&format!(
                "additional instructions: {}",
                existing_system_content
            ));
        }

        justification_prompt.push_str("\nchoices:");
        for choice in self.decider_choices() {
            justification_prompt.push_str(&self.create_prompt_choice(choice));
        }

        self.decider_config.justification_prompt = justification_prompt;
    }

    fn create_parser_prompt(&mut self) {
        let mut decision_parser_prompt = match self.decider_config.decision_parser_type {
            DecisionParserType::Grammar => INTEGER_GRAMMER_PARSER_PROMPT.to_string(),
            DecisionParserType::Basic => INTEGER_BASIC_PARSER_PROMPT.to_string(),
            DecisionParserType::LogitBias => INTEGER_LOGIT_BIAS_PARSER_PROMPT.to_string(),
        };
        decision_parser_prompt.push_str("\nchoices:");
        for choice in self.decider_choices() {
            decision_parser_prompt.push_str(&self.create_prompt_choice(choice));
        }

        self.decider_config.decision_parser_prompt = decision_parser_prompt;
    }
}

impl DecisionThing for IntegerDecider<'_> {
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
                let response = response.trim().to_lowercase();

                let mut matches: HashMap<u16, u8> = HashMap::new();

                let mut result = "IntegerDecider validate_basic_parser_response:".to_string();

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
                // If no matches found by value, check for the number. Less reliably, as it 1 could match 10, 11, 12, etc.
                // Need to parse with regex in the future.
                if matches.is_empty() {
                    for choice in &self.decider_choices {
                        let count =
                            response.matches(&choice.choice_index.to_string()).count() as u8;
                        if count > 0 {
                            result.push_str(&format!(
                                " matches found for choice_index: {} count: {}",
                                choice.choice_index, count
                            ));
                            matches.insert(choice.choice_index, count);
                        }
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

pub async fn apply_test_questions(mut decider: IntegerDecider<'_>) -> Result<()> {
    let question = "How many twenty-five cent quarters are in an American dollar?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_index == 4 {
        println!(
            "🟢 correct integer response '{}' to test question: {question}",
            res.choice_index
        );
    } else {
        println!(
            "🔴 incorrect integer response '{}' to test question: {question}",
            res.choice_index
        );
    }

    let question = "What is 1 + 1 equal to?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_index == 2 {
        println!(
            "🟢 correct integer response '{}' to test question: {question}",
            res.choice_index
        );
    } else {
        println!(
            "🔴 incorrect integer response '{}' to test question: {question}",
            res.choice_index
        );
    }

    let question = "I have three dogs. One ran away. How many dogs do I have?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_index == 2 {
        println!(
            "🟢 correct integer response '{}' to test question: {question}",
            res.choice_index
        );
    } else {
        println!(
            "🔴 incorrect integer response '{}' to test question: {question}",
            res.choice_index
        );
    }

    let question = "I turn 10 years old in 2 years. How old am I now?";
    decider.user_content(question);
    let res = decider.lower_bound(4).run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_index == 8 {
        println!(
            "🟢 correct integer response '{}' to test question: {question}",
            res.choice_index
        );
    } else {
        println!(
            "🔴 incorrect integer response '{}' to test question: {question}",
            res.choice_index
        );
    }

    let question = "I have have a cat and three dogs. When I got the cat it had all of it's nine lives. Each dog has taken one of the cat's lives. How many lives did my cat have when I got it?";
    decider.user_content(question);
    let res = decider.run_with_result().await?;
    println!("Response:\n {:?}\n", res);
    if res.choice_index == 9 {
        println!(
            "🟢 correct integer response '{}' to test question: {question}",
            res.choice_index
        );
    } else {
        println!(
            "🔴 incorrect integer response '{}' to test question: {question}",
            res.choice_index
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::deciders::integer::apply_test_questions;
    use crate::*;
    use anyhow::Result;

    #[tokio::test]
    #[serial]
    pub async fn test_basic() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_basic_backend().integer();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_grammar() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_grammar_backend().integer();
        apply_test_questions(decider).await?;

        Ok(())
    }

    #[tokio::test]
    #[serial]
    pub async fn test_logit_bias() -> Result<()> {
        let llm_client = LlmClient::llama_backend().init().await?;

        let decider = llm_client.decider().use_logit_bias_backend().integer();
        apply_test_questions(decider).await?;

        Ok(())
    }
}
