use super::*;
use std::collections::HashMap;

mod parser_basic;
mod parser_grammar;
mod parser_logit_bias;

use parser_basic::make_basic_parser_request;
pub use parser_grammar::create_parser_grammar;
use parser_grammar::make_grammar_parser_request;
use parser_logit_bias::make_logit_bias_request;
pub use parser_logit_bias::{generate_parser_logit_bias, generate_parser_logit_bias_for_backend};

pub const DYNAMIC_TEMPERATURE_MIN: f32 = 0.11;
pub const DYNAMIC_TEMPERATURE_MAX: f32 = 1.89;

#[derive(Clone, Debug)]
pub struct DeciderChoice {
    pub choice_value: String,
    pub parser_key: String,
    pub choice_index: u16,
}

#[derive(Debug, Clone)]
pub struct DeciderResult {
    pub choice_value: String,
    pub choice_index: u16,
    pub votes_for_choice: u8,
    pub total_decision_votes: u8,
    pub confidence: f32,
    pub justifications: Vec<String>,
    pub choice_vote_counts: HashMap<u16, u8>,
    pub failed_attempts: u8,
}
impl Default for DeciderResult {
    fn default() -> Self {
        Self {
            choice_value: "".to_string(),
            choice_index: 0,
            votes_for_choice: 0,
            total_decision_votes: 0,
            confidence: 0.0,
            justifications: vec![],
            choice_vote_counts: HashMap::new(),
            failed_attempts: 0,
        }
    }
}

pub trait DecisionThing {
    fn get_decider_result(&self) -> &DeciderResult;

    fn set_decider_result(&mut self) -> &mut DeciderResult;

    fn decider_config(&self) -> &DeciderConfig;

    fn decider_choices(&self) -> &Vec<DeciderChoice>;

    fn req_config(&self) -> &RequestConfig;

    fn llm_client(&self) -> &LlmClient;

    fn set_temperature(&mut self, value: f32);

    fn validate_basic_parser_response(&self, response: Result<String>) -> Result<u16>;

    async fn run_decision(&mut self) -> Result<()> {
        self.set_dynamic_temperature_on_initial();

        while self.get_decider_result().failed_attempts < self.req_config().retry_after_fail_n_times
        {
            if self.get_decider_result().failed_attempts
                >= self.req_config().retry_after_fail_n_times
            {
                break;
            }

            let justification = self
                .make_decision_justification_request(&self.decider_config().justification_prompt)
                .await?;

            self.set_decider_result()
                .justifications
                .push(justification.clone());

            let parser_response = self.run_decision_parser(&justification).await;

            if let Ok(result) = parser_response {
                self.set_decider_result().total_decision_votes += 1;
                *self
                    .set_decider_result()
                    .choice_vote_counts
                    .entry(result)
                    .or_insert(0) += 1;

                for (choice_index, choice_votes) in
                    self.get_decider_result().choice_vote_counts.clone()
                {
                    if choice_votes > self.get_decider_result().votes_for_choice {
                        self.set_decider_result().votes_for_choice = choice_votes;
                        self.set_decider_result().choice_index = choice_index;
                    }
                }

                if self.get_decider_result().votes_for_choice
                    >= (self.decider_config().best_of_n_votes
                        + (self.decider_config().best_of_n_votes % 2))
                        / 2
                {
                    let decider_choices = self.decider_choices().clone();
                    let winner = decider_choices
                        .iter()
                        .find(|choice| {
                            choice.choice_index == self.get_decider_result().choice_index
                        })
                        .unwrap();

                    self.set_decider_result()
                        .choice_value
                        .clone_from(&winner.choice_value);

                    self.set_decider_result().confidence =
                        self.get_decider_result().votes_for_choice as f32
                            / self.get_decider_result().total_decision_votes as f32;

                    if self.llm_client().backend.logging_enabled() {
                        let decider_result = self.get_decider_result();
                        tracing::info!(?decider_result);
                    }
                    return Ok(());
                } else {
                    self.set_dynamic_temperature_on_success();
                }
            } else {
                self.set_dynamic_temperature_on_fail();
                self.set_decider_result().failed_attempts += 1;
            }
        }
        Err(anyhow::format_err!(
            "BaseDecider: failed to get a valid response after {}",
            self.get_decider_result().failed_attempts
        ))
    }

    async fn make_decision_justification_request(&self, system_prompt: &str) -> Result<String> {
        let mut text = self.llm_client().text().basic_text();
        text.req_config = self.req_config().clone();
        text.req_config.system_content = Some(system_prompt.to_string());
        text.max_tokens(self.decider_config().decision_justification_token_count);
        text.run().await
    }

    async fn run_decision_parser(&mut self, justification: &str) -> Result<u16> {
        let mut parser_req_config = self.req_config().clone();

        parser_req_config.system_content =
            Some(self.decider_config().decision_parser_prompt.clone());

        parser_req_config.user_content = Some(format!(" {}", justification));
        parser_req_config.system_content_path = None;

        parser_req_config
            .build_request(&self.llm_client().backend)
            .await?;

        self.set_dynamic_temperature_on_initial_parse_attempt(&mut parser_req_config);

        let mut max_tokens_for_parser = self.decider_config().max_tokens_for_parser;
        let mut failed_attempts = 0;
        while failed_attempts < parser_req_config.retry_after_fail_n_times {
            let response = match self.decider_config().decision_parser_type {
                DecisionParserType::Basic => {
                    parser_req_config
                        .set_max_tokens_for_request_for_decision(max_tokens_for_parser)?;
                    let response =
                        make_basic_parser_request(self.llm_client(), &parser_req_config).await;
                    self.validate_basic_parser_response(response)
                }
                DecisionParserType::Grammar => {
                    parser_req_config
                        .set_max_tokens_for_request_for_decision(max_tokens_for_parser)?;
                    let response = make_grammar_parser_request(
                        self.llm_client(),
                        &parser_req_config,
                        self.decider_config().get_grammar(),
                    )
                    .await;
                    self.validate_grammar_logit_bias_parser_response(response)
                }
                DecisionParserType::LogitBias => {
                    parser_req_config.set_max_tokens_for_request_for_decision(1)?;
                    let response =
                        make_logit_bias_request(self.llm_client(), &parser_req_config).await;

                    self.validate_grammar_logit_bias_parser_response(response)
                }
            };

            match response {
                Ok(result) => {
                    return Ok(result);
                }
                Err(_) => {
                    failed_attempts += 1;
                    if failed_attempts >= parser_req_config.retry_after_fail_n_times {
                        break;
                    }
                    self.set_dynamic_temperature_on_parser_fail(
                        failed_attempts,
                        &mut parser_req_config,
                    );
                    match self.decider_config().decision_parser_type {
                        DecisionParserType::Basic => {
                            max_tokens_for_parser += 4;
                        }
                        DecisionParserType::Grammar => {
                            max_tokens_for_parser += 10;
                        }
                        DecisionParserType::LogitBias => {}
                    };

                    continue;
                }
            };
        }
        let error = format!(
            "Matcher run error: failed to get a valid response after {}",
            failed_attempts
        );
        if self.llm_client().backend.logging_enabled() {
            tracing::info!(?error);
        }
        Err(anyhow!("{}", error))
    }

    fn validate_grammar_logit_bias_parser_response(&self, response: Result<String>) -> Result<u16> {
        match response {
            Ok(response) => {
                let mut matches: Vec<u16> = Vec::new();
                let response = response.trim();
                for choice in self.decider_choices() {
                    if response.contains(&choice.parser_key) {
                        matches.push(choice.choice_index);
                    }
                }

                if matches.len() > 1 {
                    let error = format!(
                        "validate_grammar_logit_bias_parser_response error: more than one matching parser_key in response. matches.len() == {}",
                        matches.len()
                    );
                    if self.llm_client().backend.logging_enabled() {
                        tracing::info!(?error);
                    }
                    Err(anyhow!("{}", error))
                } else if matches.is_empty() {
                    let error =
                        "validate_grammar_logit_bias_parser_response error: no matching parser_key in response.";
                    if self.llm_client().backend.logging_enabled() {
                        tracing::info!(?error);
                    }
                    Err(anyhow!("{}", error))
                } else {
                    Ok(matches.remove(0))
                }
            }
            Err(error) => Err(anyhow!("{}", error)),
        }
    }

    fn set_dynamic_temperature_on_initial(&mut self) {
        if self.decider_config().dynamic_temperature {
            self.set_temperature(DYNAMIC_TEMPERATURE_MIN);
        }
    }

    fn set_dynamic_temperature_on_success(&mut self) {
        let votes_required_to_win = (self.decider_config().best_of_n_votes
            + (self.decider_config().best_of_n_votes % 2))
            / 2;
        // if votes_required_to_win - self.decider_result().votes_for_choice == 1 {
        //     self.set_temperature(DYNAMIC_TEMPERATURE_MAX);
        //     return;
        // }
        let maximum_votes_remaining = (self.decider_choices().len() as u8 * votes_required_to_win)
            - self.get_decider_result().total_decision_votes;
        let minimum_votes_remaining =
            votes_required_to_win - self.get_decider_result().votes_for_choice;
        let average_votes_remaining =
            (maximum_votes_remaining + minimum_votes_remaining) as f32 / 2.0;

        self.set_temperature(
            self.req_config().temperature
                + ((DYNAMIC_TEMPERATURE_MAX - self.req_config().temperature)
                    / average_votes_remaining),
        )
    }

    fn set_dynamic_temperature_on_fail(&mut self) {
        if self.decider_config().dynamic_temperature {
            self.set_temperature(self.req_config().temperature + DYNAMIC_TEMPERATURE_MIN);
        }
    }

    fn set_dynamic_temperature_on_initial_parse_attempt(
        &self,
        parser_req_config: &mut RequestConfig,
    ) {
        if self.decider_config().dynamic_temperature {
            parser_req_config.temperature = 0.77;
        }
    }

    fn set_dynamic_temperature_on_parser_fail(
        &self,
        failed_attempts: u8,
        parser_req_config: &mut RequestConfig,
    ) {
        if !self.decider_config().dynamic_temperature {
            return;
        }
        // math:
        // let maximum_votes_remaining = self.retry_after_fail_n_times() - failed_attempts;
        // let minimum_votes_remaining = 1;
        // let average_votes_remaining =
        //     (maximum_votes_remaining + minimum_votes_remaining) as f32 / 2.0;

        parser_req_config.temperature = self.req_config().temperature
            + ((DYNAMIC_TEMPERATURE_MAX - self.req_config().temperature)
                / (self.req_config().retry_after_fail_n_times - failed_attempts + 1) as f32
                / 2.0);
    }
}
