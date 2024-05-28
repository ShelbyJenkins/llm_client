pub mod boolean;
pub mod custom;
mod decider_backend;
pub mod integer;

use super::*;
pub use boolean::BooleanDecider;
pub use custom::CustomDecider;
pub use integer::IntegerDecider;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, PartialEq, Deserialize)]
pub enum DecisionParserType {
    LogitBias,
    Grammar,
    Basic,
}

#[derive(Clone)]
pub struct DeciderConfig {
    pub best_of_n_votes: u8,
    pub dynamic_temperature: bool,
    pub decision_parser_type: DecisionParserType,
    pub decision_justification_token_count: u32,
    pub justification_prompt: String,
    pub max_tokens_for_parser: u32,
    pub decision_parser_prompt: String,
    pub grammar: Option<String>,
}

impl DeciderConfig {
    fn new(backend: &LlmBackend) -> Self {
        let decision_parser_type = Self::default_decision_parser_type(backend);
        Self {
            best_of_n_votes: 3,
            dynamic_temperature: true,
            decision_justification_token_count: 300,
            justification_prompt: String::default(),
            max_tokens_for_parser: 0,
            decision_parser_prompt: String::default(),
            decision_parser_type,
            grammar: None,
        }
    }
    fn default_decision_parser_type(backend: &LlmBackend) -> DecisionParserType {
        match backend {
            LlmBackend::Llama(_) => DecisionParserType::Grammar,
            // LlmBackend::MistralRs(_) => DecisionParserType::Basic,
            LlmBackend::OpenAi(_) => DecisionParserType::LogitBias,
            LlmBackend::Anthropic(_) => DecisionParserType::Basic,
        }
    }
    fn check_and_set_parser_max_tokens(mut self) -> Self {
        if self.max_tokens_for_parser == 0 {
            self.max_tokens_for_parser = match self.decision_parser_type {
                DecisionParserType::LogitBias => 1,
                DecisionParserType::Grammar => 100,
                DecisionParserType::Basic => 6,
            }
        }
        self
    }
    fn get_grammar(&self) -> Option<&str> {
        self.grammar.as_deref()
    }
}

pub struct Decider<'a> {
    pub decider_config: DeciderConfig,
    pub llm_client: &'a LlmClient,
}
impl<'a> Decider<'a> {
    pub fn new(llm_client: &'a LlmClient) -> Self {
        Self {
            decider_config: DeciderConfig::new(&llm_client.backend),
            llm_client,
        }
    }

    /// A decision where the answer is True/Yes/Affirmative or Falce/No/Negative. Returns a `bool`.
    pub fn boolean(self) -> BooleanDecider<'a> {
        BooleanDecider::new(
            self.decider_config.check_and_set_parser_max_tokens(),
            self.llm_client.default_request_config.clone(),
            self.llm_client,
        )
    }

    /// A decision where the answer is an integer from a range of integers. Returns an `u16`.
    pub fn integer(self) -> IntegerDecider<'a> {
        IntegerDecider::new(
            self.decider_config.check_and_set_parser_max_tokens(),
            self.llm_client.default_request_config.clone(),
            self.llm_client,
        )
    }

    /// A decision where the answer is a word from a list of custom words. Returns a `String`.
    /// Limited support for logit bias backend because the logit bias backend requires a single token for an answer.
    pub fn custom(self) -> CustomDecider<'a> {
        CustomDecider::new(
            self.decider_config.check_and_set_parser_max_tokens(),
            self.llm_client.default_request_config.clone(),
            self.llm_client,
        )
    }

    /// Sets the number of votes to reach consensus. It is the maxium number of votes for a decision, but often the decision is reached before this number is reached.
    /// For example, with the default of `3` votes, the first decision is made after 2 votes for a choice.
    /// If given an even number, it will round up to the nearest odd number.
    pub fn best_of_n_votes(mut self, best_of_n_votes: u8) -> Self {
        self.decider_config.best_of_n_votes = best_of_n_votes;
        self
    }

    /// Dynamically scales temperature during the voting process. Starts at a low temperature and increases towards max temperature as the number of votes increases.
    pub fn dynamic_temperature(mut self, dynamic_temperature: bool) -> Self {
        self.decider_config.dynamic_temperature = dynamic_temperature;
        self
    }

    /// Sets the number of tokens to use for the decision justification. This, along with the prompt, controls the verbosity and perhaps the decision making ability, of the decider.
    pub fn decision_justification_token_count(mut self, token_count: u32) -> Self {
        self.decider_config.decision_justification_token_count = token_count;
        self
    }

    /// Likely should not be used.
    pub fn max_tokens_for_parser(mut self, token_count: u32) -> Self {
        self.decider_config.max_tokens_for_parser = token_count;
        self
    }

    /// Overides default to use the basic backend for decision parsing. The basic backend uses plain text parsing.
    pub fn use_basic_backend(mut self) -> Self {
        self.decider_config.decision_parser_type = DecisionParserType::Basic;
        self
    }

    /// Overides default to use the grammar backend for decision parsing. The grammar backend constricts the decision to a specific grammar schema. This is the superior choice for decision parsing.
    pub fn use_grammar_backend(mut self) -> Self {
        match &self.llm_client.backend {
            LlmBackend::Llama(_) => {
                self.decider_config.decision_parser_type = DecisionParserType::Grammar;
            }
            // LlmBackend::MistralRs(_) => todo!(),
            LlmBackend::OpenAi(_) => {
                panic!("OpenAI backend is not supported for Grammar based calls.")
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for Grammar based calls.")
            }
        }
        self
    }

    /// Overides default to use the logit bias backend for decision parsing. The logit bias backend uses the logit bias to constrict the decision to specific tokens.
    pub fn use_logit_bias_backend(mut self) -> Self {
        match &self.llm_client.backend {
            LlmBackend::Llama(_) => {
                self.decider_config.decision_parser_type = DecisionParserType::LogitBias;
            }
            // LlmBackend::MistralRs(_) => todo!(),
            LlmBackend::OpenAi(_) => {
                self.decider_config.decision_parser_type = DecisionParserType::LogitBias;
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for LogitBias based calls.")
            }
        }
        self
    }
}
