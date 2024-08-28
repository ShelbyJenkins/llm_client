use super::*;
use crate::{
    components::{base_request::BaseRequestConfig, instruct_prompt::InstructPrompt},
    primitives::*,
    BaseRequestConfigTrait,
    InstructPromptTrait,
};
use std::collections::HashMap;

const DYNAMIC_TEMPERATURE_MIN: f32 = 0.11;
const DYNAMIC_TEMPERATURE_MAX: f32 = 1.89;

pub struct Decision<D: DecisionTrait> {
    pub base_req: BaseLlmRequest,
    pub best_of_n_votes: u8,
    pub dynamic_temperature: bool,
    pub reason: D,
    pub result_can_be_none: bool,
}

impl<D: DecisionTrait> Decision<D> {
    pub async fn return_primitive(
        &mut self,
    ) -> Result<<D::ReasonPrimitive as PrimitiveTrait>::PrimitiveResult> {
        let res = self.return_result().await?;
        if let Some(primitive_result) = self
            .reason
            .primitive()
            .result_index_to_primitive(res.winner_index)?
        {
            Ok(primitive_result)
        } else {
            Err(anyhow::format_err!("No result returned."))
        }
    }
    pub async fn return_optional_primitive(
        &mut self,
    ) -> Result<Option<<D::ReasonPrimitive as PrimitiveTrait>::PrimitiveResult>> {
        let res = self.return_optional_result().await?;
        self.reason
            .primitive()
            .result_index_to_primitive(res.winner_index)
    }

    pub async fn return_result(&mut self) -> Result<DecisionResult> {
        self.result_can_be_none = false;
        self.run_decision().await
    }

    pub async fn return_optional_result(&mut self) -> Result<DecisionResult> {
        self.result_can_be_none = true;
        self.run_decision().await
    }

    async fn run_decision(&mut self) -> Result<DecisionResult> {
        let start = std::time::Instant::now();
        let mut decision_result = DecisionResult::new();
        let mut failed_attempts = 0;
        let mut none_count = 0;

        self.set_dynamic_temperature_on_initial(self.dynamic_temperature, self.best_of_n_votes);

        while failed_attempts < self.base_req.config.retry_after_fail_n_times {
            if failed_attempts >= self.base_req.config.retry_after_fail_n_times {
                break;
            }
            *self.reason.base_req_mut() = self.base_req.clone();
            let reason_result = match self
                .reason
                .return_reason_result(self.result_can_be_none)
                .await
            {
                Ok(reason_result) => reason_result,
                Err(_) => {
                    self.set_dynamic_temperature_on_fail(self.dynamic_temperature);
                    failed_attempts += 1;
                    continue;
                }
            };

            match self.reason.primitive().parse_reason_result(&reason_result) {
                Err(_) => {
                    self.set_dynamic_temperature_on_fail(self.dynamic_temperature);
                    failed_attempts += 1;
                }
                Ok(primitive_result) => {
                    decision_result.total_votes += 1;
                    if let Some(result_index) = reason_result.result_index {
                        *decision_result.votes.entry(result_index).or_insert(0) += 1;
                        for (choice_index, choice_votes) in &mut decision_result.votes {
                            if *choice_votes > decision_result.winner_votes {
                                decision_result.winner_votes = *choice_votes;
                                decision_result.winner_index = Some(*choice_index);
                            }
                        }
                    } else {
                        none_count += 1;
                    }
                    if decision_result.winner_votes
                        >= (self.best_of_n_votes + (self.best_of_n_votes % 2)) / 2
                    {
                        decision_result.confidence = decision_result.winner_votes as f32
                            / decision_result.total_votes as f32;
                        decision_result.duration = start.elapsed();
                        tracing::info!("{}", decision_result.to_string());

                        decision_result.winner_primitive_result =
                            Some(primitive_result.unwrap().to_string());

                        decision_result.reason_results.push(reason_result);

                        return Ok(decision_result);
                    } else if none_count >= (self.best_of_n_votes + (self.best_of_n_votes % 2)) / 2
                    {
                        decision_result.winner_votes = none_count;
                        decision_result.confidence =
                            none_count as f32 / decision_result.total_votes as f32;
                        decision_result.duration = start.elapsed();
                        tracing::info!("{}", decision_result.to_string());

                        decision_result.winner_primitive_result = Some("none".to_string());

                        decision_result.reason_results.push(reason_result);

                        return Ok(decision_result);
                    } else {
                        self.set_dynamic_temperature_on_success(
                            self.best_of_n_votes,
                            &decision_result,
                        );
                        decision_result.reason_results.push(reason_result);
                    }
                }
            }
        }
        Err(anyhow::format_err!(
            "BaseDecider: failed to get a valid response after {}",
            failed_attempts
        ))
    }

    fn set_dynamic_temperature_on_initial(
        &mut self,
        dynamic_temperature: bool,
        best_of_n_votes: u8,
    ) {
        if dynamic_temperature && best_of_n_votes > 1 {
            self.base_req.config.temperature = DYNAMIC_TEMPERATURE_MIN;
        }
    }

    fn set_dynamic_temperature_on_success(
        &mut self,
        best_of_n_votes: u8,
        decision_result: &DecisionResult,
    ) {
        let votes_required_to_win = (best_of_n_votes + (best_of_n_votes % 2)) / 2;
        if votes_required_to_win - decision_result.winner_votes == 1 {
            self.base_req.config.temperature = DYNAMIC_TEMPERATURE_MAX;
            return;
        }

        let minimum_votes_remaining = votes_required_to_win - decision_result.winner_votes;

        let maybe_average_votes_remaining =
            (votes_required_to_win + minimum_votes_remaining) as f32 / 2.0;

        self.base_req.config.temperature = self.base_req.config.temperature
            + ((DYNAMIC_TEMPERATURE_MAX - self.base_req.config.temperature)
                / maybe_average_votes_remaining);
    }

    fn set_dynamic_temperature_on_fail(&mut self, dynamic_temperature: bool) {
        if dynamic_temperature {
            self.base_req.config.temperature += DYNAMIC_TEMPERATURE_MIN;
        }
    }

    /// Sets the number of votes to reach consensus. It is the maxium number of votes for a decision, but often the decision is reached before this number is reached.
    /// For example, with the default of `3` votes, the first decision is made after 2 votes for a choice.
    /// If given an even number, it will round up to the nearest odd number.
    pub fn best_of_n_votes(&mut self, best_of_n_votes: u8) -> &mut Self {
        if best_of_n_votes % 2 == 0 {
            self.best_of_n_votes = best_of_n_votes + 1;
        } else {
            self.best_of_n_votes = best_of_n_votes;
        }
        self
    }

    /// Dynamically scales temperature during the voting process. Starts at a low temperature and increases towards max temperature as the number of votes increases.
    pub fn dynamic_temperature(&mut self, dynamic_temperature: bool) -> &mut Self {
        self.dynamic_temperature = dynamic_temperature;
        self
    }
}

#[allow(async_fn_in_trait)]
pub trait DecisionTrait: Sized {
    type ReasonPrimitive: PrimitiveTrait + ReasonTrait;
    fn base_req(&self) -> &BaseLlmRequest;

    fn base_req_mut(&mut self) -> &mut BaseLlmRequest;

    fn primitive(&self) -> &Self::ReasonPrimitive;

    async fn return_reason_result(&mut self, result_can_be_none: bool) -> Result<ReasonResult>;

    fn decision(self) -> Decision<Self> {
        Decision {
            base_req: self.base_req().clone(),
            best_of_n_votes: 3,
            dynamic_temperature: true,
            reason: self,
            result_can_be_none: false,
        }
    }
}

impl<D: DecisionTrait> BaseRequestConfigTrait for Decision<D> {
    fn base_config(&mut self) -> &mut BaseRequestConfig {
        &mut self.base_req.config
    }

    fn clear_request(&mut self) {
        self.base_req.reset_base_request();
    }
}

impl<D: DecisionTrait> InstructPromptTrait for Decision<D> {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.base_req.instruct_prompt
    }
}

#[derive(Clone)]
pub struct DecisionResult {
    pub votes: HashMap<u32, u8>,
    pub confidence: f32,
    pub duration: std::time::Duration,
    pub winner_primitive_result: Option<String>,
    pub reason_results: Vec<ReasonResult>,
    pub total_votes: u8,
    pub winner_votes: u8,
    pub winner_index: Option<u32>,
}

impl DecisionResult {
    fn new() -> Self {
        Self {
            votes: HashMap::new(),
            confidence: 0.0,
            duration: std::time::Duration::new(0, 0),
            winner_primitive_result: None,
            reason_results: Vec::new(),
            total_votes: 0,
            winner_votes: 0,
            winner_index: None,
        }
    }
}

impl std::fmt::Display for DecisionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f)?;
        writeln!(f, "\x1b[38;5;45m\x1b[1mDecision\x1b[0m",)?;
        for (i, res) in self.reason_results.iter().enumerate() {
            writeln!(f)?;
            writeln!(
                f,
                "\x1b[38;5;33m\x1b[1m{} {}\x1b[0m:",
                res.workflow.cascade_name,
                i + 1
            )?;
            writeln!(f)?;
            if let Some(primitive_result) = &res.primitive_result {
                writeln!(
                    f,
                    "\x1b[38;5;32mprimitive_result\x1b[0m: {}",
                    primitive_result
                )?;
            } else {
                writeln!(f, "\x1b[38;5;32mprimitive_result\x1b[0m: None")?;
            };
            writeln!(f, "\x1b[38;5;31mreason duration\x1b[0m: {:?}", res.duration)?;
            writeln!(
                f,
                "\x1b[38;5;30mreason temperature\x1b[0m: {:?}",
                res.temperature
            )?;
        }

        writeln!(f)?;
        writeln!(f)?;
        writeln!(f, "\x1b[38;5;45m\x1b[1mDecisionResult\x1b[0m:")?;
        writeln!(f)?;
        writeln!(
            f,
            "\x1b[38;5;44mvote results\x1b[0m: {} out of {} votes for winner.",
            self.winner_votes, self.total_votes
        )?;
        writeln!(f, "\x1b[38;5;44mconfidence\x1b[0m: {}", self.confidence)?;
        writeln!(
            f,
            "\x1b[38;5;43mdecision duration\x1b[0m: {:?}",
            self.duration
        )?;
        if let Some(winner_primitive_result) = &self.winner_primitive_result {
            writeln!(
                f,
                "\x1b[38;5;42m\x1b[1mDecision primitive result\x1b[0m: {}",
                winner_primitive_result
            )?;
        } else {
            writeln!(
                f,
                "\x1b[38;5;42mfs\x1b[1mdecision primitive result\x1b[0m: None"
            )?;
        }
        writeln!(f)
    }
}
