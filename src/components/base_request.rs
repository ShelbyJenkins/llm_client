use super::{
    cascade::step::InferenceStep,
    constraints::{
        logit_bias::LogitBias,
        stop_sequence::{StopSequences, StoppingSequence},
    },
    instruct_prompt::InstructPrompt,
    response::{LlmClientResponse, LlmClientResponseError, LlmClientResponseStopReason},
};
use crate::llm_backends::LlmBackend;
use anyhow::{anyhow, Result};
use llm_utils::prompting::{
    token_count::{check_and_get_max_tokens, RequestTokenLimitError},
    LlmPrompt,
};
use std::{rc::Rc, thread::sleep, time::Duration};

#[derive(Clone)]
pub struct BaseRequestConfig {
    // Tokens
    pub ctx_size: u32,        // ctx_size is the limit of input + output
    pub ctx_output_size: u32, // ctx_output_size is the maximum inference output tokens
    pub requested_response_tokens: Option<u32>, // requested_response_tokens is the maximum output tokens requested by the user
    pub safety_tokens: u32, // safety_tokens is a buffer to ensure the model doesn't go over the limit
    pub actual_request_tokens: Option<u32>, // actual_request_tokens is an adjusted value of requested_response_tokens to ensure the request stays within the model's limits
    // Inference
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: f32,
    pub retry_after_fail_n_times: u8,
    pub increase_limit_on_stopped_limit: bool,
    pub cache_prompt: bool,
    pub backend: Rc<LlmBackend>,
}

impl BaseRequestConfig {
    pub fn new(backend: &Rc<LlmBackend>) -> Self {
        let backend = Rc::clone(backend);
        let (ctx_size, ctx_output_size) = match backend.as_ref() {
            LlmBackend::Llama(backend) => (
                backend.server_config.ctx_size,
                backend.server_config.ctx_size,
            ),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(backend) => (backend.ctx_size, backend.ctx_size),
            LlmBackend::OpenAi(backend) => (
                backend.model.context_length,
                backend.model.max_tokens_output,
            ),
            LlmBackend::Anthropic(backend) => (
                backend.model.context_length,
                backend.model.max_tokens_output,
            ),
            LlmBackend::Perplexity(backend) => (
                backend.model.context_length,
                backend.model.max_tokens_output,
            ),
        };

        Self {
            ctx_size,
            ctx_output_size,
            requested_response_tokens: None,
            actual_request_tokens: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            temperature: 1.0,
            top_p: 1.0,
            safety_tokens: 10,
            retry_after_fail_n_times: 3,
            increase_limit_on_stopped_limit: false,
            cache_prompt: false,
            backend,
        }
    }

    pub fn new_prompt(&self) -> LlmPrompt {
        match self.backend.as_ref() {
            LlmBackend::Llama(b) => LlmPrompt::new_from_os_llm(&b.model),
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(b) => LlmPrompt::new_from_os_llm(&b.model),
            LlmBackend::OpenAi(b) => LlmPrompt::new_from_openai_llm(&b.model),
            LlmBackend::Perplexity(b) => LlmPrompt::new_from_openai_llm(&b.model),
            LlmBackend::Anthropic(b) => LlmPrompt::new_from_anthropic_llm(&b.model),
        }
    }
}

pub trait BaseRequestConfigTrait {
    fn base_config(&mut self) -> &mut BaseRequestConfig;

    fn clear_request(&mut self);

    /// Number of tokens to use for the model's output. Not nessecarily what the model will use, but the maxium it's allowed to use.
    /// Before the request is built, the total input (prompt) tokens and the requested output (max_tokens) are used to ensure the request stays within the model's limits.
    fn max_tokens(&mut self, max_tokens: u32) -> &mut Self {
        self.base_config().requested_response_tokens = Some(max_tokens);
        self
    }

    /// Number of retries to attempt after a failure.
    /// Default is 3.
    fn retry_after_fail_n_times(&mut self, retry_after_fail_n_times: u8) -> &mut Self {
        self.base_config().retry_after_fail_n_times = retry_after_fail_n_times;
        self
    }

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        match frequency_penalty {
            value if (-2.0..=2.0).contains(&value) => self.base_config().frequency_penalty = value,
            _ => self.base_config().frequency_penalty = 0.0,
        };
        self
    }

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    ///
    /// [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
    fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        match presence_penalty {
            value if (-2.0..=2.0).contains(&value) => self.base_config().presence_penalty = value,
            _ => self.base_config().presence_penalty = 0.0,
        };
        self
    }

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    fn temperature(&mut self, temperature: f32) -> &mut Self {
        match temperature {
            value if (0.0..=2.0).contains(&value) => self.base_config().temperature = value,
            _ => self.base_config().temperature = 1.0,
        };
        self
    }

    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    ///
    ///  We generally recommend altering this or `temperature` but not both.
    fn top_p(&mut self, top_p: f32) -> &mut Self {
        match top_p {
            value if (0.0..=2.0).contains(&value) => self.base_config().top_p = value,
            _ => self.base_config().top_p = 1.0,
        };
        self
    }
}

#[derive(Clone)]
pub struct BaseLlmRequest {
    pub stop_sequences: StopSequences,
    pub grammar_string: Option<String>,
    pub logit_bias: Option<LogitBias>,
    pub instruct_prompt: InstructPrompt,
    pub config: BaseRequestConfig,
}

impl BaseLlmRequest {
    pub fn new(config: BaseRequestConfig) -> Self {
        Self {
            config: config.clone(),
            grammar_string: None,
            logit_bias: None,
            stop_sequences: StopSequences::default(),
            instruct_prompt: InstructPrompt::new(&config),
        }
    }

    pub fn new_from_backend(backend: &Rc<LlmBackend>) -> Self {
        let config = BaseRequestConfig::new(backend);
        Self {
            grammar_string: None,
            logit_bias: None,
            stop_sequences: StopSequences::default(),
            instruct_prompt: InstructPrompt::new(&config),
            config,
        }
    }

    pub fn reset_base_request(&mut self) {
        self.instruct_prompt.reset_instruct_prompt();
        self.instruct_prompt.prompt.reset_prompt();
        self.stop_sequences.sequences.clear();
        self.grammar_string = None;
        self.logit_bias = None;
    }

    pub fn set_base_req_stop_sequences(
        &mut self,
        stop_word_done: &Option<String>,
        stop_word_null_result: &Option<String>,
    ) {
        if stop_word_done.is_some() || stop_word_null_result.is_some()
        // || step.stop_word_steps_done.is_some()
        {
            self.stop_sequences.required = true;
            self.stop_sequences.sequences.clear();
        }
        if let Some(stop_word_done) = &stop_word_done {
            self.stop_sequences.set_stop_word_done(stop_word_done);
        }

        if let Some(null_result_stop_word) = &stop_word_null_result {
            self.stop_sequences
                .set_stop_word_null_result(null_result_stop_word);
        }

        // if step.stop_word_steps_done {
        //     self.set_req_stop_word_steps_done();
        // }
    }

    pub async fn cascade_request(&mut self, step: &mut InferenceStep) -> Result<()> {
        let mut failed_attempts = 0;
        while failed_attempts < self.config.retry_after_fail_n_times {
            let res: LlmClientResponse = self.base_llm_client_request().await?;
            if matches!(
                res.stop_reason,
                LlmClientResponseStopReason::StoppingSequence(StoppingSequence::NullResult(_))
            ) {
                step.llm_content = None;
                return Ok(());
            }

            match step.step_config.grammar.validate_clean(&res.content) {
                Ok(content) => {
                    step.llm_content = Some(content.clone());
                    return Ok(());
                }
                Err(e) => {
                    tracing::info!(?e);

                    failed_attempts += 1;
                    continue;
                }
            }
        }
        Err(anyhow!(
            "Failed to parse response as primitive after {} attempts",
            self.config.retry_after_fail_n_times
        ))
    }

    pub async fn set_cache_request(&self, clear: bool) -> Result<()> {
        let mut errors: Vec<LlmClientResponseError> = Vec::new();
        for i in 1..(self.config.retry_after_fail_n_times) {
            let res = match self.config.backend.as_ref() {
                LlmBackend::Llama(b) => b.set_cache(clear, self).await,
                _ => unimplemented!(),
            };
            match res {
                Err(error) => match error {
                    LlmClientResponseError::InferenceError { .. } => {
                        sleep(Duration::from_millis(2u64.pow(i.into()) * 1000));
                        tracing::info!(?error);
                        errors.push(error);
                    }
                    LlmClientResponseError::RequestBuilderError { .. } => {
                        return Err(anyhow!("set_cache_request error: {}", error))
                    }
                    _ => unreachable!(),
                },
                Ok(_) => return Ok(()),
            }
        }

        Err(anyhow!(
            "set_cache_request error: Failed to get a response after {} retries,\nerrors:\n{:?}",
            self.config.retry_after_fail_n_times,
            errors
        ))
    }

    pub async fn base_llm_client_request(
        &mut self,
    ) -> Result<LlmClientResponse, LlmClientResponseError> {
        self.set_max_tokens_for_request().map_err(|e| {
            LlmClientResponseError::RequestBuilderError {
                error: format!("base_llm_client_request builder error: {}", e),
            }
        })?;
        self.build_logit_bias()
            .await
            .map_err(|e| LlmClientResponseError::RequestBuilderError {
                error: format!("base_llm_client_request builder error: {}", e),
            })?;
        let mut errors: Vec<LlmClientResponseError> = Vec::new();
        for i in 1..(self.config.retry_after_fail_n_times) {
            match self.request().await {
                Err(error) => match error {
                    LlmClientResponseError::InferenceError { .. } => {
                        sleep(Duration::from_millis(2u64.pow(i.into()) * 1000));
                        tracing::info!(?error);
                        errors.push(error);
                    }
                    LlmClientResponseError::RequestBuilderError { .. } => return Err(error),
                    _ => unreachable!(),
                },
                Ok(res) => {
                    if let Some(error) = res.error {
                        tracing::info!(?error);
                        match error {
                            LlmClientResponseError::StopSequenceError { .. }
                            | LlmClientResponseError::StopSequenceReasonWithoutValue { .. } => {
                                if res.stop_reason == LlmClientResponseStopReason::StopLimit {
                                    self.increase_limit().map_err(|e| {
                                        LlmClientResponseError::RequestBuilderError {
                                            error: format!(
                                                "base_llm_client_request builder error: {}",
                                                e
                                            ),
                                        }
                                    })?;
                                }
                                errors.push(error);
                            }
                            LlmClientResponseError::StopLimitError { .. } => {
                                self.increase_limit().map_err(|e| {
                                    LlmClientResponseError::RequestBuilderError {
                                        error: format!(
                                            "base_llm_client_request builder error: {}",
                                            e
                                        ),
                                    }
                                })?;
                                errors.push(error);
                            }
                            LlmClientResponseError::UnknownStopReason { .. } => {
                                errors.push(error);
                            }
                            _ => unimplemented!(),
                        }
                    } else {
                        return Ok(res);
                    }
                }
            }
        }

        let error = format!(
            "base_llm_client_request error: Failed to get a response after {} retries,\nerrors:\n{:?}",
            self.config.retry_after_fail_n_times, errors
        );

        tracing::info!(error);
        eprintln!("{error}");
        Err(LlmClientResponseError::InferenceError { error })
    }

    async fn request(&self) -> Result<LlmClientResponse, LlmClientResponseError> {
        let res = match self.config.backend.as_ref() {
            LlmBackend::Llama(b) => b.llm_request(self).await,
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(b) => b.llm_request(self).await,
            LlmBackend::OpenAi(b) => b.llm_request(self).await,
            LlmBackend::Perplexity(b) => b.llm_request(self).await,
            LlmBackend::Anthropic(b) => b.llm_request(self).await,
        };
        match res {
            Err(e) => Err(e),
            Ok(mut res) => {
                if self.stop_sequences.required {
                    if matches!(
                        res.stop_reason,
                        LlmClientResponseStopReason::StoppingSequence(_)
                    ) {
                        Ok(res)
                    } else {
                        res.error = Some(LlmClientResponseError::StopSequenceError {
                            error: self.stop_sequences.error_on_required(),
                        });
                        Ok(res)
                    }
                } else if res.stop_reason == LlmClientResponseStopReason::StopLimit {
                    if self.config.increase_limit_on_stopped_limit {
                        res.error = Some(LlmClientResponseError::StopLimitError {
                            error: "res.stopped_limit == true &&  self.retry_stopped_limit == true"
                                .to_string(),
                        });
                    }
                    Ok(res)
                } else if res.stop_reason == LlmClientResponseStopReason::Eos {
                    Ok(res)
                } else {
                    res.error = Some(LlmClientResponseError::UnknownStopReason {
                        error: "Unknown stop reason".to_string(),
                    });
                    Ok(res)
                }
            }
        }
    }

    fn increase_limit(&mut self) -> Result<()> {
        let mut log =
            "res.stopped_limit: true. Attempting to increase requested_response_tokens to retry. "
                .to_string();
        let old_actual_request_tokens = self.config.actual_request_tokens;
        let old_requested_response_tokens = self.config.requested_response_tokens;
        self.config.requested_response_tokens =
            Some((self.config.requested_response_tokens.unwrap() as f32 * 1.33) as u32);

        self.set_max_tokens_for_request()?;
        let new_actual_request_tokens = self.config.actual_request_tokens;
        let new_requested_response_tokens = self.config.requested_response_tokens;

        log.push_str(&format!(
            "actual_request_tokens increased from {:?} to {:?} by increasing requested_response_tokens from {:?} to {:?}. ", old_actual_request_tokens, old_actual_request_tokens, old_requested_response_tokens, new_requested_response_tokens
        ));
        if new_actual_request_tokens <= old_actual_request_tokens {
            log.push_str("increase_limit failed.");
            tracing::info!(log);
            Err(anyhow!(log))
        } else {
            log.push_str("increase_limit success. Retrying.");
            tracing::info!(log);
            Ok(())
        }
    }

    async fn build_logit_bias(&mut self) -> Result<()> {
        if let Some(logit_bias) = &mut self.logit_bias {
            match self.config.backend.as_ref() {
                LlmBackend::Llama(_) => logit_bias.build_llama()?,
                #[cfg(feature = "mistralrs_backend")]
                LlmBackend::MistralRs(_) => logit_bias.build_llama()?,
                LlmBackend::OpenAi(_) => logit_bias.build_openai()?,
                LlmBackend::Anthropic(_) => unreachable!(),
                LlmBackend::Perplexity(_) => unreachable!(),
            };
        }
        Ok(())
    }

    fn set_max_tokens_for_request(&mut self) -> Result<(), RequestTokenLimitError> {
        let total_prompt_tokens = match self.instruct_prompt.prompt.total_prompt_tokens() {
            Ok(tokens) => tokens,
            Err(e) => return Err(RequestTokenLimitError::GenericPromptError { e: e.to_string() }),
        };

        let actual_request_tokens = check_and_get_max_tokens(
            self.config.ctx_size,
            Some(self.config.ctx_output_size),
            total_prompt_tokens,
            Some(self.config.safety_tokens),
            self.config.requested_response_tokens,
        )?;
        self.config.actual_request_tokens = Some(actual_request_tokens);
        if self.config.requested_response_tokens.is_none() {
            self.config.requested_response_tokens = Some(actual_request_tokens);
        }
        Ok(())
    }
}
