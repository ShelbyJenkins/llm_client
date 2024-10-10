use super::cascade_request;
use crate::components::grammar::Grammar;
use llm_interface::requests::completion::CompletionRequest;

#[derive(Clone)]
pub enum CascadeStep {
    Inference(InferenceStep),
    Guidance(GuidanceStep),
}

impl CascadeStep {
    pub fn new_inference_step(step_config: StepConfig, step_counter: usize) -> Self {
        CascadeStep::Inference(InferenceStep {
            llm_content: None,
            dynamic_suffix: None,
            outcome: std::cell::RefCell::new(None),
            step_config,
            step_counter,
        })
    }

    pub fn new_guidance_step<S: Into<String>>(
        step_config: StepConfig,
        step_counter: usize,
        llm_content: S,
    ) -> Self {
        CascadeStep::Guidance(GuidanceStep {
            llm_content: llm_content.into(),
            step_counter,
            step_config,
        })
    }

    pub fn display_step_prefix(&self) -> Option<String> {
        match self {
            Self::Inference(step) => step.step_config.display_prefix(step.step_counter),
            Self::Guidance(step) => step.step_config.display_prefix(step.step_counter),
        }
    }

    pub async fn run_step(
        &mut self,
        generation_prefix: Option<&str>,
        base_req: &mut CompletionRequest,
    ) -> crate::Result<()> {
        match self {
            Self::Inference(step) => step.run(generation_prefix, base_req).await,
            Self::Guidance(_) => self.set_cache_up_to_step(generation_prefix, base_req).await,
        }
    }

    pub async fn set_cache_up_to_step(
        &mut self,
        generation_prefix: Option<&str>,
        base_req: &mut CompletionRequest,
    ) -> crate::Result<()> {
        if let Some(generation_prefix) = generation_prefix {
            base_req.prompt.set_generation_prefix(generation_prefix);
        }
        base_req
            .backend
            .set_cache(&base_req.prompt)
            .await
            .map_err(|e| crate::anyhow!("Failed to set cache up to step: {}", e))?;
        Ok(())
    }

    pub fn set_dynamic_suffix<S: Into<String>>(&mut self, dynamic_suffix: S) {
        match self {
            Self::Inference(step) => step.dynamic_suffix = Some(dynamic_suffix.into()),
            Self::Guidance(_) => panic!("GuidanceStep does not have dynamic_suffix."),
        }
    }

    pub fn display_step_outcome(&self) -> crate::Result<String> {
        match self {
            Self::Inference(step) => step.display_outcome(),
            Self::Guidance(step) => Ok(step.display_outcome()),
        }
    }

    pub fn primitive_result(&self) -> Option<String> {
        match self {
            Self::Inference(step) => step.llm_content.clone(),
            Self::Guidance(_) => panic!("GuidanceStep does not have primitive_result."),
        }
    }
}

#[derive(Clone)]
pub struct InferenceStep {
    pub llm_content: Option<String>, // raw, unformatted result from llm.
    pub dynamic_suffix: Option<String>, // suffix to be added to the result.
    pub outcome: std::cell::RefCell<Option<String>>,
    pub step_config: StepConfig,
    pub step_counter: usize,
}

impl InferenceStep {
    async fn run(
        &mut self,
        generation_prefix: Option<&str>,
        base_req: &mut CompletionRequest,
    ) -> crate::Result<()> {
        // Request tokens
        base_req.config.requested_response_tokens = None;
        // Request stop words
        base_req.stop_sequences.required = true;
        base_req.set_base_req_stop_sequences(
            &Some(self.step_config.stop_word_done.clone()),
            &self.step_config.stop_word_no_result,
        );
        if let Some(stop_word_no_result) = &self.step_config.stop_word_no_result {
            self.step_config
                .grammar
                .set_stop_word_no_result(stop_word_no_result);
        }
        // Request grammar
        self.step_config
            .grammar
            .set_stop_word_done(&self.step_config.stop_word_done);
        base_req.grammar_string = Some(self.step_config.grammar.grammar_string());

        // Request prompt
        if let Some(generation_prefix) = generation_prefix {
            base_req.prompt.set_generation_prefix(generation_prefix);
        } else {
            base_req.prompt.clear_generation_prefix();
        }
        base_req.config.cache_prompt = self.step_config.cache_prompt;
        cascade_request(base_req, self).await
    }

    // step_counter + step_prefix + prefix_delimiter + (llm_content | stop_word_no_result) + dynamic_suffix
    fn display_outcome(&self) -> crate::Result<String> {
        let llm_content = if let Some(llm_content) = &self.llm_content {
            llm_content
        } else if let Some(stop_word_no_result) = &self.step_config.stop_word_no_result {
            stop_word_no_result
        } else {
            crate::bail!("llm_content not yet set and stop_word_no_result not set.")
        };

        Ok(
            match (
                self.step_config.display_prefix(self.step_counter),
                &self.dynamic_suffix,
            ) {
                (Some(step_prefix), Some(dynamic_suffix)) => {
                    format!("{}{}{}", step_prefix, llm_content, dynamic_suffix)
                }
                (Some(step_prefix), None) => format!("{}{}", step_prefix, llm_content),
                (None, Some(dynamic_suffix)) => {
                    format!("{}{}", llm_content, dynamic_suffix)
                }
                (None, None) => llm_content.to_owned(),
            },
        )
    }
}

#[derive(Clone)]
pub struct GuidanceStep {
    pub llm_content: String,
    pub step_config: StepConfig,
    pub step_counter: usize,
}

impl GuidanceStep {
    fn display_outcome(&self) -> String {
        match self.step_config.display_prefix(self.step_counter) {
            Some(step_prefix) => format!("{}{}", step_prefix, self.llm_content),
            None => self.llm_content.to_owned(),
        }
    }
}

#[derive(Clone)]
pub struct StepConfig {
    pub step_prefix: Option<String>,
    pub stop_word_done: String,
    pub stop_word_no_result: Option<String>,
    pub use_counter: bool,
    pub cache_prompt: bool,
    pub grammar: Grammar,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            step_prefix: None,
            stop_word_done: "Done.".to_owned(),
            stop_word_no_result: None,
            use_counter: false,
            cache_prompt: true,
            grammar: Grammar::default(),
        }
    }
}

impl StepConfig {
    fn display_prefix(&self, step_counter: usize) -> Option<String> {
        match (self.use_counter, &self.step_prefix) {
            (true, Some(step_prefix)) => Some(format!("{} {}", step_counter, step_prefix)),
            (true, None) => Some(step_counter.to_string()),
            (false, Some(step_prefix)) => Some(step_prefix.to_string()),
            (false, None) => None,
        }
    }
}
