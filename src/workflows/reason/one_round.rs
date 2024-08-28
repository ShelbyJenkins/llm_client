use super::{
    decision::DecisionTrait,
    PrimitiveTrait,
    ReasonResult,
    ReasonTrait,
    SentencesPrimitive,
};
use crate::{
    components::{
        base_request::{BaseLlmRequest, BaseRequestConfig},
        cascade::{step::StepConfig, CascadeFlow},
        instruct_prompt::InstructPrompt,
    },
    BaseRequestConfigTrait,
    InstructPromptTrait,
};
use anyhow::Result;

pub struct ReasonOneRound<P> {
    pub reasoning_sentences: u8,
    pub conclusion_sentences: u8,
    pub result_can_be_none: bool,
    pub primitive: P,
    pub base_req: BaseLlmRequest,
}

impl<P: PrimitiveTrait + ReasonTrait> ReasonOneRound<P> {
    pub async fn return_primitive(&mut self) -> Result<P::PrimitiveResult> {
        let res = self.return_result().await?;
        if let Some(primitive_result) =
            self.primitive.result_index_to_primitive(res.result_index)?
        {
            Ok(primitive_result)
        } else {
            Err(anyhow::format_err!("No result returned."))
        }
    }

    pub async fn return_optional_primitive(&mut self) -> Result<Option<P::PrimitiveResult>> {
        let res = self.return_optional_result().await?;
        self.primitive.result_index_to_primitive(res.result_index)
    }

    pub async fn return_result(&mut self) -> Result<ReasonResult> {
        self.result_can_be_none = false;
        let mut flow: CascadeFlow = self.reason_one_round()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        ReasonResult::new(flow, &self.primitive, &self.base_req)
    }

    pub async fn return_optional_result(&mut self) -> Result<ReasonResult> {
        self.result_can_be_none = true;
        let mut flow: CascadeFlow = self.reason_one_round()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        ReasonResult::new(flow, &self.primitive, &self.base_req)
    }

    pub fn reasoning_sentences(&mut self, reasoning_sentences: u8) -> &mut Self {
        self.reasoning_sentences = reasoning_sentences;
        self
    }

    pub fn conclusion_sentences(&mut self, conclusion_sentences: u8) -> &mut Self {
        self.conclusion_sentences = conclusion_sentences;
        self
    }

    fn reason_one_round(&mut self) -> Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("Reason One Round");

        flow.new_round(
        "A request will be provided. Think out loud about the request. State the arguments before arriving at a conclusion with, 'Therefore, we can conclude:...', and finish with a solution by saying, 'Thus, the solution...'. With no yapping.").add_guidance_step(
        &StepConfig {
            ..StepConfig::default()
        },
        "'no yapping' refers to a design principle or behavior where the AI model provides direct, concise responses without unnecessary verbosity or filler content. Therefore, we can conclude: The user would like to get straight to the point. Thus, the solution is to to resolve the request as efficiently as possible.",
    );

        let task = self.build_task()?;

        // CoT reasoning
        let step_config = StepConfig {
            step_prefix: Some("Thinking out loud about the users request...".to_string()),
            stop_word_done: "Therefore, we can conclude".to_string(),
            cache_prompt: false, // Clears the cache on the initial request
            grammar: SentencesPrimitive::default()
                .min_count(1)
                .max_count(self.reasoning_sentences)
                .grammar(),
            ..StepConfig::default()
        };

        flow.new_round(task).add_inference_step(&step_config);

        // Conclusion
        let step_config = StepConfig {
            step_prefix: Some(format!(
                "The user requested a conclusion of {}. Therefore, we can conclude:",
                self.primitive.solution_description(self.result_can_be_none),
            )),
            stop_word_done: "Thus, the solution".to_string(),
            grammar: SentencesPrimitive::default()
                .min_count(1)
                .max_count(self.conclusion_sentences)
                .grammar(),

            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);

        // Instructions Restatement
        if let Some(instructions) = self.base_req.instruct_prompt.build_instructions() {
            let instructions_restatement =
                format!("The user's original request was '{}'.", &instructions,);
            let step_config = StepConfig {
                step_prefix: None,
                grammar: SentencesPrimitive::default().grammar(),
                ..StepConfig::default()
            };
            flow.last_round()?
                .add_guidance_step(&step_config, instructions_restatement);
        };

        // Solution
        let solution = format!(
            "Thus, the {} solution to the user's request is:",
            self.primitive.type_description(self.result_can_be_none),
        );
        let step_config = StepConfig {
            step_prefix: Some(solution),
            stop_word_null_result: self
                .primitive
                .stop_word_result_is_none(self.result_can_be_none),
            grammar: self.primitive.grammar(),
            ..StepConfig::default()
        };
        flow.last_round()?.add_inference_step(&step_config);

        Ok(flow)
    }

    fn build_task(&mut self) -> Result<String> {
        let instructions = self.base_req.instruct_prompt.build_instructions();
        let supporting_material = self.base_req.instruct_prompt.build_supporting_material();

        Ok(match (instructions, supporting_material) {
            (Some(instructions), Some(supporting_material)) => {
                format!("The user provided some supporting material: {supporting_material}\n The user's request is: {instructions}", )
            }
            (Some(instructions), None) => {
                format!("The user's request is: {instructions}",)
            }
            (None, Some(supporting_material)) => {
                format!("The user's request is: {supporting_material}",)
            }
            (None, None) => {
                return Err(anyhow::format_err!(
                    "No instructions or supporting material provided."
                ));
            }
        })
    }
}

impl<P: PrimitiveTrait> BaseRequestConfigTrait for ReasonOneRound<P> {
    fn base_config(&mut self) -> &mut BaseRequestConfig {
        &mut self.base_req.config
    }

    fn clear_request(&mut self) {
        self.base_req.reset_base_request();
    }
}

impl<P: PrimitiveTrait + ReasonTrait> InstructPromptTrait for ReasonOneRound<P> {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.base_req.instruct_prompt
    }
}

impl<P: PrimitiveTrait + ReasonTrait> DecisionTrait for ReasonOneRound<P> {
    type ReasonPrimitive = P;
    fn base_req(&self) -> &BaseLlmRequest {
        &self.base_req
    }

    fn base_req_mut(&mut self) -> &mut BaseLlmRequest {
        &mut self.base_req
    }

    fn primitive(&self) -> &Self::ReasonPrimitive {
        &self.primitive
    }

    async fn return_reason_result(&mut self, result_can_be_none: bool) -> Result<ReasonResult> {
        if result_can_be_none {
            self.return_optional_result().await
        } else {
            self.return_result().await
        }
    }
}
