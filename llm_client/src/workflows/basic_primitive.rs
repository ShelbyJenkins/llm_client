use crate::{
    components::{
        cascade::{step::StepConfig, CascadeFlow},
        instruct_prompt::InstructPrompt,
        InstructPromptTrait,
    },
    primitives::*,
};
use llm_interface::{
    llms::LlmBackend,
    requests::{
        completion::CompletionRequest,
        req_components::{RequestConfig, RequestConfigTrait},
    },
};

pub struct BasicPrimitiveWorkflow<P> {
    pub primitive: P,
    pub base_req: CompletionRequest,
    pub result_can_be_none: bool,
    pub instruct_prompt: InstructPrompt,
}

impl<P: PrimitiveTrait> BasicPrimitiveWorkflow<P> {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            primitive: P::default(),
            base_req: CompletionRequest::new(backend),
            result_can_be_none: false,
            instruct_prompt: InstructPrompt::default(),
        }
    }

    pub async fn return_primitive(&mut self) -> crate::Result<P::PrimitiveResult> {
        self.result_can_be_none = false;
        let res = self.return_result().await?;
        if let Some(primitive_result) = res.primitive_result {
            Ok(self.primitive.parse_to_primitive(&primitive_result)?)
        } else {
            Err(anyhow::format_err!("No result returned."))
        }
    }

    pub async fn return_optional_primitive(&mut self) -> crate::Result<Option<P::PrimitiveResult>> {
        self.result_can_be_none = true;
        let res = self.return_result().await?;
        if let Some(primitive_result) = res.primitive_result {
            Ok(Some(self.primitive.parse_to_primitive(&primitive_result)?))
        } else {
            Ok(None)
        }
    }

    pub async fn return_result(&mut self) -> crate::Result<BasicPrimitiveResult> {
        self.result_can_be_none = false;
        let mut flow = self.basic_primitive()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        BasicPrimitiveResult::new(flow)
    }

    pub async fn return_optional_result(&mut self) -> crate::Result<BasicPrimitiveResult> {
        self.result_can_be_none = true;
        let mut flow = self.basic_primitive()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        BasicPrimitiveResult::new(flow)
    }

    fn basic_primitive(&mut self) -> crate::Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("BasicPrimitive");
        let task = self.instruct_prompt.build_instruct_prompt(false)?;

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "generating {}:",
                self.primitive.solution_description(self.result_can_be_none),
            )),
            stop_word_no_result: self
                .primitive
                .stop_word_result_is_none(self.result_can_be_none),
            grammar: self.primitive.grammar(),
            ..StepConfig::default()
        };

        flow.new_round(task).add_inference_step(&step_config);

        Ok(flow)
    }
}

impl<P: PrimitiveTrait> RequestConfigTrait for BasicPrimitiveWorkflow<P> {
    fn config(&mut self) -> &mut RequestConfig {
        &mut self.base_req.config
    }

    fn reset_request(&mut self) {
        self.instruct_prompt.reset_instruct_prompt();
        self.base_req.reset_completion_request();
    }
}

impl<P: PrimitiveTrait> InstructPromptTrait for BasicPrimitiveWorkflow<P> {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.instruct_prompt
    }
}

pub struct BasicPrimitiveWorkflowBuilder {
    pub base_req: CompletionRequest,
}

impl BasicPrimitiveWorkflowBuilder {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    fn build<P: PrimitiveTrait>(self) -> BasicPrimitiveWorkflow<P> {
        BasicPrimitiveWorkflow {
            primitive: P::default(),
            base_req: self.base_req,
            result_can_be_none: false,
            instruct_prompt: InstructPrompt::default(),
        }
    }
}

macro_rules! basic_primitive_workflow_primitive_impl {
    ($($name:ident => $type:ty),*) => {
        impl BasicPrimitiveWorkflowBuilder {
            $(
                pub fn $name(self) -> BasicPrimitiveWorkflow<$type> {
                    self.build()
                }
            )*
        }
    }
}

basic_primitive_workflow_primitive_impl! {
    boolean => BooleanPrimitive,
    integer => IntegerPrimitive,
    sentences => SentencesPrimitive,
    words => WordsPrimitive,
    exact_string => ExactStringPrimitive
}

#[derive(Clone)]
pub struct BasicPrimitiveResult {
    pub primitive_result: Option<String>,
    pub duration: std::time::Duration,
    pub workflow: CascadeFlow,
}

impl BasicPrimitiveResult {
    pub fn new(flow: CascadeFlow) -> crate::Result<Self> {
        let reason_result = BasicPrimitiveResult {
            primitive_result: flow.primitive_result(),
            duration: flow.duration,
            workflow: flow,
        };
        Ok(reason_result)
    }
}
