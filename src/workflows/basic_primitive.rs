use crate::{
    components::{
        base_request::{BaseLlmRequest, BaseRequestConfig},
        cascade::{step::StepConfig, CascadeFlow},
        instruct_prompt::InstructPrompt,
    },
    llm_backends::LlmBackend,
    primitives::*,
    BaseRequestConfigTrait,
    InstructPromptTrait,
};
use anyhow::Result;
use std::rc::Rc;

pub struct BasicPrimitiveWorkflow<P> {
    pub primitive: P,
    pub base_req: BaseLlmRequest,
    pub result_can_be_none: bool,
}

impl<P: PrimitiveTrait> BasicPrimitiveWorkflow<P> {
    pub fn new(backend: &Rc<LlmBackend>) -> Self {
        Self {
            primitive: P::default(),
            base_req: BaseLlmRequest::new_from_backend(backend),
            result_can_be_none: false,
        }
    }

    pub async fn return_primitive(&mut self) -> Result<P::PrimitiveResult> {
        self.result_can_be_none = false;
        let res = self.return_result().await?;
        if let Some(primitive_result) = res.primitive_result {
            Ok(self.primitive.parse_to_primitive(&primitive_result)?)
        } else {
            Err(anyhow::format_err!("No result returned."))
        }
    }

    pub async fn return_optional_primitive(&mut self) -> Result<Option<P::PrimitiveResult>> {
        self.result_can_be_none = true;
        let res = self.return_result().await?;
        if let Some(primitive_result) = res.primitive_result {
            Ok(Some(self.primitive.parse_to_primitive(&primitive_result)?))
        } else {
            Ok(None)
        }
    }

    pub async fn return_result(&mut self) -> Result<BasicPrimitiveResult> {
        self.result_can_be_none = false;
        let mut flow = self.basic_primitive()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        BasicPrimitiveResult::new(flow)
    }

    pub async fn return_optional_result(&mut self) -> Result<BasicPrimitiveResult> {
        self.result_can_be_none = true;
        let mut flow = self.basic_primitive()?;
        flow.run_all_rounds(&mut self.base_req).await?;
        BasicPrimitiveResult::new(flow)
    }

    fn basic_primitive(&mut self) -> Result<CascadeFlow> {
        let mut flow = CascadeFlow::new("BasicPrimitive");
        let task = self.base_req.instruct_prompt.build_instruct_prompt(false)?;

        let step_config = StepConfig {
            step_prefix: Some(format!(
                "generating {}:",
                self.primitive.solution_description(self.result_can_be_none),
            )),
            stop_word_null_result: self
                .primitive
                .stop_word_result_is_none(self.result_can_be_none),
            grammar: self.primitive.grammar(),
            ..StepConfig::default()
        };

        flow.new_round(task).add_inference_step(&step_config);

        Ok(flow)
    }
}

impl<P: PrimitiveTrait> BaseRequestConfigTrait for BasicPrimitiveWorkflow<P> {
    fn base_config(&mut self) -> &mut BaseRequestConfig {
        &mut self.base_req.config
    }

    fn clear_request(&mut self) {
        self.base_req.reset_base_request();
    }
}

impl<P: PrimitiveTrait> InstructPromptTrait for BasicPrimitiveWorkflow<P> {
    fn instruct_prompt_mut(&mut self) -> &mut InstructPrompt {
        &mut self.base_req.instruct_prompt
    }
}

pub struct BasicPrimitiveWorkflowBuilder {
    pub base_req: BaseLlmRequest,
}

impl BasicPrimitiveWorkflowBuilder {
    pub fn new(backend: &Rc<LlmBackend>) -> Self {
        Self {
            base_req: BaseLlmRequest::new_from_backend(backend),
        }
    }

    fn build<P: PrimitiveTrait>(self) -> BasicPrimitiveWorkflow<P> {
        BasicPrimitiveWorkflow {
            primitive: P::default(),
            base_req: self.base_req,
            result_can_be_none: false,
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
    pub fn new(flow: CascadeFlow) -> Result<Self> {
        let reason_result = BasicPrimitiveResult {
            primitive_result: flow.primitive_result(),
            duration: flow.duration,
            workflow: flow,
        };
        Ok(reason_result)
    }
}
