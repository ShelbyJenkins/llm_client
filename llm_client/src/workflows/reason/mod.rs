pub mod decision;
pub mod one_round;

use super::*;
use crate::{
    components::{cascade::CascadeFlow, instruct_prompt::InstructPrompt},
    primitives::*,
};
use one_round::ReasonOneRound;

pub trait ReasonTrait: PrimitiveTrait {
    fn primitive_to_result_index(&self, content: &str) -> u32;

    fn result_index_to_primitive(
        &self,
        result_index: Option<u32>,
    ) -> crate::Result<Option<Self::PrimitiveResult>>;

    fn parse_reason_result(
        &self,
        reason_result: &ReasonResult,
    ) -> crate::Result<Option<Self::PrimitiveResult>> {
        if let Some(result_index) = reason_result.result_index {
            self.result_index_to_primitive(Some(result_index))
        } else {
            Ok(None)
        }
    }
}

pub struct ReasonWorkflowBuilder {
    pub base_req: CompletionRequest,
}

impl ReasonWorkflowBuilder {
    pub fn new(backend: std::sync::Arc<LlmBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    fn build<P: PrimitiveTrait>(self) -> ReasonOneRound<P> {
        ReasonOneRound {
            primitive: P::default(),
            base_req: self.base_req,
            reasoning_sentences: 3,
            conclusion_sentences: 2,
            result_can_be_none: false,
            instruct_prompt: InstructPrompt::default(),
        }
    }
}

macro_rules! reason_workflow_primitive_impl {
    ($($name:ident => $type:ty),*) => {
        impl ReasonWorkflowBuilder {
            $(
                pub fn $name(self) -> ReasonOneRound<$type> {
                    self.build()
                }
            )*
        }
    }
}

reason_workflow_primitive_impl! {
    boolean => BooleanPrimitive,
    integer => IntegerPrimitive,
    exact_string => ExactStringPrimitive
}

#[derive(Clone)]
pub struct ReasonResult {
    pub primitive_result: Option<String>,
    pub duration: std::time::Duration,
    pub workflow: CascadeFlow,
    pub result_index: Option<u32>,
    pub temperature: f32,
}

impl ReasonResult {
    fn new<P: PrimitiveTrait + ReasonTrait>(
        flow: CascadeFlow,
        primitive: &P,
        base_req: &CompletionRequest,
    ) -> crate::Result<Self> {
        let primitive_result = flow.primitive_result();
        let result_index = primitive_result
            .as_ref()
            .map(|primitive_result| primitive.primitive_to_result_index(primitive_result));
        Ok(ReasonResult {
            primitive_result,
            duration: flow.duration,
            workflow: flow,
            result_index,
            temperature: base_req.config.temperature,
        })
    }
}

impl std::fmt::Display for ReasonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.workflow)?;
        writeln!(
            f,
            "{}Reason duration\x1b[0m: {:?}",
            SETTINGS_GRADIENT[2], self.duration
        )?;
        writeln!(
            f,
            "{}Reason temperature\x1b[0m: {:?}",
            SETTINGS_GRADIENT[1], self.temperature
        )?;
        if let Some(primitive_result) = &self.primitive_result {
            writeln!(
                f,
                "{}Reason primitive_result\x1b[0m: {}",
                SETTINGS_GRADIENT[0], primitive_result
            )?;
        } else {
            writeln!(
                f,
                "{}Reason primitive_result\x1b[0m: None",
                SETTINGS_GRADIENT[0],
            )?;
        };
        Ok(())
    }
}

static SETTINGS_GRADIENT: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    vec![
        "\x1B[38;2;92;244;37m",
        "\x1B[38;2;0;239;98m",
        "\x1B[38;2;0;225;149m",
        "\x1B[38;2;0;212;178m",
        "\x1B[38;2;0;201;196m",
        "\x1B[38;2;0;190;207m",
        "\x1B[38;2;0;180;215m",
        "\x1B[38;2;0;170;222m",
        "\x1B[38;2;0;159;235m",
        "\x1B[38;2;0;142;250m",
    ]
});
