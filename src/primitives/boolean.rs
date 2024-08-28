use super::PrimitiveTrait;
use crate::workflows::reason::ReasonTrait;
use anyhow::Result;
use llm_utils::grammar::{BooleanGrammar, Grammar};

#[derive(Default)]
pub struct BooleanPrimitive {}

impl BooleanPrimitive {
    fn grammar_inner(&self) -> BooleanGrammar {
        Grammar::boolean()
    }
}

impl PrimitiveTrait for BooleanPrimitive {
    type PrimitiveResult = bool;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "boolean or 'Neither.'"
        } else {
            "boolean"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            "a boolean or 'neither'; If the answer is true/yes/affirmative, then it's 'true'. If the answer is false/no/negative, then it's 'false'. If it's neither, then it's 'Neither.'".to_owned()
        } else {
            "a boolean; If the answer is true/yes/affirmative, then it's 'true'. If the answer is false/no/negative, then it's 'false'.".to_owned()
        }
    }

    fn stop_word_result_is_none(&self, result_can_be_none: bool) -> Option<String> {
        if result_can_be_none {
            Some("Neither.".to_string())
        } else {
            None
        }
    }

    fn grammar(&self) -> Grammar {
        self.grammar_inner().wrap()
    }

    fn parse_to_primitive(&self, content: &str) -> Result<Self::PrimitiveResult> {
        let parsed: Self::PrimitiveResult = self.grammar_inner().grammar_parse(content)?;
        Ok(parsed)
    }
}

impl ReasonTrait for BooleanPrimitive {
    fn primitive_to_result_index(&self, content: &str) -> u32 {
        let output = self.parse_to_primitive(content).unwrap();
        if output {
            1
        } else {
            0
        }
    }

    fn result_index_to_primitive(&self, result_index: Option<u32>) -> Result<Option<bool>> {
        if let Some(result_index) = result_index {
            Ok(match result_index {
                0 => Some(false),
                1 => Some(true),
                _ => return Err(anyhow::format_err!("Decision: no winner")),
            })
        } else {
            Ok(None)
        }
    }
}
