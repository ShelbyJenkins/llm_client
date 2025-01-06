use super::PrimitiveTrait;
use crate::components::grammar::{Grammar, IntegerGrammar};
use crate::workflows::reason::ReasonTrait;
use anyhow::Result;

pub struct IntegerPrimitive {
    pub lower_bound: u32,
    pub upper_bound: u32,
}

impl Default for IntegerPrimitive {
    fn default() -> Self {
        IntegerPrimitive {
            lower_bound: 0,
            upper_bound: 9999,
        }
    }
}

impl IntegerPrimitive {
    /// Set the lower bound of the integer range. Default is 0.
    pub fn lower_bound(&mut self, lower_bound: u32) -> &mut Self {
        if self.lower_bound != lower_bound {
            self.lower_bound = lower_bound;
        }
        self
    }

    /// Set the upper bound of the integer range. Default is 9.
    pub fn upper_bound(&mut self, upper_bound: u32) -> &mut Self {
        if self.upper_bound != upper_bound {
            self.upper_bound = upper_bound;
        }
        self
    }

    fn grammar_inner(&self) -> IntegerGrammar {
        Grammar::integer()
            .lower_bound(self.lower_bound)
            .upper_bound(self.upper_bound)
    }
}

impl PrimitiveTrait for IntegerPrimitive {
    type PrimitiveResult = u32;

    fn clear_primitive(&mut self) {}

    fn type_description(&self, result_can_be_none: bool) -> &str {
        if result_can_be_none {
            "number or 'Unknown.'"
        } else {
            "number"
        }
    }

    fn solution_description(&self, result_can_be_none: bool) -> String {
        if result_can_be_none {
            format!(
                "a number between {}-{} or, if the solution is unknown or not in range, 'Unknown.'",
                self.lower_bound, self.upper_bound
            )
        } else {
            format!("a number between {}-{}", self.lower_bound, self.upper_bound)
        }
    }

    fn stop_word_result_is_none(&self, result_can_be_none: bool) -> Option<String> {
        if result_can_be_none {
            Some("Unknown.".to_string())
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

impl ReasonTrait for IntegerPrimitive {
    fn primitive_to_result_index(&self, content: &str) -> u32 {
        self.parse_to_primitive(content).unwrap()
    }

    fn result_index_to_primitive(&self, result_index: Option<u32>) -> Result<Option<u32>> {
        if result_index.is_some() {
            Ok(result_index)
        } else {
            Ok(None)
        }
    }
}
