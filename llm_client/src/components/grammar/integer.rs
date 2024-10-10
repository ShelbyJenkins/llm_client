use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::cell::RefCell;

#[derive(Clone, Default)]
pub struct IntegerGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    pub lower_bound: u32,
    pub upper_bound: u32,
    grammar_string: RefCell<Option<String>>,
}

impl IntegerGrammar {
    pub fn new() -> Self {
        Self {
            stop_word_done: None,
            stop_word_no_result: None,
            lower_bound: 1,
            upper_bound: 9,
            grammar_string: RefCell::new(None),
        }
    }

    pub fn wrap(self) -> Grammar {
        Grammar::Integer(self)
    }

    pub fn lower_bound(mut self, lower_bound: u32) -> Self {
        self.lower_bound = lower_bound;

        self
    }

    pub fn upper_bound(mut self, upper_bound: u32) -> Self {
        self.upper_bound = upper_bound;

        self
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(integer_grammar(
                self.lower_bound,
                self.upper_bound,
                &self.stop_word_done,
                &self.stop_word_no_result,
            ));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        integer_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<u32, GrammarError> {
        integer_parse(content)
    }
}

impl GrammarSetterTrait for IntegerGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn integer_grammar<T: AsRef<str>>(
    lower_bound: u32,
    upper_bound: u32,
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    match upper_bound.cmp(&lower_bound) {
        std::cmp::Ordering::Less => {
            panic!("Upper bound must be greater than or equal to lower bound.")
        }
        std::cmp::Ordering::Equal => panic!("Bounds must not be the same."),
        _ => (),
    }
    // let mut base = "root ::= \" \" ".to_string();
    let range = create_range(lower_bound, upper_bound, stop_word_done);
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= \" \" ( {range} | \"{}\" ) \" {}\"",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= \" \" ( {range} | \"{}\" )",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!("root ::= \" \" {range} \" {}\"", stop_word_done.as_ref())
        }
        (None, None) => format!("root ::= \" \" {range}"),
    }
}

fn create_range<T: AsRef<str>>(
    lower_bound: u32,
    upper_bound: u32,
    stop_word_done: &Option<T>,
) -> String {
    let digits = (upper_bound as f64).log10().floor() as u32 + 1;
    let mut range = String::new();
    if digits == 1 {
        range.push_str(&format!("[{}-{}]", lower_bound, upper_bound));
        return range;
    }

    // Need to add the actual math here to restrict the range.
    for i in 1..=digits {
        if i > 1 && 10_u32.pow(i - 1) > lower_bound {
            if let Some(stop_word_done) = stop_word_done {
                range.push_str(&format!("([0-9] | \" {}\")", stop_word_done.as_ref()));
            } else {
                range.push_str("[0-9]?");
            }
        } else {
            range.push_str("[0-9]");
        }
    }
    range
}

pub fn integer_validate_clean(content: &str) -> Result<String, GrammarError> {
    let content: &str = content.trim();
    if integer_parse(content).is_ok() {
        Ok(content.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "u32".to_string(),
        })
    }
}

pub fn integer_parse(content: &str) -> Result<u32, GrammarError> {
    content
        .trim()
        .parse::<u32>()
        .map_err(|_| GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "u32".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut grammar = Grammar::integer().lower_bound(1).upper_bound(9);
        let grammar_string = grammar.set_stop_word_done("stop").grammar_string();

        assert_eq!(grammar_string, "root ::= \" \" [1-9] \" stop\"");
        assert_eq!(5, grammar.grammar_parse(" 5 ").unwrap());

        let mut grammar = Grammar::integer().lower_bound(0).upper_bound(99);
        let grammar_string = grammar.set_stop_word_done("stop").grammar_string();

        assert_eq!(
            grammar_string,
            "root ::= \" \" [0-9]([0-9] | \" stop\") \" stop\""
        );
        assert_eq!(55, grammar.grammar_parse(" 55 ").unwrap());

        let mut grammar = Grammar::integer().lower_bound(99).upper_bound(999);
        let grammar_string = grammar.set_stop_word_done("stop").grammar_string();

        assert_eq!(
            grammar_string,
            "root ::= \" \" [0-9][0-9]([0-9] | \" stop\") \" stop\""
        );
        assert_eq!(555, grammar.grammar_parse(" 555 ").unwrap());

        let mut grammar = Grammar::integer().lower_bound(0).upper_bound(10000);

        let grammar_string = grammar
            .set_stop_word_done("stop")
            .set_stop_word_no_result("unknown")
            .grammar_string();

        assert_eq!(
            grammar_string,
            "root ::= \" \" ( [0-9]([0-9] | \" stop\")([0-9] | \" stop\")([0-9] | \" stop\")([0-9] | \" stop\") | \"unknown\" ) \" stop\""
        );
        assert_eq!(5555, grammar.grammar_parse(" 5555 ").unwrap());
    }
}
