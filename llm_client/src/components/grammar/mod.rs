use thiserror::Error;
pub mod basic_url;
pub mod boolean;
pub mod exact_string;
pub mod faux_url;
pub mod integer;
pub mod sentences;
pub mod text;
pub mod words;

pub use basic_url::BasicUrlGrammar;
pub use boolean::BooleanGrammar;
pub use exact_string::ExactStringGrammar;
pub use faux_url::FauxUrlGrammar;
pub use integer::IntegerGrammar;
pub use sentences::SentencesGrammar;
pub use text::TextGrammar;
pub use words::WordsGrammar;

#[derive(Clone)]
pub enum Grammar {
    Boolean(BooleanGrammar),
    Integer(IntegerGrammar),
    Text(TextGrammar),
    Sentences(SentencesGrammar),
    Words(WordsGrammar),
    BasicUrl(BasicUrlGrammar),
    ExactString(ExactStringGrammar),
    FauxUrl(FauxUrlGrammar),
}

macro_rules! grammar_default {
    ($enum_name:ident {
        $($variant:ident => $fn_name:ident: $inner_type:ident),* $(,)?
    }) => {
        impl $enum_name {
            $(
                pub fn $fn_name() -> $inner_type {
                    $inner_type::default()
                }
            )*

            pub fn grammar_string(&self) -> String {
                match self {
                    $(
                        $enum_name::$variant(grammar) => grammar.grammar_string(),
                    )*
                }
            }

            pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
                match self {
                    $(
                        $enum_name::$variant(grammar) => grammar.validate_clean(content),
                    )*
                }
            }

            pub fn set_stop_word_done<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self {
                match self {
                    $(
                        $enum_name::$variant(grammar) => {
                           if grammar.stop_word_done.as_deref() != Some(stop_word.as_ref()) {
                               grammar.stop_word_done = Some(stop_word.as_ref().to_owned());
                            }
                            self
                        }
                    )*
                }
            }

            pub fn set_stop_word_no_result<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self {
                match self {
                    $(
                        $enum_name::$variant(grammar) => {
                           if grammar.stop_word_no_result.as_deref() != Some(stop_word.as_ref()) {
                               grammar.stop_word_no_result = Some(stop_word.as_ref().to_owned());
                            }
                            self
                        }
                    )*
                }
            }
        }
    };
}

grammar_default! {
    Grammar {
        Boolean => boolean: BooleanGrammar,
        Integer => integer: IntegerGrammar,
        Text => text: TextGrammar,
        Sentences => sentences: SentencesGrammar,
        Words => words: WordsGrammar,
        BasicUrl => basic_url: BasicUrlGrammar,
        ExactString => exact_string: ExactStringGrammar,
        FauxUrl => faux_url: FauxUrlGrammar,
    }
}

impl Default for Grammar {
    fn default() -> Self {
        Grammar::Text(TextGrammar::default())
    }
}

pub trait GrammarSetterTrait {
    fn stop_word_done_mut(&mut self) -> &mut Option<String>;

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String>;

    fn set_stop_word_done<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self
    where
        Self: Sized,
    {
        if self.stop_word_done_mut().as_deref() != Some(stop_word.as_ref()) {
            *self.stop_word_done_mut() = Some(stop_word.as_ref().to_owned());
        }
        self
    }

    fn set_stop_word_no_result<T: AsRef<str>>(&mut self, stop_word: T) -> &mut Self
    where
        Self: Sized,
    {
        if self.stop_word_no_result_mut().as_deref() != Some(stop_word.as_ref()) {
            *self.stop_word_no_result_mut() = Some(stop_word.as_ref().to_owned());
        }
        self
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum GrammarError {
    #[error("grammar not set")]
    GrammarNotSet,
    #[error("response ({response}) lacks the correct prefix for given grammar ({correct_prefix})")]
    PrefixIncorrect {
        correct_prefix: String,
        response: String,
    },
    #[error("failed to parse response_content ({content}) as type ({parse_type})")]
    ParseValueError { content: String, parse_type: String },
    #[error("incorrect destructuring function ({function}) for grammar type ({grammar_type})")]
    DestructuringIncorrect {
        function: String,
        grammar_type: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let grammar = Grammar::integer().lower_bound(0).upper_bound(100);

        assert!(grammar.grammar_string().contains('9'));
        let mut grammar = Grammar::boolean();
        assert!(grammar
            .set_stop_word_done("done")
            .set_stop_word_no_result("nah")
            .grammar_string()
            .contains("done"));

        let res: bool = grammar.grammar_parse("false").unwrap();
        assert!(!res);
        let res: bool = grammar.grammar_parse("true").unwrap();
        assert!(res);
    }
}
