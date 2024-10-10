use super::{Grammar, GrammarError, GrammarSetterTrait};
use std::{cell::RefCell, str::FromStr};
use url::Url;

#[derive(Clone, Default)]
pub struct BasicUrlGrammar {
    pub stop_word_done: Option<String>,
    pub stop_word_no_result: Option<String>,
    grammar_string: RefCell<Option<String>>,
}

impl BasicUrlGrammar {
    pub fn new() -> Self {
        Self {
            stop_word_done: None,
            stop_word_no_result: None,
            grammar_string: RefCell::new(None),
        }
    }
}

impl BasicUrlGrammar {
    pub fn wrap(self) -> Grammar {
        Grammar::BasicUrl(self)
    }

    pub fn grammar_string(&self) -> String {
        let mut grammar_string = self.grammar_string.borrow_mut();
        if grammar_string.is_none() {
            *grammar_string = Some(url_grammar(&self.stop_word_done, &self.stop_word_no_result));
        }
        grammar_string.as_ref().unwrap().clone()
    }

    pub fn validate_clean(&self, content: &str) -> Result<String, GrammarError> {
        url_validate_clean(content)
    }

    pub fn grammar_parse(&self, content: &str) -> Result<Url, GrammarError> {
        url_parse(content)
    }
}

impl GrammarSetterTrait for BasicUrlGrammar {
    fn stop_word_done_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_done
    }

    fn stop_word_no_result_mut(&mut self) -> &mut Option<String> {
        &mut self.stop_word_no_result
    }
}

pub fn url_grammar<T: AsRef<str>>(
    stop_word_done: &Option<T>,
    stop_word_no_result: &Option<T>,
) -> String {
    match (stop_word_done, stop_word_no_result) {
        (Some(stop_word_done), Some(stop_word_no_result)) => format!(
            "root ::= \" \" ( url | \"{}\" ) \" {}\"\n{URL_GRAMMAR}",
            stop_word_no_result.as_ref(),
            stop_word_done.as_ref()
        ),
        (None, Some(stop_word_no_result)) => {
            format!(
                "root ::= \" \" ( url | \"{}\" )\n{URL_GRAMMAR}",
                stop_word_no_result.as_ref()
            )
        }
        (Some(stop_word_done), None) => {
            format!(
                "root ::= \" \" url \" {}\"\n{URL_GRAMMAR}",
                stop_word_done.as_ref()
            )
        }
        (None, None) => format!("root ::= \" \" url\n{URL_GRAMMAR}"),
    }
}

pub const URL_GRAMMAR: &str = r##"url ::= scheme "://" authority path{0,}
scheme ::= "https" | "http"
authority ::= host (":" port){0,1}
host ::= domain | ipv4address
domain ::= subdomains tld
subdomains ::= (label "."){1,3}
label ::= [a-zA-Z0-9] [a-zA-Z0-9-]{0,62}
tld ::= [a-zA-Z]{2,63}
ipv4address ::= dec-octet "." dec-octet "." dec-octet "." dec-octet
dec-octet ::= [0-9] | [1-9][0-9] | "1"[0-9][0-9] | "2"[0-4][0-9] | "25"[0-5]
port ::= [0-9]{1,5}
path ::= ("/" segment)
segment ::= pchar{0,}
pchar ::= unreserved | pct-encoded | sub-delims | ":" | "@"
unreserved ::= [a-zA-Z0-9-._~]
pct-encoded ::= "%" hexdig hexdig
hexdig ::= [0-9a-fA-F]
sub-delims ::= [!$&'()*+,;=]
"##;

pub fn url_validate_clean(content: &str) -> Result<String, GrammarError> {
    if let Ok(url) = url_parse(content) {
        Ok(url.to_string())
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "Url".to_string(),
        })
    }
}

pub fn url_parse(content: &str) -> Result<Url, GrammarError> {
    if let Ok(url) = Url::from_str(content.trim()) {
        Ok(url)
    } else {
        Err(GrammarError::ParseValueError {
            content: content.to_string(),
            parse_type: "Url".to_string(),
        })
    }
}
