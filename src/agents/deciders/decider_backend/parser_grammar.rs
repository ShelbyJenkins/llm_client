use super::*;
use gbnf::{
    Grammar,
    GrammarItem,
    NonTerminalSymbol,
    Production,
    ProductionItem,
    RepetitionType,
    Rule,
    TerminalSymbol,
};

pub async fn make_grammar_parser_request(
    llm_client: &LlmClient,
    req_config: &RequestConfig,
    grammar: Option<&str>,
) -> Result<String> {
    match &llm_client.backend {
        LlmBackend::Llama(backend) => {
            backend
                .decision_request(req_config, None, None, grammar)
                .await
        }
        // LlmBackend::MistralRs(_) => todo!(),
        LlmBackend::OpenAi(_) => {
            panic!("OpenAI backend is not supported for Grammar based calls.")
        }
        LlmBackend::Anthropic(_) => {
            panic!("Anthropic backend is not supported for Grammar based calls.")
        }
    }
}

pub fn create_parser_grammar(choices: &[DeciderChoice]) -> String {
    fn add_choice(parser_key: &str) -> Production {
        Production {
            items: vec![ProductionItem::Terminal(
                TerminalSymbol {
                    value: parser_key.to_string(),
                },
                RepetitionType::One,
            )],
        }
    }
    let production_choices: Vec<Production> = choices
        .iter()
        .map(|choice| add_choice(&choice.parser_key))
        .collect();

    let g = Grammar {
        items: vec![
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "root".to_string(),
                },
                rhs: Production {
                    items: vec![ProductionItem::NonTerminal(
                        NonTerminalSymbol {
                            name: "choices".to_string(),
                        },
                        RepetitionType::One,
                    )],
                },
            }),
            GrammarItem::Rule(Rule {
                lhs: NonTerminalSymbol {
                    name: "choices".to_string(),
                },
                rhs: Production {
                    items: vec![ProductionItem::OneOf(production_choices)],
                },
            }),
        ],
    };
    g.to_string()
}
