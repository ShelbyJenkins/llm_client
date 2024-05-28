use super::*;

impl<'a> RequestConfigTrait for GrammarTextList<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

#[derive(Clone)]
pub struct GrammarTextList<'a> {
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
    pub min_items: u16,
    pub max_items: u16,
    pub model_token_utilization: Option<f32>,
}

impl<'a> GrammarTextList<'a> {
    pub fn new(llm_client: &'a LlmClient, default_config: RequestConfig) -> Self {
        Self {
            req_config: default_config,
            llm_client,
            model_token_utilization: None,
            min_items: 1,
            max_items: 7,
        }
    }

    /// Sets the minimum number of items in the generated text list.
    /// The default value is 1.
    ///
    /// # Arguments
    ///
    /// * `min_items` - The minimum number of items.
    pub fn min_items(&mut self, min_items: u16) -> &mut Self {
        self.min_items = min_items;
        self
    }

    /// Sets the maximum number of items in the generated text list.
    /// The default value is 7.
    ///
    /// # Arguments
    ///
    /// * `max_items` - The maximum number of items.
    pub fn max_items(&mut self, max_items: u16) -> &mut Self {
        self.max_items = max_items;
        self
    }

    /// Runs the text generation process and returns the generated text list.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of strings representing the generated text list, or an error if the generation process fails.
    pub async fn run(&mut self) -> Result<Vec<String>> {
        let grammar = llm_utils::grammar::create_list_grammar(self.min_items, self.max_items);

        // Increases total tokens requested to account for the grammar on a per item basis
        // "- <text>\n"
        if self.req_config.requested_response_tokens.is_some() {
            let additional_tokens = self.max_items * 3;
            self.req_config.requested_response_tokens =
                Some(self.req_config.requested_response_tokens.unwrap() + additional_tokens as u32);
        }

        self.req_config
            .build_request(&self.llm_client.backend)
            .await?;

        self.req_config
            .set_max_tokens_for_request(self.model_token_utilization)?;

        let response = match &self.llm_client.backend {
            LlmBackend::Llama(backend) => {
                let res = backend
                    .text_generation_request(&self.req_config, None, Some(&grammar))
                    .await?;
                if backend.logging_enabled {
                    tracing::info!(?res);
                }
                res.content
            }
            // LlmBackend::MistralRs(_) => {
            //     panic!("Mistral backend is not supported for grammar based calls.")
            // }
            LlmBackend::OpenAi(_) => {
                panic!("OpenAI backend is not supported for grammar based calls.")
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for grammar based calls.")
            }
        };

        let response_items: Vec<String> = response
            .lines()
            .map(|line| line[1..].trim().to_string())
            .collect();

        Ok(response_items)
    }
}

pub async fn apply_test(mut text_gen: GrammarTextList<'_>) -> Result<()> {
    text_gen
        .system_content("Please provide keywords for the topics of this text.")
        .user_content(BNF)
        .max_tokens(100);

    let res = text_gen.run().await?;

    for (i, r) in res.into_iter().enumerate() {
        println!("\n{i}: {r}");
    }

    text_gen.system_content(
    "For each discrete topic in this text, please provide a short ELI5 sentence describing the topic.",
)
.max_items(5)
.min_items(3)
.user_content(BNF)
.max_tokens(300);

    let res = text_gen.run().await?;
    assert!(res.len() > 2);
    for (i, r) in res.into_iter().enumerate() {
        println!("\n{i}: {r}");
    }
    Ok(())
}

const BNF: &str = "In computer science, Backus–Naur form (/ˌbækəs ˈnaʊər/) (BNF or Backus normal form) is a notation used to describe the syntax of programming languages or other formal languages. It was developed by John Backus and Peter Naur. BNF can be described as a metasyntax notation for context-free grammars. Backus–Naur form is applied wherever exact descriptions of languages are needed, such as in official language specifications, in manuals, and in textbooks on programming language theory. BNF can be used to describe document formats, instruction sets, and communication protocols.

Over time, many extensions and variants of the original Backus–Naur notation have been created; some are exactly defined, including extended Backus–Naur form (EBNF) and augmented Backus–Naur form (ABNF). Invented in 1976.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[tokio::test]
    #[serial]
    pub async fn test() -> Result<()> {
        let llm = LlmClient::llama_backend().init().await?;

        let text_gen: GrammarTextList = llm.text().grammar_list();
        apply_test(text_gen).await?;
        Ok(())
    }
}
