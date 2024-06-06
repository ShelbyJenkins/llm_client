use super::*;
use llm_utils::grammar::RestrictedCharacterSet;

impl<'a> RequestConfigTrait for GrammarText<'a> {
    fn config_mut(&mut self) -> &mut RequestConfig {
        &mut self.req_config
    }
}

#[derive(Clone)]
pub struct GrammarText<'a> {
    pub req_config: RequestConfig,
    pub llm_client: &'a LlmClient,
    pub restricted: Vec<RestrictedCharacterSet>,
    pub model_token_utilization: Option<f32>,
}

impl<'a> GrammarText<'a> {
    pub fn new(llm_client: &'a LlmClient, default_config: RequestConfig) -> Self {
        Self {
            req_config: default_config,
            llm_client,
            restricted: Vec::new(),
            model_token_utilization: None,
        }
    }

    /// Sets the model token utilization. This sets the 'max_tokens' parameter for the request to a percent of the available 'ctx_size' for the model or server settings.
    /// The value should be between 0.0 and 1.0.
    ///
    /// # Arguments
    ///
    /// * `model_token_utilization` - The model token utilization value.
    pub fn model_token_utilization(&mut self, model_token_utilization: f32) -> &mut Self {
        self.model_token_utilization = Some(model_token_utilization);
        self
    }

    /// Excludes lowercase alphabetic characters from the generated text.
    pub fn restrict_alpha_lower(&mut self) -> &mut Self {
        self.restricted.push(RestrictedCharacterSet::AlphaLower);
        self
    }

    /// Excludes uppercase alphabetic characters from the generated text.
    pub fn restrict_alpha_upper(&mut self) -> &mut Self {
        self.restricted.push(RestrictedCharacterSet::AlphaUpper);
        self
    }

    /// Excludes number characters (any combination of 0-9) from the generated text.
    pub fn restrict_numeric(&mut self) -> &mut Self {
        self.restricted.push(RestrictedCharacterSet::Numeric);
        self
    }

    /// Excludes punctuation characters (.,!?,;'") from the generated text.
    pub fn restrict_grammar_punctuation(&mut self) -> &mut Self {
        self.restricted
            .push(RestrictedCharacterSet::PunctuationGrammar);
        self
    }

    /// Excludes extended punctuation characters (:{}()<>@#$%^&*+=~|/\[]) from the generated text.
    pub fn restrict_extended_punctuation(&mut self) -> &mut Self {
        self.restricted
            .push(RestrictedCharacterSet::PunctuationExtended);
        self
    }

    /// Clears all the restrictions on the generated text. Clears all the restrictions set by the restrict_* methods from the request object.
    /// Useful for testing.
    pub fn clear_restricted(&mut self) -> &mut Self {
        self.restricted.clear();
        self
    }

    /// Runs the text generation request and returns the generated text.
    ///
    /// # Returns
    ///
    /// A Result containing the generated text as a String, or an error if the request fails.
    pub async fn run(&mut self) -> Result<String> {
        let grammar = llm_utils::grammar::create_text_structured_grammar(self.restricted.clone());

        if self.llm_client.backend.logging_enabled() {
            tracing::info!(?grammar);
        }

        self.req_config
            .build_request(&self.llm_client.backend)
            .await?;
        self.req_config
            .set_max_tokens_for_request(self.model_token_utilization)?;

        match &self.llm_client.backend {
            LlmBackend::Llama(backend) => {
                backend
                    .text_generation_request(&self.req_config, None, Some(&grammar))
                    .await
            }
            #[cfg(feature = "mistralrs_backend")]
            LlmBackend::MistralRs(_) => {
                panic!("Mistral backend is not supported for grammar based calls.")
            }
            LlmBackend::OpenAi(_) => {
                panic!("OpenAI backend is not supported for grammar based calls.")
            }
            LlmBackend::Anthropic(_) => {
                panic!("Anthropic backend is not supported for grammar based calls.")
            }
        }
    }
}

#[warn(dead_code)]
pub async fn apply_test(mut text_gen: GrammarText<'_>) -> Result<()> {
    const DOOM: &str = "Doom (stylized as DOOM) is an American media franchise created by John Carmack, John Romero, Adrian Carmack, Kevin Cloud, and Tom Hall.[1] The series usually focuses on the exploits of an unnamed space marine (often referred to as Doomguy or Doom Slayer) operating under the auspices of the Union Aerospace Corporation (UAC), who fights hordes of demons and the undead to save Earth from an apocalyptic invasion.
    
The original Doom is considered one of the first pioneering first-person shooter games, introducing to IBM-compatible computers features such as 3D graphics, third-dimension spatiality, networked multiplayer gameplay, and support for player-created modifications with the Doom WAD format. Over ten million copies of games in the Doom series have been sold; the series has spawned numerous sequels, novels, comic books, board games, and film adaptations.


The Doom video games consist of first-person shooters in which the player controls an unnamed space marine commonly referred to as Doomguy; in the 2016 series, the protagonist is called the \"Doom Slayer\" or just \"Slayer\" in later entries. The player battles the forces of Hell, consisting of demons and the undead. The games are usually set within sprawling bases on Mars or its moons, while some parts occur in Hell. The classic series only focused on the story, much of which was in the manuals rather than the games.[2] More recent titles, notably the 2016 series, would feature a heavier focus on narrative.[3]

The original game featured eight weapons, designed so that no weapon became obsolete after the acquisition of another. With the player carrying all these weapons at once, the strategy of \"gun juggling\"—rapidly switching between the weapons depending on circumstance—can be employed.[4] Outside of combat mechanics, Doom levels often feature mazes, colored key cards and hidden areas.[5][6] Due to technical limitations, the player could not jump or look up and down in the classic series. These features were added in newer titles.[7]";

    let res = text_gen
        .system_content(DOOM)
        .user_content("Why is doom so loud?")
        .restrict_alpha_lower()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('d'));
    println!("\n{res}");

    text_gen.clear_restricted();
    let res = text_gen
        .system_content(DOOM)
        .user_content("How is the title of the game Doom written?")
        .restrict_alpha_upper()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('D'));
    println!("\n{res}");

    text_gen.clear_restricted();
    let res = text_gen
        .system_content(DOOM)
        .user_content("How many weapons was in the original Doom game?")
        .restrict_numeric()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('8'));
    assert!(res.contains("eight"));
    println!("\n{res}");

    text_gen.clear_restricted();
    let res = text_gen
        .system_content(DOOM)
        .user_content("Generate a numbered list of doom creators!")
        .restrict_numeric()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('1'));
    println!("\n{res}");

    text_gen.clear_restricted();
    let res = text_gen
        .system_content(DOOM)
        .user_content("Tell me about the creators of Doom.")
        .restrict_grammar_punctuation()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('.'));
    println!("\n{res}");

    text_gen.clear_restricted();
    let res = text_gen
        .system_content(DOOM)
        .user_content("Generate a numbered list of doom creators!")
        .restrict_extended_punctuation()
        .max_tokens(100)
        .run()
        .await?;
    assert!(!res.contains('#'));
    println!("\n{res}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    #[tokio::test]
    #[serial]
    pub async fn test() -> Result<()> {
        let llm = LlmClient::llama_backend().init().await?;
        let text_gen = llm.text().grammar_text();
        apply_test(text_gen).await?;
        Ok(())
    }
}
