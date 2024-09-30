use super::*;

mod basic_completion_unit_tests {
    use super::*;

    #[cfg(feature = "llama_cpp_backend")]
    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_llama() -> crate::Result<()> {
        let llm_client = llama_cpp_tiny_llm().await?;
        basic_completion_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[cfg(feature = "llama_cpp_backend")]
    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_llama_logit_bias() -> crate::Result<()> {
        let llm_client = llama_cpp_tiny_llm().await?;
        basic_completion_logit_bias_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[cfg(feature = "mistral_rs_backend")]
    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_mistral() -> crate::Result<()> {
        let llm_client = mistral_rs_tiny_llm().await?;
        basic_completion_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_openai() -> crate::Result<()> {
        let llm_client = LlmClient::openai().gpt_3_5_turbo().init()?;
        basic_completion_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_openai_logit_bias() -> crate::Result<()> {
        let llm_client = LlmClient::openai().gpt_3_5_turbo().init()?;
        basic_completion_logit_bias_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_anthropic() -> crate::Result<()> {
        let llm_client = LlmClient::anthropic().claude_3_haiku().init()?;
        basic_completion_integration_tester(&llm_client).await?;
        Ok(())
    }

    #[tokio::test]
    #[serial]
    #[ignore]
    pub async fn test_perplexity() -> crate::Result<()> {
        let llm_client = LlmClient::perplexity().sonar_small().init()?;
        basic_completion_integration_tester(&llm_client).await?;
        Ok(())
    }
}

pub(super) async fn basic_completion_integration_tester(
    llm_client: &LlmClient,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_completion();
    gen.prompt()
        .add_user_message()
        .unwrap()
        .set_content("write a buzzfeed style listicle for the given input.")
        .append_content("boy howdy, how ya'll doing?");
    gen.max_tokens(200);
    let res = gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.is_empty());
    Ok(())
}

pub(super) async fn basic_completion_logit_bias_integration_tester(
    llm_client: &LlmClient,
) -> crate::Result<()> {
    let mut gen = llm_client.basic_completion();
    gen.prompt().reset_prompt();
    gen.prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input! Be excited.")
        .append_content("Boy howdy, how ya'll doing?");
    gen.max_tokens(100).add_logit_bias_from_char('!', -100.0);
    let res = gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains(" ! "));

    gen.prompt().reset_prompt();
    gen.clear_logit_bias();
    gen.prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    gen.max_tokens(100)
        .add_logit_bias_from_word("ya'll", -100.0);

    let res = gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains("ya'll"));

    gen.prompt().reset_prompt();
    gen.clear_logit_bias();
    gen.prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    gen.max_tokens(100)
        .add_logit_bias_from_text("boy howdy", -100.0);

    let res = gen.run().await?;
    println!("Response:\n {}\n", res.content);
    assert!(!res.content.contains("boy howdy"));

    gen.prompt().reset_prompt();
    gen.clear_logit_bias();
    gen.prompt()
        .add_user_message()
        .unwrap()
        .set_content("Write a buzzfeed style listicle for the given input!")
        .append_content("Boy howdy, how ya'll doing?");
    gen.max_tokens(100).add_logit_bias_from_word("cowbell", 5.0);
    let res = gen.run().await?;
    println!("Response:\n {}\n", res.content);
    // assert!(res.content.contains("cowbell"));

    Ok(())
}
