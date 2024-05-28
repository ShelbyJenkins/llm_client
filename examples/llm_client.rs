use anyhow::Result;
use llm_client::prelude::*;

#[tokio::main]
pub async fn main() -> Result<()> {
    // Using a preset model from Hugging Face
    //
    let mut llm_client = LlmClient::llama_backend()
        .available_vram(48)
        .mistral_7b_instruct()
        .init()
        .await?;

    // Using a model from url
    //
    let mut _llm_client = LlmClient::llama_backend()
        .model_url("https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf")
        .ctx_size(4444)
        .n_gpu_layers(33) // Using a custom model requires configuring llama.cpp manually
        .init();

    // Using an OpenAI model
    //
    let mut _llm_client = LlmClient::openai_backend().gpt_4_o().init();

    // Using an Anthropic model
    //
    let mut _llm_client = LlmClient::anthropic_backend().claude_3_haiku().init();

    // llm_client has a default configuration that is passed to new requests
    //
    llm_client.temperature(0.9);
    llm_client.frequency_penalty(0.9);
    llm_client.presence_penalty(0.9);
    llm_client.top_p(0.9);
    llm_client.max_tokens(200);

    //
    // All features are accessible from the llm_client instance
    //

    // Text generation
    //
    let mut basic_text = llm_client.text().basic_text();

    basic_text.temperature(1.5); // Request configuration can be changed on a per-request basis
    let response = basic_text
        .system_content("You're a country robot!")
        .user_content("howdy!")
        .run()
        .await?;
    println!("Response: {response}");

    // Requests can be reused
    //
    let response = basic_text.user_content("hello!").run().await?;
    println!("Response: {response}");

    // Decider requests
    //
    let mut boolean_decider = llm_client.decider().boolean();

    boolean_decider.temperature(1.5); // Decider requests can also have custom configurations
    let response = boolean_decider
        .user_content("is this a good idea?")
        .run()
        .await?;
    println!("Response: {response}");

    // Deciders have three backend options: Grammar, LogitBias, and Basic.
    // Each LlmBackend has a default decider backend that can be overridden.
    //
    let mut _integer_decider = llm_client.decider().use_grammar_backend().integer();
    let mut _integer_decider = llm_client.decider().use_logit_bias_backend().integer();
    let mut _integer_decider = llm_client.decider().use_basic_backend().integer();

    // Deciders have some other options
    //
    let mut integer_decider = llm_client
        .decider()
        .use_grammar_backend()
        .best_of_n_votes(5)
        .dynamic_temperature(true)
        .decision_justification_token_count(300)
        .integer();

    let response = integer_decider
        .user_content("what is the best number?")
        .lower_bound(0)
        .upper_bound(7)
        .run()
        .await?;
    println!("Response: {response}");

    Ok(())
}
