use llm_client::*;

/// The most basic request. Implemented for API based LLMs and for the sake of completness.
#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    // Create a client. We'll use an Anthropic model
    // For API based LLMs, only the most basic completion is supported.
    let llm_client = LlmClient::anthropic().claude_3_haiku().init().unwrap();

    // Or use a preset model from Hugging Face
    // let llm_client = LlmClient::llama_cpp()
    //     .mistral_7b_instruct_v0_3()
    //     .init()
    //     .await
    //     .unwrap();

    // Text generation
    let mut basic_completion = llm_client.basic_completion();

    basic_completion
        .prompt()
        .add_system_message()
        .unwrap()
        .set_content("You're a country robot.");
    basic_completion
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("howdy!");
    let response = basic_completion.run().await.unwrap();
    println!("Response: {}", response.content);

    // Requests can be reused. All fields are preserved between runs
    let response = basic_completion.run().await.unwrap();
    println!("Response: {}", response.content);

    // You change the request settings
    basic_completion
        .temperature(1.5)
        .frequency_penalty(0.9)
        .max_tokens(200);

    // Or create a conversation
    basic_completion
        .prompt()
        .add_assistant_message()
        .unwrap()
        .set_content(response.content);
    basic_completion
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content("That's nice, but how's the weather.unwrap()");

    // To start fresh you can reset the request, or create a new one
    basic_completion.reset_request();
    let _basic_completion = llm_client.basic_completion();
}
