use llm_client::prelude::*;

/// This example demonstrates how to use the `BasicPrimitiveWorkflow` to generate a response where the output is constrained to one of the defined primitive types. See `llm_client/src/primitives/mod.rs` for the available primitive types.

#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    // Using a preset model from Hugging Face
    let llm_client = LlmClient::anthropic().claude_3_haiku().init().unwrap();

    // A request constrained to N sentences
    let mut reason_request = llm_client.basic_primitive().sentences();
    reason_request.primitive.min_count(3).max_count(3);
    reason_request
        .instructions()
        .set_content("Can you write a haiku?");
    let response = reason_request.return_primitive().await.unwrap();
    println!("{}", response);

    // A request constrained to N words
    let mut reason_request = llm_client.basic_primitive().words();
    reason_request.primitive.min_count(1).max_count(9);
    reason_request
        .instructions()
        .set_content("Can you give some synonyms for 'happy'?");
    let response = reason_request.return_primitive().await.unwrap();
    println!("{}", response);
}
