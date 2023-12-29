use llm_client::agents::prelude::basic_text_gen;
use llm_client::prelude::{
    llama_cpp::models::{LlamaDef, LlamaPromptFormat},
    llama_cpp::server::kill_server,
    llm_openai::models::OpenAiDef,
    LlmDefinition,
};

const MISTRAL7BCHAT_MODEL_URL: &str =
    "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf";

#[tokio::main]
pub async fn main() {
    // Using OpenAI
    let llm_definition: LlmDefinition = LlmDefinition::OpenAiLlm(OpenAiDef::Gpt35Turbo);
    let response = basic_text_gen::generate(
        &llm_definition,
        None,           // Optional base prompt for system prompt
        Some("Howdy!"), // Optional input for user prompt
        Some("tests/prompt_templates/basic_text_gen.yaml"), // Optional path to yaml file for system prompt
        Some(0.5), // Model utilization is a percentage of max tokens for model
    )
    .await;
    println!("{}", response.unwrap());

    // Using Llama.cpp
    // Define the model you'd like to use.

    let llm_definition: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        MISTRAL7BCHAT_MODEL_URL,
        LlamaPromptFormat::Mistral7BChat,
        Some(9001), // Max tokens for model AKA context size
        Some(2),    // Number of threads to use for server
        Some(22),   // Layers to load to GPU. Dependent on VRAM
        None,       // This starts the llama.cpp server with embedding flag disabled
        None,       // Logging disabled
    ));

    let response = basic_text_gen::generate(
        &llm_definition,
        None,
        Some("Howdy!"),
        Some("tests/prompt_templates/basic_text_gen.yaml"),
        Some(0.5),
    )
    .await;
    println!("{}", response.unwrap());
    kill_server();
}
