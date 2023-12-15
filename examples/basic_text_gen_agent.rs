use llm_client::agents::prelude::basic_text_gen;
use llm_client::prelude::{
    llama_cpp::models::{LlamaLlmModel, LlamaPromptFormat},
    llama_cpp::server::start_server,
    llm_openai::models::OpenAiLlmModels,
    LlmDefinition,
};

#[tokio::main]
pub async fn main() {
    // Using OpenAI
    let llm_definition: LlmDefinition = LlmDefinition::OpenAiLlm(OpenAiLlmModels::Gpt35Turbo);
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
    let zephyr_7b_chat = LlamaLlmModel::new(
        "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf", // Full url to model file on hf
        LlamaPromptFormat::Mistral7BChat, // Prompt format
        Some(2000), // Max tokens for model AKA context size
    );
    let _ = start_server(
        &zephyr_7b_chat.model_id,
        &zephyr_7b_chat.model_filename,
        None,    // HF token if you want to use a private model
        Some(4), // Number of threads to use for server
        Some(zephyr_7b_chat.max_tokens_for_model),
        Some(12), // Layers to load to GPU. Dependent on VRAM
    )
    .await;

    let response = basic_text_gen::generate(
        &LlmDefinition::LlamaLlm(zephyr_7b_chat),
        None,
        Some("Howdy!"),
        Some("tests/prompt_templates/basic_text_gen.yaml"),
        Some(0.5),
    )
    .await;
    println!("{}", response.unwrap());
}
