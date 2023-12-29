use llm_client::prelude::{
    llama_cpp::models::{LlamaDef, LlamaPromptFormat},
    llama_cpp::server::kill_server,
    prompting, LlmDefinition, ProviderClient,
};

const MISTRAL7BCHAT_MODEL_URL: &str =
    "https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q5_K_M.gguf";

#[tokio::main]
pub async fn main() {
    // In basic_text_gen_agent.rs we interact with a prebuilt 'agent'
    // Here, we will interact directly with the llm_client which gives us more control over the process

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

    // Init a client and make some changes to the default model params
    let mut llm_client = ProviderClient::new(&llm_definition, None).await;
    llm_client.temperature(Some(0.9));
    llm_client.frequency_penalty(Some(0.9));
    llm_client.presence_penalty(Some(0.9));
    llm_client.top_p(Some(0.9));
    llm_client.max_tokens_for_model(Some(2000));

    // Generate a prompt with default formatting
    let base_prompt = "This is a prompt. ";
    let prompt_template_path = "path/to/prompt_template.json";
    let feature = "A feature is extra content provided in the user prompt to the model.";

    let prompt: std::collections::HashMap<String, std::collections::HashMap<String, String>> =
        prompting::create_prompt_with_default_formatting(
            prompting::load_system_prompt_template(Some(base_prompt), Some(prompt_template_path)),
            Some(feature),
            Some("Howdy!"),
        );

    // The prompt will be formatted for the model provided by the llm_definition by the llm_client
    let response = llm_client
        .generate_text(&prompt, &None, Some(0.5), None, None)
        .await;
    if let Err(error) = response {
        panic!("Failed to generate text: {}", error);
    } else {
        println!("{}", response.unwrap());
    }

    // The server is still running, and needs to be killed manually for now.
    // In the future I will work out a way to kill the server automaticaly somehow.
    kill_server();
}
