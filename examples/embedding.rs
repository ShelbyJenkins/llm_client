use llm_client::providers::llama_cpp::models::{
    LlamaDef, DEFAULT_CTX_SIZE, DEFAULT_N_GPU_LAYERS, DEFAULT_THREADS,
    TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT, TEST_LLM_URL_2_INSTRUCT,
};
use llm_client::{
    providers::llm_openai::models::OpenAiDef, EmbeddingExceedsMaxTokensBehavior, LlmDefinition,
    ProviderClient,
};

#[tokio::main]
pub async fn main() {
    let client_openai: ProviderClient =
        ProviderClient::new(&LlmDefinition::OpenAiLlm(OpenAiDef::EmbeddingAda002), None).await;

    let _: Vec<Vec<f32>> = client_openai
        .generate_embeddings(
            &vec![
                "Hello, my dog is cute".to_string(),
                "Hello, my cat is cute".to_string(),
            ],
            Some(EmbeddingExceedsMaxTokensBehavior::Panic),
        )
        .await
        .unwrap();

    // llama.cpp works as well, but it's not fully qualified yet and is limited to the dimensions of the model.
    let llm_definition: LlmDefinition = LlmDefinition::LlamaLlm(LlamaDef::new(
        TEST_LLM_URL_2_INSTRUCT,
        TEST_LLM_PROMPT_TEMPLATE_2_INSTRUCT,
        Some(DEFAULT_CTX_SIZE),     // Max tokens for model AKA context size
        Some(DEFAULT_THREADS),      // Number of threads to use for server
        Some(DEFAULT_N_GPU_LAYERS), // Layers to load to GPU. Dependent on VRAM
        Some(true),                 // This starts the llama.cpp server with embedding flag
        Some(true),                 // Logging enabled
    ));

    let client_llama: ProviderClient = ProviderClient::new(&llm_definition, None).await;

    let _: Vec<Vec<f32>> = client_llama
        .generate_embeddings(
            &vec![
                "Hello, my dog is cute".to_string(),
                "Hello, my cat is cute".to_string(),
            ],
            Some(EmbeddingExceedsMaxTokensBehavior::Panic),
        )
        .await
        .unwrap();
}
