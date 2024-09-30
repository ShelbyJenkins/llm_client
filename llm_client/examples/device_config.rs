use llm_client::prelude::*;

#[cfg(not(target_os = "macos"))]
#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    // Automatically detect and use all available GPUs
    let llm_client = LlmClient::llama_cpp().init().await.unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            > 0
    );

    // Use only a single GPU with index/ordinal 0
    let cuda_config = CudaConfig::new(vec![0], None);

    let llm_client = LlmClient::llama_cpp()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 1
    );

    // Use two GPUs with indices/ordinals 0 and 1
    let cuda_config = CudaConfig::new(vec![0, 1], None);

    let llm_client = LlmClient::llama_cpp()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 2
    );

    // Use only the CPU
    let llm_client = LlmClient::llama_cpp().cpu_only().init().await.unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 0
    );
}

#[cfg(target_os = "macos")]
#[tokio::main(flavor = "current_thread")]
pub async fn main() {
    // Automatically detect and use the Metal GPU
    let llm_client = LlmClient::llama_cpp().init().await.unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 1
    );

    // Use a Metal Config
    let metal_config = MetalConfig::new_from_ram_gb(5.0);

    let llm_client = LlmClient::llama_cpp()
        .metal_config(metal_config)
        .init()
        .await
        .unwrap();

    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 1
    );

    // Use only the CPU
    let llm_client = LlmClient::llama_cpp()
        .use_metal(false)
        .init()
        .await
        .unwrap();
    assert!(
        llm_client
            .backend
            .llama_cpp()
            .unwrap()
            .server
            .local_config
            .device_config
            .gpu_count()
            == 0
    );
}
