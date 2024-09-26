use llm_client::prelude::*;

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
    let cuda_map = CudaDeviceMap::new(vec![0], None);

    let llm_client = LlmClient::llama_cpp()
        .cuda_device_map(cuda_map)
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
    let cuda_map = CudaDeviceMap::new(vec![0, 1], None);

    let llm_client = LlmClient::llama_cpp()
        .cuda_device_map(cuda_map)
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
