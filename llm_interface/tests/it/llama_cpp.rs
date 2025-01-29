use super::*;
use llm_interface::llms::local::llama_cpp::server::{get_all_server_pids, kill_server_from_model};

#[tokio::test]
#[serial]
async fn test_dropping_server() {
    let loaded = LlmInterface::llama_cpp().init().await.unwrap();
    std::mem::drop(loaded);
    let pids = get_all_server_pids().unwrap();
    assert!(pids.is_empty());
}

#[tokio::test]
#[serial]
async fn test_killing_server() {
    let loaded = LlmInterface::llama_cpp().init().await.unwrap();

    loaded.shutdown();

    let pids = get_all_server_pids().unwrap();
    assert!(pids.is_empty());
}

#[tokio::test]
#[serial]
async fn test_killing_server_with_model_id() {
    let loaded = LlmInterface::llama_cpp().init().await.unwrap();

    let model_id = loaded.model_id();
    kill_server_from_model(&model_id).unwrap();

    let pids = get_all_server_pids().unwrap();
    assert!(pids.is_empty());
}

#[tokio::test]
#[serial]
async fn test_multiple_servers() {
    let loaded_1 = LlmInterface::llama_cpp().init().await.unwrap();
    let _loaded_2 = LlmInterface::llama_cpp()
        .with_api_port("8081")
        .init()
        .await
        .unwrap();
    let pids = get_all_server_pids().unwrap();
    assert_eq!(pids.len(), 2);
    loaded_1.shutdown();
    let pids = get_all_server_pids().unwrap();
    assert_eq!(pids.len(), 1);
}

#[tokio::test]
#[serial]
async fn test_auto_gpu() {
    let backend = LlmInterface::llama_cpp().init().await.unwrap();
    assert!(
        backend
            .llama_cpp()
            .unwrap()
            .server
            .device_config
            .gpu_count()
            > 0
    );
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
#[serial]
async fn test_single_gpu_map() {
    use llm_interface::llms::local::LlmLocalTrait;

    let cuda_config = CudaConfig::new_from_cuda_devices(vec![0]);

    let backend = LlmInterface::llama_cpp()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        backend
            .llama_cpp()
            .unwrap()
            .server
            .device_config
            .gpu_count()
            == 1
    );
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
#[tokio::test]
#[serial]
async fn test_two_gpu_map() {
    let cuda_config = CudaConfig::new_from_cuda_devices(vec![0, 1]);

    let backend = LlmInterface::llama_cpp()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        backend
            .llama_cpp()
            .unwrap()
            .server
            .device_config
            .gpu_count()
            == 2
    );
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}

#[tokio::test]
#[serial]
async fn test_cpu_only() {
    let backend = LlmInterface::llama_cpp().cpu_only().init().await.unwrap();
    assert!(
        backend
            .llama_cpp()
            .unwrap()
            .server
            .device_config
            .gpu_count()
            == 0
    );
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[serial]
async fn test_metal() {
    let metal_config = MetalConfig::new_from_ram_gb(5.0);
    let backend = LlmInterface::llama_cpp()
        .metal_config(metal_config)
        .init()
        .await
        .unwrap();
    assert!(
        backend
            .llama_cpp()
            .unwrap()
            .server
            .device_config
            .gpu_count()
            == 1
    );
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}
