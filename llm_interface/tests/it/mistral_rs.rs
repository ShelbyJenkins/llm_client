use super::*;

#[tokio::test]
#[serial]
async fn test_auto_gpu() {
    let backend = LlmInterface::mistral_rs().init().await.unwrap();
    assert!(
        backend
            .mistral_rs()
            .unwrap()
            .config
            .local_config
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

#[tokio::test]
#[serial]
async fn test_single_gpu_map() {
    let cuda_config = CudaConfig::new_from_cuda_devices(vec![0]);

    let backend = LlmInterface::mistral_rs()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        backend
            .mistral_rs()
            .unwrap()
            .config
            .local_config
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

#[tokio::test]
#[serial]
async fn test_two_gpu_map() {
    let cuda_config = CudaConfig::new_from_cuda_devices(vec![0, 1]);

    let backend = LlmInterface::mistral_rs()
        .cuda_config(cuda_config)
        .init()
        .await
        .unwrap();
    assert!(
        backend
            .mistral_rs()
            .unwrap()
            .config
            .local_config
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
    let backend = LlmInterface::mistral_rs().cpu_only().init().await.unwrap();
    assert!(
        backend
            .mistral_rs()
            .unwrap()
            .config
            .local_config
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
