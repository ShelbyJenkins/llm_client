use super::*;

#[tokio::test]
#[serial]
async fn test_perplexity() {
    let backend = LlmInterface::perplexity().init().unwrap();
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
async fn test_openai() {
    let backend = LlmInterface::openai().init().unwrap();
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
async fn test_anthropic() {
    let backend = LlmInterface::anthropic().init().unwrap();
    let mut req = CompletionRequest::new(backend);
    req.prompt
        .add_user_message()
        .unwrap()
        .set_content("Hello, world!");

    let res = req.request().await.unwrap();
    println!("{res}");
}
