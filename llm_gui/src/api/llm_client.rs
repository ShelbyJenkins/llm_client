use dioxus::fullstack::prelude::*;

#[server]
pub async fn get_llm_client_state() -> Result<Option<String>, ServerFnError> {
    Ok(server::get_llm_client_state())
}

#[server]
pub async fn available_memory_bytes() -> Result<u64, ServerFnError> {
    Ok(server::available_memory_bytes())
}

#[server]
pub async fn submit_inference_request_llm_client(
    prompt: String,
    selected_model_id: String,
) -> Result<String, ServerFnError> {
    let llm_client = server::get_or_init_llm_client(selected_model_id)
        .await
        .unwrap();
    let mut basic_completion = llm_client.basic_completion();

    basic_completion
        .prompt()
        .add_system_message()
        .unwrap()
        .set_content("You're a country robot.");
    basic_completion
        .prompt()
        .add_user_message()
        .unwrap()
        .set_content(prompt);
    let response = basic_completion.run().await.unwrap();
    Ok(response.content.to_string())
}

#[cfg(feature = "server")]
pub mod server {
    use llm_client::*;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static LLM_CLIENT: Mutex<Option<LlmClient>> = Mutex::new(None);
    static AVAILABLE_MEMORY_BYTES: OnceLock<u64> = OnceLock::new();

    pub fn get_llm_client() -> Option<LlmClient> {
        let guard = LLM_CLIENT.lock().unwrap();
        if let Some(client) = guard.as_ref() {
            Some(client.clone())
        } else {
            None
        }
    }

    pub async fn get_or_init_llm_client(
        selected_model_id: String,
    ) -> Result<LlmClient, crate::Error> {
        if let Some(client) = get_llm_client() {
            if client.backend.model_id() == selected_model_id {
                return Ok(client.clone());
            }
            client.shutdown();
        }

        let client = LlmClient::llama_cpp()
            .preset_from_str(&selected_model_id)?
            .init()
            .await
            .unwrap();

        {
            let mut guard = LLM_CLIENT.lock().unwrap();
            *guard = Some(client.clone());
        }

        Ok(client)
    }

    pub fn get_llm_client_state() -> Option<String> {
        if let Some(client) = get_llm_client() {
            Some(client.backend.model_id().to_owned())
        } else {
            None
        }
    }

    pub fn available_memory_bytes() -> u64 {
        *AVAILABLE_MEMORY_BYTES.get_or_init(|| {
            if let Some(client) = get_llm_client() {
                client
                    .device_config()
                    .unwrap()
                    .available_memory_bytes()
                    .unwrap() as u64
            } else {
                let mut device_config = DeviceConfig::default();
                device_config.initialize().unwrap();
                device_config.available_memory_bytes().unwrap() as u64
            }
        })
    }
}
