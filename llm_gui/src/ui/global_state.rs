use dioxus::prelude::*;

use crate::api::{
    get_llm_client_state, submit_inference_request_llm_client, LlmGuiLocalModel, LoadState,
    ModelState,
};

#[derive(Clone, Copy)]
pub struct LlmGuiState {
    pub response: Signal<String>,
    pub query: Signal<String>,
    pub models: Signal<Vec<LlmGuiLocalModel>>,
    pub current_model: Signal<LlmGuiLocalModel>,
    pub inference_in_progress: Signal<bool>,
    pub available_memory_bytes: u64,
}

impl LlmGuiState {
    pub fn new(
        get_models_fn: Resource<Vec<LlmGuiLocalModel>>,
        available_memory_bytes_fn: Resource<u64>,
    ) -> Self {
        let models = get_models_fn().expect("Failed to get models");
        let current_model = models
            .iter()
            .find(|model| matches!(model.model_state, ModelState::Selected(_)))
            .expect("No currently selected model found")
            .clone();
        Self {
            response: Signal::new("Hola!".to_string()),
            query: Signal::new("".to_string()),
            inference_in_progress: Signal::new(false),
            models: Signal::new(models),
            current_model: Signal::new(current_model),
            available_memory_bytes: available_memory_bytes_fn().expect("Failed to get memory"),
        }
    }

    pub async fn submit_inference_request(&mut self) {
        if (self.inference_in_progress)() {
            return;
        };
        let current_model = self.get_and_update_current_model().await;
        self.response.set("Thinking    ".into());
        self.inference_in_progress.set(true);
        if let Ok(content) =
            submit_inference_request_llm_client((self.query)(), current_model.base.model_id).await
        {
            self.inference_in_progress.set(false);
            let _ = self.get_and_update_current_model().await;
            self.response.set(content);
        }
    }

    pub fn model_query(&self, query: String) -> Result<Vec<LlmGuiLocalModel>, crate::Error> {
        let models: Vec<LlmGuiLocalModel> =
            (self.models)().iter().map(|model| model.clone()).collect();
        let mut models = if query.is_empty() {
            models
        } else {
            models
                .iter()
                .filter(|model| model.base.model_id.to_lowercase().contains(&query))
                .map(|model| model.clone())
                .collect::<Vec<_>>()
        };
        models.sort_by(|a, b| {
            // First compare by ModelState priority
            let state_order = |state: &ModelState| -> i32 {
                match state {
                    ModelState::Selected(_) => 0, // Highest priority
                    ModelState::Downloaded => 1,
                    ModelState::Downloadable => 2,
                }
            };

            let state_comparison = state_order(&a.model_state).cmp(&state_order(&b.model_state));

            // If states are equal, sort by friendly_name
            if state_comparison == std::cmp::Ordering::Equal {
                a.base.friendly_name.cmp(&b.base.friendly_name)
            } else {
                state_comparison
            }
        });
        Ok(models)
    }

    async fn get_and_update_current_model(&mut self) -> LlmGuiLocalModel {
        let llm_client_state = get_llm_client_state().await.unwrap();
        let mut models: Write<'_, Vec<LlmGuiLocalModel>> = self.models.write();
        match models
            .iter_mut()
            .find(|model| matches!(model.model_state, ModelState::Selected(_)))
        {
            Some(model) => {
                if let Some(loaded_model_id) = &llm_client_state {
                    if *loaded_model_id == model.base.model_id {
                        model.model_state = ModelState::Selected(LoadState::Loaded);
                    } else {
                        model.model_state = ModelState::Selected(LoadState::Unloaded);
                    }
                } else {
                    model.model_state = ModelState::Selected(LoadState::Unloaded);
                }
                self.current_model.set(model.clone());
                model.clone()
            }
            None => {
                panic!("No currently selected model found");
            }
        }
    }

    pub async fn select_model(&mut self, model_id: &str) {
        let llm_client_state = get_llm_client_state().await.unwrap();

        let mut models: Write<'_, Vec<LlmGuiLocalModel>> = self.models.write();

        for model in models.iter_mut() {
            if model.base.model_id == model_id {
                if let Some(loaded_model_id) = &llm_client_state {
                    if *loaded_model_id == model.base.model_id {
                        model.model_state = ModelState::Selected(LoadState::Loaded);
                        self.current_model.set(model.clone());
                    } else {
                        model.model_state = ModelState::Selected(LoadState::Unloaded);
                        self.current_model.set(model.clone());
                    }
                } else {
                    model.model_state = ModelState::Selected(LoadState::Unloaded);
                    self.current_model.set(model.clone());
                }
            } else {
                model.reset_model_state();
            }
        }
    }

    pub fn available_memory_gigabytes(&self) -> String {
        format!(
            "{:.2} GB",
            self.available_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }
}
