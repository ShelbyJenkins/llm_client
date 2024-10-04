pub mod types;
use llm_client::{CompletionResponse, LlmClient};
use types::*;

use crate::backends::TestBackendConfig;
use llm_client::RequestConfigTrait;

impl SpeedBenchmark {
    pub async fn run(&mut self) -> crate::Result<()> {
        self.start_time = std::time::Instant::now();
        for backend in &self.backends {
            let res = self.benchmark_backend(backend).await?;
            self.backend_results.push(res);
        }

        self.finalize();

        Ok(())
    }

    async fn benchmark_backend(&self, backend: &TestBackendConfig) -> crate::Result<BackendResult> {
        let mut result = BackendResult::new(backend);
        for model in &self.models {
            let llm_client = match backend.to_llm_client_with_preset(model).await {
                Ok(client) => client,
                Err(e) => {
                    eprintln!("Error creating client: {}", e);
                    continue;
                }
            };
            let res = self.benchmark_model(&llm_client).await?;
            llm_client.shutdown();
            std::mem::drop(llm_client);
            result.model_results.push(res);
        }
        Ok(result)
    }

    async fn benchmark_model(&self, llm_client: &LlmClient) -> crate::Result<ModelResult> {
        let mut result = ModelResult::new(llm_client.backend.model_id());
        for max_tok in &self.max_tokens {
            let res = self.benchmark_run(llm_client, *max_tok).await?;
            result.runs.push(RunResult::new(*max_tok, res));
        }
        Ok(result)
    }

    async fn benchmark_run(
        &self,
        llm_client: &LlmClient,
        max_tok: u32,
    ) -> crate::Result<CompletionResponse> {
        let mut gen = llm_client.basic_completion();
        gen.prompt().add_user_message()?.set_content(&self.prompt);
        gen.max_tokens(max_tok.into());
        gen.run().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_utils::models::local_model::gguf::preset::LlmPreset;

    #[tokio::test]
    #[ignore]
    pub async fn test_llama() -> crate::Result<()> {
        let mut benchmark = SpeedBenchmark {
            models: vec![
                LlmPreset::Phi3_5MiniInstruct,
                LlmPreset::Llama3_1_8bInstruct,
                LlmPreset::Llama3_2_1bInstruct,
                LlmPreset::Llama3_2_3bInstruct,
            ],
            backends: vec![TestBackendConfig::default_llama_cpp()],
            ..Default::default()
        };
        benchmark.run().await?;
        println!("{benchmark}",);
        Ok(())
    }

    #[cfg(feature = "mistral_rs_backend")]
    #[tokio::test]
    #[ignore]
    pub async fn test_mistral() -> crate::Result<()> {
        let mut benchmark = SpeedBenchmark {
            models: vec![LlmPreset::Llama3_2_1bInstruct],
            backends: vec![TestBackendConfig::default_mistral_rs()],
            ..Default::default()
        };
        benchmark.run().await?;
        println!("{benchmark}",);
        Ok(())
    }
}
