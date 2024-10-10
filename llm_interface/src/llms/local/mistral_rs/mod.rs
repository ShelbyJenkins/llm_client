use std::num::NonZeroUsize;

use super::LocalLlmConfig;
use crate::requests::completion::{CompletionError, CompletionRequest, CompletionResponse};

use devices::mistral_rs_device_map;
use llm_devices::logging::LoggingConfig;
use llm_models::local_model::{gguf::GgufLoader, LocalLlmModel};
use mistralrs::{
    DefaultSchedulerMethod,  GGUFLoaderBuilder, GGUFSpecificConfig, MemoryGpuConfig, MistralRs, MistralRsBuilder, ModelDType,  PagedAttentionConfig,  Response, SchedulerConfig, TokenSource
};


pub mod builder;
pub mod completion;
pub mod devices;

pub struct MistralRsBackend {
    pub model: LocalLlmModel,
    pub config: MistralRsConfig,
    client: std::sync::Arc<MistralRs>,
}

impl MistralRsBackend {
    pub async fn new(mut config: MistralRsConfig, llm_loader: GgufLoader) -> crate::Result<Self> {
        config.logging_config.load_logger()?;
        config.local_config.device_config.initialize()?;
        let model = config.local_config.load_model(llm_loader)?;
        let client = Self::init_from_gguf_local(&model, &config.local_config).await?;
        Ok(Self {
            client,
            config,
            model,
        })
    }

    async fn init_from_gguf_local(
        model: &LocalLlmModel,
        local_config: &LocalLlmConfig,
    ) -> crate::Result<std::sync::Arc<MistralRs>> {
        std::env::set_var("MISTRALRS_DEBUG", "1");

        let prompt_batchsize = if local_config.batch_size > 0 {
            Some(NonZeroUsize::new(local_config.batch_size.try_into().unwrap()).unwrap())
        } else {
            anyhow::bail!("`prompt_batchsize` must be a strictly positive integer, got 0.",)

        };

        let cache_config = Some(PagedAttentionConfig::new(
            None,
            512,
            MemoryGpuConfig::Utilization(0.9),
        )?);

        let (device, mapper) = mistral_rs_device_map(&local_config.device_config)?;

        let directory = model.local_model_path.parent().and_then(|p| p.to_str()).expect("Model path must have a parent directory");
        let filename = model.local_model_path.file_name().and_then(|s| s.to_str()).expect("Model path must have a filename");

        let loader = GGUFLoaderBuilder::new(
            None,
            None,
            directory.to_string(),
            vec![filename.to_string()],
            GGUFSpecificConfig {
                prompt_batchsize,
                topology: None,
            },
        )
        .build();

        let pipeline = loader.load_model_from_hf(
            None,
            TokenSource::None,
            &ModelDType::Auto,
            &device,
            false,
            mapper,
            None,
            cache_config,
        )?;

        let scheduler_config = if cache_config.is_some() {
            // Handle case where we may have device mapping
            if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: 1,
                    config: cache_config.clone(),
                }
            } else {
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(
                        (1)
                            .try_into()
                            .unwrap(),
                    ),
                }
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(
                    (5)
                        .try_into()
                        .unwrap(),
                ),
            }
        };
        Ok(MistralRsBuilder::new(
            pipeline,
            scheduler_config,
        )
        .with_throughput_logging()
        .build())
    }

    pub async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {

        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let id = 0;
        let mistral_request = completion::new(request, tx, id)?;

       let sender = self.client
            .get_sender()
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?;
        sender
            .send(mistral_request)
            .await
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?;

        let response = match rx.recv().await {
            Some(response) => response,
            None => return Err(CompletionError::LocalClientError("MistralRsBackend request error: Response is None".to_string()))
        };
    
        match response {
            Response::InternalError(e) | Response::ValidationError(e) => {
                Err(CompletionError::LocalClientError(e.to_string()))
            }
            Response::Chunk(_) | Response::Done(_) | Response::CompletionChunk(_) => {
                Err(CompletionError::LocalClientError("MistralRsBackend request error: Response::Chunk(_) | Response::Done(_) | Response::CompletionChunk(_)".to_string()))
                
            }
            Response::ModelError(e, _) | Response::CompletionModelError(e, _) => {
                Err(CompletionError::LocalClientError(e.to_string()))
            }
            Response::CompletionDone(completion) => {
                crate::trace!(?completion);
                Ok(CompletionResponse::new_from_mistral(request, completion)?)
            }
            Response::ImageGeneration(_) => {
                Err(CompletionError::LocalClientError("MistralRsBackend request error: Response::ImageGeneration(_)".to_string()))
            }
        }
        
    }

}

#[derive(Clone, Debug)]
pub struct MistralRsConfig {
    pub logging_config: LoggingConfig,
    pub local_config: LocalLlmConfig,
}

impl Default for MistralRsConfig {
    fn default() -> Self {
        Self {
            logging_config: LoggingConfig {
                logger_name: "mistral_rs".to_string(),
                ..Default::default()
            },
            local_config: LocalLlmConfig::default(),
        }
    }
}

impl MistralRsConfig {
    pub fn new() -> Self {
        Default::default()
    }
}
