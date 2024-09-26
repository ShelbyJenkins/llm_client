use super::LocalLlmConfig;
use crate::{
    logging::LoggingConfig,
    requests::completion::{CompletionError, CompletionRequest, CompletionResponse},
};

use devices::mistral_rs_device_map;
use llm_utils::models::local_model::{gguf::GgufLoader, LocalLlmModel};
use mistralrs::{
    DefaultSchedulerMethod,  GGUFLoaderBuilder, GGUFSpecificConfig, LocalModelPaths, MistralRs, MistralRsBuilder, ModelDType, ModelPaths, Pipeline, Response, SchedulerConfig
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
        let client = Self::init_from_gguf_local(&model, &config.local_config)?;
        Ok(Self {
            client,
            config,
            model,
        })
    }

    fn init_from_gguf_local(
        model: &LocalLlmModel,
        local_config: &LocalLlmConfig,
    ) -> crate::Result<std::sync::Arc<MistralRs>> {
        std::env::set_var("MISTRALRS_DEBUG", "1");

        let loader = GGUFLoaderBuilder::new(
            None,
            None,
            "".to_string(),
            vec![],
            GGUFSpecificConfig {
                prompt_batchsize: None,
                topology: None,
            },
        )
        .build();

        let local_model_paths: Vec<std::path::PathBuf> = vec![model.local_model_path.clone()];

        let paths: Box<dyn ModelPaths> = Box::new(LocalModelPaths {
            tokenizer_filename: std::path::PathBuf::new(),
            config_filename: std::path::PathBuf::new(),
            template_filename: None,
            filenames: local_model_paths,
            xlora_adapter_filenames: None,
            xlora_adapter_configs: None,
            classifier_path: None,
            classifier_config: None,
            xlora_ordering: None,
            gen_conf: None,
            lora_preload_adapter_info: None,
            preprocessor_config: None,
            processor_config: None,
        });

        let (device, mapper) = mistral_rs_device_map(&local_config.device_config)?;

        let pipeline: std::sync::Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>> = loader.load_model_from_path(
            &paths,
            &ModelDType::Auto,
            &device,
            false,
            mapper,
            None,
            None,
        )?;

        Ok(MistralRsBuilder::new(
            pipeline,
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
            },
        )
        .with_log("gguf_tensors.txt".to_string())
        .build())
    }

    pub async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);

        let mistral_request = completion::new(request, tx)?;

        self.client
            .get_sender()
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?
            .send(mistral_request)
            .await
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?;

        match rx.recv().await {
            None =>  Err(CompletionError::LocalClientError("MistralRsBackend request error: Response is None".to_string())),
            Some(res) => 
            match res {
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
