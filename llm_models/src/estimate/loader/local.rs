use std::path::Path;

use bon::Builder;
use local_loader_builder::{IsComplete, IsUnset, SetLocalModelPath, State};
use serde::{Deserialize, Serialize};

use crate::{
    estimate::{loader::tokenizer::TokenizerLoader, memory::RuntimeMemorySpec},
    fs::{error::FileSystemError, file_path::ExistingFile},
};

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct LocalLoader {
    #[builder(setters(vis = "", name = local_model_path_internal))]
    pub local_model_path: ExistingFile,

    #[builder(default)]
    pub tokenizer_source: TokenizerLoader,

    #[builder(default)]
    pub memory_usage_config: RuntimeMemorySpec,
}

impl<S: State> LocalLoaderBuilder<S> {
    pub fn local_model_path<V>(
        self,
        value: V,
    ) -> Result<LocalLoaderBuilder<SetLocalModelPath<S>>, FileSystemError>
    where
        V: AsRef<Path>,
        S::LocalModelPath: IsUnset,
    {
        Ok(self.local_model_path_internal(ExistingFile::try_new(value)?))
    }
}

impl<State: local_loader_builder::State> LocalLoaderBuilder<State>
where
    State: IsComplete,
{
    pub fn build(self) -> LocalLoader {
        self.build_internal()
    }

    // pub fn load(self) -> GgufModel {
    //     self.build_internal().load()
    // }
}

impl LocalLoader {}
