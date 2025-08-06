use std::path::PathBuf;

use bon::Builder;
use gguf_loader_builder::{
    IsComplete, IsSet, IsUnset, SetFromHfFile, SetFromLocalPath, SetFromSmartSelect,
    SetHasModelSource, State,
};
use serde::{Deserialize, Serialize};

use crate::{gguf::loader::smart::SmartLoader, hf::types::HfFile};

// // HuggingFace token and environment variable for authentication
// #[serde(skip)]
// pub hf_token: Option<String>,
// #[serde(skip)]
// pub hf_token_env_var: Option<String>,

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(derive(Debug, Clone), on(String, into), builder_type(name = GgufLoaderBuilder), finish_fn(vis = "", name = build_internal))]
pub struct GgufLoader {
    #[allow(dead_code)]
    #[serde(skip_serializing)]
    #[builder(default, setters(vis = "pub", name = has_model_source_internal), getter)]
    has_model_source: bool,

    #[builder(setters(vis = "", name = from_local_path_internal))]
    pub from_local_path: Option<PathBuf>,

    #[builder(setters(vis = "", name = hf_file_internal))]
    pub from_hf_file: Option<HfFile>,

    #[builder(setters(vis = "", name = smart_select_internal))]
    pub from_smart_select: Option<SmartLoader>,
}

impl<S: State> GgufLoaderBuilder<S> {
    pub fn from_local_path<V>(
        self,
        value: V,
    ) -> GgufLoaderBuilder<SetFromLocalPath<SetHasModelSource<S>>>
    where
        V: Into<PathBuf>,
        S::FromLocalPath: IsUnset,
        S::HasModelSource: IsUnset,
    {
        let path = value.into();
        self.has_model_source_internal(true)
            .from_local_path_internal(path)
    }

    pub fn from_hf_quant_file_url<V>(
        self,
        url: V,
    ) -> GgufLoaderBuilder<SetFromHfFile<SetHasModelSource<S>>>
    where
        V: AsRef<str>,
        S::FromHfFile: IsUnset,
        S::HasModelSource: IsUnset,
    {
        let hf_file = HfFile::from_url(url).expect("Failed to create HfFile from URL");
        self.has_model_source_internal(true)
            .hf_file_internal(hf_file)
    }

    pub fn from_smart_select(
        self,
        smart_select: SmartLoader,
    ) -> GgufLoaderBuilder<SetFromSmartSelect<SetHasModelSource<S>>>
    where
        S::FromSmartSelect: IsUnset,
        S::HasModelSource: IsUnset,
    {
        self.has_model_source_internal(true)
            .smart_select_internal(smart_select)
    }
}

impl<State: gguf_loader_builder::State> GgufLoaderBuilder<State>
where
    State: IsComplete,
    State::HasModelSource: IsSet,
{
    pub fn build(self) -> GgufLoader {
        let args = self.build_internal();
        args
    }
}
