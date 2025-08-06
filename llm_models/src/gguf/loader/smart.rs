use bon::Builder;
use serde::{Deserialize, Serialize};
use smart_loader_builder::{
    IsComplete, IsSet, IsUnset, SetFromAvailableMemory, SetFromQuantizationLevel,
    SetHasSelectionMethod, SetHfRepo, State,
};

use crate::hf::types::HfRepo;

#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct SmartLoader {
    #[builder(setters(vis = "", name = hf_repo_internal))]
    pub hf_repo: HfRepo,

    #[allow(dead_code)]
    #[serde(skip_serializing)]
    #[builder(default, setters(vis = "pub", name = has_selection_method_internal), getter)]
    has_selection_method: bool,

    #[builder(setters(vis = "", name = from_available_memory_internal))]
    pub from_available_memory: Option<u64>,

    #[builder(setters(vis = "", name = from_quantization_level_internal))]
    pub from_quantization_level: Option<u8>,
}

impl<S: State> SmartLoaderBuilder<S> {
    pub fn from_available_memory<V>(
        self,
        value: V,
    ) -> SmartLoaderBuilder<SetFromAvailableMemory<SetHasSelectionMethod<S>>>
    where
        V: Into<u64>,
        S::FromAvailableMemory: IsUnset,
        S::HasSelectionMethod: IsUnset,
    {
        self.has_selection_method_internal(true)
            .from_available_memory_internal(value.into())
    }

    pub fn from_quantization_level<V>(
        self,
        value: V,
    ) -> SmartLoaderBuilder<SetFromQuantizationLevel<SetHasSelectionMethod<S>>>
    where
        V: Into<u8>,
        S::FromQuantizationLevel: IsUnset,
        S::HasSelectionMethod: IsUnset,
    {
        self.has_selection_method_internal(true)
            .from_quantization_level_internal(value.into())
    }

    pub fn hf_repo_url<V>(self, url: V) -> SmartLoaderBuilder<SetHfRepo<S>>
    where
        V: AsRef<str>,
        S::HfRepo: IsUnset,
    {
        let hf_repo = HfRepo::from_url(url).expect("Failed to create HfRepo from URL");
        self.hf_repo_internal(hf_repo)
    }

    pub fn hf_repo_id<V>(self, repo_id: V) -> SmartLoaderBuilder<SetHfRepo<S>>
    where
        V: AsRef<str>,
        S::HfRepo: IsUnset,
    {
        let hf_repo = HfRepo::from_repo_id(repo_id).expect("Failed to create HfRepo from ID");

        self.hf_repo_internal(hf_repo)
    }
}

impl<State: smart_loader_builder::State> SmartLoaderBuilder<State>
where
    State: IsComplete,
    State::HasSelectionMethod: IsSet,
{
    pub fn build(self) -> SmartLoader {
        self.build_internal()
    }

    // pub fn load(self) -> GgufModel {
    //     self.build_internal().load()
    // }
}

// impl SmartLoader {
//     pub fn load(&self) -> GgufModel {
//         let local_model_path = std::path::PathBuf::from("local_model.gguf");
//         LocalLoader::builder()
//             .local_model_path(local_model_path)
//             .load()
//     }
// }
