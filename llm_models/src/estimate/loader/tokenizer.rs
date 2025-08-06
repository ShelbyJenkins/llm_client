use std::path::Path;

use bon::Builder;
use serde::{Deserialize, Serialize};
use tokenizer_loader_builder::{
    IsComplete, IsSet, IsUnset, SetFromHfFile, SetFromLocalPath, State,
};
use tokenizers::Tokenizer;

use crate::{
    estimate::loader::tokenizer::tokenizer_loader_builder::SetHasTokenizerSource,
    fs::{error::FileSystemError, file_path::ExistingFile},
    hf::id::{HfFile, HfFileError, HfRFilename, HfRepoId},
};

#[derive(Builder, Clone, Debug, Serialize, Deserialize, Default)]
#[builder(finish_fn(vis = "", name = build_internal))]
pub struct TokenizerLoader {
    #[allow(dead_code)]
    #[serde(skip_serializing)]
    #[builder(default, setters(vis = "", name = has_tokenizer_source_internal))]
    has_tokenizer_source: bool,

    #[builder(setters(vis = "", name = from_local_path_internal))]
    pub from_local_path: Option<ExistingFile>,

    #[builder(setters(vis = "", name = hf_file_internal))]
    pub from_hf_file: Option<HfFile>,
}

impl<S: State> TokenizerLoaderBuilder<S> {
    pub fn from_local_path<V>(
        self,
        value: V,
    ) -> Result<TokenizerLoaderBuilder<SetFromLocalPath<SetHasTokenizerSource<S>>>, FileSystemError>
    where
        V: AsRef<Path>,
        S::FromLocalPath: IsUnset,
        S::HasTokenizerSource: IsUnset,
    {
        Ok(self
            .has_tokenizer_source_internal(true)
            .from_local_path_internal(ExistingFile::try_new(value)?))
    }

    pub fn from_hf_file_url<V>(
        self,
        url: V,
    ) -> Result<TokenizerLoaderBuilder<SetFromHfFile<SetHasTokenizerSource<S>>>, HfFileError>
    where
        V: AsRef<str>,
        S::FromHfFile: IsUnset,
        S::HasTokenizerSource: IsUnset,
    {
        let hf_file = HfFile::try_from_url(url)?;
        hf_file.is_extension("json")?;
        Ok(self
            .has_tokenizer_source_internal(true)
            .hf_file_internal(hf_file))
    }

    pub fn from_hf_repo_id<V, O>(
        self,
        repo_id: V,
        sha: Option<O>,
    ) -> Result<TokenizerLoaderBuilder<SetFromHfFile<SetHasTokenizerSource<S>>>, HfFileError>
    where
        V: AsRef<str>,
        O: AsRef<str>,
        S::FromHfFile: IsUnset,
        S::HasTokenizerSource: IsUnset,
    {
        let hf_repo = HfRepoId::try_from_repo_id(repo_id, sha)?;
        let hf_file =
            HfFile::try_from_repo_id_and_file(&hf_repo, HfRFilename::try_new("tokenizer.json")?)?;
        Ok(self
            .has_tokenizer_source_internal(true)
            .hf_file_internal(hf_file))
    }
}

impl<State: tokenizer_loader_builder::State> TokenizerLoaderBuilder<State>
where
    State: IsComplete,
    State::HasTokenizerSource: IsSet,
{
    pub fn build(self) -> TokenizerLoader {
        let args = self.build_internal();
        args
    }
}

impl TokenizerLoader {
    pub fn load(&self) -> Option<Tokenizer> {
        if !self.has_tokenizer_source {
            return None;
        }
        todo!("Implement loading tokenizer from local path or HF file");
    }
}
