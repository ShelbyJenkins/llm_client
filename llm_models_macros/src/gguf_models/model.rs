use super::{
    quant::{DeQuantFileNames, MacroQuantFile},
    *,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub(super) struct DeGgufPreset {
    pub model_id: String,
    pub friendly_name: String,
    pub gguf_repo_id: String,
    pub model_repo_id: String,
    pub number_of_parameters: f64,
    pub quant_file_names: DeQuantFileNames,
    pub tokenizer_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeConfigJson {
    #[serde(alias = "max_position_embeddings")]
    #[serde(alias = "n_ctx")]
    pub context_length: u64,
    #[serde(alias = "hidden_size")]
    #[serde(alias = "n_embd")]
    pub embedding_length: u64,
    #[serde(alias = "intermediate_size")]
    #[serde(alias = "n_inner")]
    pub feed_forward_length: Option<u64>,
    #[serde(alias = "num_attention_heads")]
    #[serde(alias = "n_head")]
    pub head_count: u64,
    #[serde(alias = "num_key_value_heads")]
    pub head_count_kv: Option<u64>,
    #[serde(alias = "num_hidden_layers")]
    #[serde(alias = "n_layers")]
    #[serde(alias = "n_layer")]
    #[serde(alias = "num_layers")]
    pub block_count: u64,
    pub torch_dtype: String,
    pub vocab_size: u32,
    #[serde(alias = "model_type")]
    pub architecture: String,
    pub model_size_bytes: Option<u64>,
}

impl DeConfigJson {
    fn to_token_stream(&self) -> TokenStream {
        let context_length = self.context_length as usize;
        let embedding_length = self.embedding_length as usize;
        let feed_forward_length = match self.feed_forward_length {
            Some(value) => quote! { Some(#value  as usize) },
            None => quote! { None },
        };
        let head_count = self.head_count as usize;
        let head_count_kv = match self.head_count_kv {
            Some(value) => quote! { Some(#value  as usize) },
            None => quote! { None },
        };
        let block_count = self.block_count as usize;
        let torch_dtype = &self.torch_dtype;
        let vocab_size = self.vocab_size as usize;
        let architecture = &self.architecture;
        let model_size_bytes = match self.model_size_bytes {
            Some(value) => quote! { Some(#value  as usize) },
            None => quote! { None },
        };

        quote! {
            ConfigJson {
                context_length: #context_length,
                embedding_length: #embedding_length,
                feed_forward_length: #feed_forward_length,
                head_count: #head_count,
                head_count_kv: #head_count_kv,
                block_count: #block_count,
                torch_dtype: #torch_dtype,
                vocab_size: #vocab_size,
                architecture: #architecture,
                model_size_bytes: #model_size_bytes,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct MacroGgufPreset {
    pub organization: MacroPresetOrganization,
    pub model_id: String,
    pub friendly_name: String,
    pub gguf_repo_id: String,
    pub model_repo_id: String,
    pub number_of_parameters: f64,
    pub input_tokenizer_path: Option<std::path::PathBuf>, // The tokenizer file to move to the output directory
    pub output_tokenizer_path: Option<std::path::PathBuf>, // The path to the tokenizer file relative to the output directory
    pub config: DeConfigJson,
    pub quants: Vec<MacroQuantFile>,
}

impl MacroGgufPreset {
    pub fn new(organization: MacroPresetOrganization, preset_path: &std::path::PathBuf) -> Self {
        let preset: DeGgufPreset = open_and_parse(
            "model_macro_data.json",
            &preset_path.join("model_macro_data.json"),
        );
        let quants = MacroQuantFile::get_quants(&preset);

        let (input_tokenizer_path, output_tokenizer_path) =
            if let Some(tokenizer_path) = &preset.tokenizer_path {
                let input = PATH_TO_ORGS_DATA_DIR.join(tokenizer_path);
                open_file("tokenizer.json", &input);
                let output = tokenizer_path
                    .to_str()
                    .unwrap()
                    .to_lowercase()
                    .replace(|c: char| !c.is_alphanumeric(), ".");
                (Some(input), Some(std::path::PathBuf::from(output)))
            } else {
                (None, None)
            };

        let config: DeConfigJson = open_and_parse("config.json", &preset_path.join("config.json"));

        Self {
            organization,
            model_id: preset.model_id,
            friendly_name: preset.friendly_name,
            model_repo_id: preset.model_repo_id,
            gguf_repo_id: preset.gguf_repo_id,
            number_of_parameters: preset.number_of_parameters,
            input_tokenizer_path,
            output_tokenizer_path,
            config,
            quants,
        }
    }

    pub(super) fn const_ident(&self) -> Ident {
        format_ident!(
            "{}",
            self.friendly_name
                .to_uppercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        )
    }

    pub fn to_token_stream(self) -> TokenStream {
        let const_ident = self.const_ident();
        let organization_const_ident = self.organization.const_ident();
        let model_id = self.model_id;
        let friendly_name = self.friendly_name;
        let model_repo_id = self.model_repo_id;
        let gguf_repo_id = self.gguf_repo_id;
        let number_of_parameters = self.number_of_parameters;
        let model_ctx_size = self.config.context_length as usize;

        let tokenizer_path_ts = match &self.output_tokenizer_path {
            Some(path_buf) => {
                let path_str = path_buf.to_string_lossy().to_string();

                quote! { Some(#path_str) }
            }
            None => {
                quote! { None }
            }
        };

        let quants_ts = MacroQuantFile::to_token_stream(&self.quants);
        let config_ts = self.config.to_token_stream();

        quote! {
            pub const #const_ident: GgufPreset = GgufPreset {
                organization: LocalLlmOrganization::#organization_const_ident,
                model_id: #model_id,
                friendly_name: #friendly_name,
                model_repo_id: #model_repo_id,
                gguf_repo_id: #gguf_repo_id,
                number_of_parameters: #number_of_parameters,
                model_ctx_size: #model_ctx_size,
                tokenizer_path: #tokenizer_path_ts,
                config: #config_ts,
                quants: #quants_ts,
            };
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let organizations = MacroPresetOrganizations::new();

    let mut preset_associated_consts = Vec::new();
    let mut preset_associated_consts_idents = Vec::new();

    let mut trait_setter_fns = Vec::new();
    for org in organizations.0 {
        for model in org.load_models() {
            let model_const_ident = model.const_ident();
            let fn_name_ident = format_ident!(
                "{}",
                model
                    .friendly_name
                    .to_lowercase()
                    .replace(|c: char| !c.is_alphanumeric(), "_")
            );

            trait_setter_fns.push(quote! {
                    fn #fn_name_ident(mut self) -> Self
                    where
                    Self: Sized,
                    {
                        self.preset_loader().llm_preset = GgufPreset::#model_const_ident;
                        self
                    }
            });

            preset_associated_consts_idents.push(model_const_ident);
            preset_associated_consts.push(model.to_token_stream());
        }
    }

    let code = quote! {
        use super::*;

        impl GgufPreset {
            pub fn all_models() -> Vec<Self> {
                vec![#(Self::#preset_associated_consts_idents),*]
            }

            #(#preset_associated_consts)*
        }

        pub trait GgufPresetTrait {
            fn preset_loader(&mut self) -> &mut GgufPresetLoader;

            fn preset_from_str(mut self, selected_model_id: &str) -> Result<Self, crate::Error>
            where
                Self: Sized,
            {
                let preset = GgufPreset::all_models()
                .into_iter()
                .find(|preset| preset.model_id == selected_model_id)
                .ok_or_else(|| crate::anyhow!("Invalid selected_model_id: {}", selected_model_id))?;

                self.preset_loader().llm_preset = preset;
                Ok(self)
            }

            fn preset_with_memory_gb(mut self, preset_with_memory_gb: usize) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().preset_with_memory_gb = Some(preset_with_memory_gb);
                self
            }


            fn preset_with_quantization_level(mut self, level: u8) -> Self
            where
                Self: Sized,
            {
                self.preset_loader().preset_with_quantization_level = Some(level);
                self
            }

            #(#trait_setter_fns)*
        }


    };

    format_and_write(&output_path.join("models.rs"), code);
}
