use super::{
    quant::{DeQuantFileNames, MacroQuantFile},
    *,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeGgufPreset {
    pub model_id: String,
    pub friendly_name: String,
    pub gguf_repo_id: String,
    pub model_repo_id: String,
    pub number_of_parameters: f64,
    pub quant_file_names: DeQuantFileNames,
    pub tokenizer_path: Option<std::path::PathBuf>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
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
    pub vocab_size: u64,
    #[serde(alias = "model_type")]
    pub architecture: String,
    pub model_size_bytes: Option<u64>,
}

impl DeConfigJson {
    fn load(preset_path: &std::path::Path, model_repo_id: &str) -> DeConfigJson {
        match open_and_parse::<DeConfigJson>("config.json", &preset_path.join("config.json")) {
            Ok(config) => config,
            Err(_) => {
                let config_path = load_file("config.json", model_repo_id);
                open_and_parse("config.json", &config_path).unwrap()
            }
        }
    }

    fn to_token_stream(&self) -> TokenStream {
        let context_length = self.context_length;
        let embedding_length = self.embedding_length;
        let feed_forward_length = match self.feed_forward_length {
            Some(value) => quote! { Some(#value) },
            None => quote! { None },
        };
        let head_count = self.head_count;
        let head_count_kv = match self.head_count_kv {
            Some(value) => quote! { Some(#value) },
            None => quote! { None },
        };
        let block_count = self.block_count;
        let torch_dtype = &self.torch_dtype;
        let vocab_size = self.vocab_size;
        let architecture = &self.architecture;
        let model_size_bytes = match self.model_size_bytes {
            Some(value) => quote! { Some(#value) },
            None => quote! { None },
        };

        quote! {
            GgufPresetConfig {
                context_length: #context_length,
                embedding_length: #embedding_length,
                feed_forward_length: #feed_forward_length,
                head_count: #head_count,
                head_count_kv: #head_count_kv,
                block_count: #block_count,
                torch_dtype: Cow::Borrowed(#torch_dtype),
                vocab_size: #vocab_size,
                architecture: Cow::Borrowed(#architecture),
                model_size_bytes: #model_size_bytes,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacroGgufPreset {
    pub organization: MacroPresetOrganization,
    pub model_id: String,
    pub friendly_name: String,
    pub gguf_repo_id: String,
    pub model_repo_id: String,
    pub number_of_parameters: f64,
    pub input_tokenizer_path: Option<std::path::PathBuf>, // The tokenizer file to move to the output directory
    pub output_tokenizer_file_name: Option<String>,       // The file name after renaming
    pub config: DeConfigJson,
    pub quants: Vec<MacroQuantFile>,
}

impl MacroGgufPreset {
    pub fn new(organization: MacroPresetOrganization, preset_path: &std::path::PathBuf) -> Self {
        let preset: DeGgufPreset = open_and_parse(
            "model_macro_data.json",
            &preset_path.join("model_macro_data.json"),
        )
        .unwrap();
        let quants = MacroQuantFile::get_quants(&preset);

        let (input_tokenizer_path, output_tokenizer_file_name) =
            if let Some(tokenizer_path) = &preset.tokenizer_path {
                let input = PATH_TO_ORGS_DATA_DIR.join(tokenizer_path);
                open_file("tokenizer.json", &input).unwrap();
                let output = tokenizer_path
                    .to_str()
                    .unwrap()
                    .to_lowercase()
                    .replace(|c: char| !c.is_alphanumeric(), ".");
                (Some(input), Some(output))
            } else {
                (None, None)
            };

        let config = DeConfigJson::load(preset_path, &preset.model_repo_id);
        Self {
            organization,
            model_id: preset.model_id,
            friendly_name: preset.friendly_name,
            model_repo_id: preset.model_repo_id,
            gguf_repo_id: preset.gguf_repo_id,
            number_of_parameters: preset.number_of_parameters,
            input_tokenizer_path,
            output_tokenizer_file_name,
            config,
            quants,
        }
    }

    pub fn const_ident(&self) -> Ident {
        format_ident!(
            "{}",
            self.friendly_name
                .to_uppercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        )
    }

    pub fn enum_ident(&self) -> Ident {
        format_ident!("{}", to_enum(&self.model_id))
    }

    pub fn fn_name_ident(&self) -> Ident {
        format_ident!("{}", to_func(&self.model_id))
    }

    pub fn to_token_stream(self) -> TokenStream {
        let const_ident = self.const_ident();
        let enum_variant_ident = self.enum_ident();
        let organization_const_ident = self.organization.const_ident();
        let model_id = self.model_id;
        let friendly_name = self.friendly_name;
        let model_repo_id = self.model_repo_id;
        let gguf_repo_id = self.gguf_repo_id;
        let number_of_parameters = self.number_of_parameters;
        let model_ctx_size = self.config.context_length;

        let tokenizer_path_ts = match &self.output_tokenizer_file_name {
            Some(fname) => {
                quote! { Some(Cow::Borrowed(#fname)) }
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
                model_base: LlmModelBase {
                    model_id: Cow::Borrowed(#model_id),
                    friendly_name: Cow::Borrowed(#friendly_name),
                    model_ctx_size: #model_ctx_size,
                    inference_ctx_size: #model_ctx_size,
                },
                model_repo_id: Cow::Borrowed(#model_repo_id),
                gguf_repo_id: Cow::Borrowed(#gguf_repo_id),
                number_of_parameters: #number_of_parameters,
                tokenizer_file_name: #tokenizer_path_ts,
                config: #config_ts,
                quants: Cow::Borrowed(#quants_ts),
                preset_llm_id: GgufPresetId::#enum_variant_ident,
            };
        }
    }
}

pub fn generate(output_path: &std::path::PathBuf) {
    let organizations = MacroPresetOrganizations::new();

    let mut preset_associated_consts = Vec::new();
    let mut preset_associated_consts_idents = Vec::new();
    let mut preset_ids_enum_variants = Vec::new();
    let mut model_arms = Vec::new();
    let mut model_id_arms = Vec::new();
    let mut trait_setter_fns = Vec::new();

    for org in organizations.0 {
        for model in org.load_models() {
            let model_const_ident = model.const_ident();
            let enum_variant_ident = model.enum_ident();
            let fn_name_ident = model.fn_name_ident();
            let model_id = model.model_id.clone();
            trait_setter_fns.push(quote! {
                    fn #fn_name_ident(mut self) -> Self
                    where
                    Self: Sized,
                    {
                        *self.preset() = GgufPreset::#model_const_ident;
                        self
                    }
            });

            model_arms.push(quote! {
                Self::#enum_variant_ident => GgufPreset::#model_const_ident
            });

            model_id_arms.push(quote! {
                Self::#enum_variant_ident => #model_id
            });

            preset_associated_consts_idents.push(model_const_ident);
            preset_ids_enum_variants.push(enum_variant_ident);
            preset_associated_consts.push(model.to_token_stream());
        }
    }

    let preset_associated_consts_len = preset_associated_consts.len();
    let code = quote! {
        use super::*;

        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
        pub enum GgufPresetId {
            #(#preset_ids_enum_variants),*
        }

        impl GgufPresetId {
            pub fn preset(&self) -> GgufPreset {
                match self {
                    #(#model_arms),*
                }
            }
            pub fn model_id(&self) -> &'static str {
                match self {
                    #(#model_id_arms),*
                }
            }
        }

        impl GgufPreset {
            pub const ALL_MODELS: [GgufPreset; #preset_associated_consts_len] = [ #(Self::#preset_associated_consts_idents),* ];

            #(#preset_associated_consts)*
        }

        pub trait GgufPresetTrait {
            fn preset(&mut self) -> &mut GgufPreset;

            fn preset_from_str(mut self, selected_model_id: &str) -> crate::Result<Self>
            where
                Self: Sized,
            {
                let preset = GgufPreset::ALL_MODELS
                .into_iter()
                .find(|preset| preset.model_base.model_id == selected_model_id)
                .ok_or_else(|| crate::anyhow!("Invalid selected_model_id: {}", selected_model_id))?;
                *self.preset() = preset;
                Ok(self)
            }

            #(#trait_setter_fns)*
        }


    };

    format_and_write(&output_path.join("preset_models.rs"), code);
}
