use super::*;

#[derive(Debug, serde::Deserialize)]
pub(super) struct DeApiLlmPresets(Vec<DeApiLlmPreset>);

impl DeApiLlmPresets {
    pub(super) fn into_macro_models(self, provider: MacroApiLlmProvider) -> Vec<MacroApiLlmPreset> {
        let mut models: Vec<MacroApiLlmPreset> = Vec::new();
        for de_model in self.0 {
            models.push(MacroApiLlmPreset {
                provider: provider.clone(),
                model_id: de_model.model_id,
                friendly_name: de_model.friendly_name,
                model_ctx_size: de_model.model_ctx_size,
                inference_ctx_size: de_model.inference_ctx_size,
                cost_per_m_in_tokens: de_model.cost_per_m_in_tokens,
                cost_per_m_out_tokens: de_model.cost_per_m_out_tokens,
                tokens_per_message: de_model.tokens_per_message,
                tokens_per_name: de_model.tokens_per_name,
            });
        }
        models
    }
}

#[derive(Debug, serde::Deserialize)]
struct DeApiLlmPreset {
    model_id: String,
    friendly_name: String,
    model_ctx_size: usize,
    inference_ctx_size: usize,
    cost_per_m_in_tokens: usize,
    cost_per_m_out_tokens: usize,
    tokens_per_message: usize,
    tokens_per_name: Option<isize>,
}

#[derive(Debug)]
pub(super) struct MacroApiLlmPreset {
    pub(super) provider: MacroApiLlmProvider,
    pub(super) model_id: String,
    pub(super) friendly_name: String,
    pub(super) model_ctx_size: usize,
    pub(super) inference_ctx_size: usize,
    pub(super) cost_per_m_in_tokens: usize,
    pub(super) cost_per_m_out_tokens: usize,
    pub(super) tokens_per_message: usize,
    pub(super) tokens_per_name: Option<isize>,
}

impl MacroApiLlmPreset {
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
        let provider_variant_ident = format_ident!("{}", self.provider.enum_variant());
        let model_id = self.model_id;
        let friendly_name = self.friendly_name;
        let model_ctx_size = self.model_ctx_size;
        let inference_ctx_size = self.inference_ctx_size;
        let cost_per_m_in_tokens = self.cost_per_m_in_tokens;
        let cost_per_m_out_tokens = self.cost_per_m_out_tokens;
        let tokens_per_message = self.tokens_per_message;
        let tokens_per_name = match self.tokens_per_name {
            Some(isize) => {
                quote! { Some(#isize) }
            }
            None => {
                quote! { None }
            }
        };

        quote! {
            pub const #const_ident: ApiLlmPreset = ApiLlmPreset {
                provider: ApiLlmProvider::#provider_variant_ident,
                model_id: #model_id,
                friendly_name: #friendly_name,
                model_ctx_size: #model_ctx_size,
                inference_ctx_size: #inference_ctx_size,
                cost_per_m_in_tokens: #cost_per_m_in_tokens,
                cost_per_m_out_tokens: #cost_per_m_out_tokens,
                tokens_per_message: #tokens_per_message,
                tokens_per_name: #tokens_per_name,
            };
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let all_providers = MacroApiLlmProvider::all();

    let mut model_associated_consts = Vec::new();
    let mut model_associated_consts_idents = Vec::new();

    for provider in all_providers {
        for model in provider.models() {
            model_associated_consts_idents.push(model.const_ident());
            model_associated_consts.push(model.to_token_stream());
        }
    }

    let code = quote! {
        use super::*;

        #[derive(Debug, Clone)]
        pub struct ApiLlmPreset {
            pub provider: ApiLlmProvider,
            pub model_id: &'static str,
            pub friendly_name: &'static str,
            pub model_ctx_size: usize,
            pub inference_ctx_size: usize,
            pub cost_per_m_in_tokens: usize,
            pub cost_per_m_out_tokens: usize,
            pub tokens_per_message: usize,
            pub tokens_per_name: Option<isize>,
        }

        impl ApiLlmModel {
            pub fn model_from_model_id(&self, model_id: &str) -> Result<ApiLlmModel, crate::Error> {
                let providers = ApiLlmProvider::all_providers();
                for provider in providers {
                    if let Ok(preset) = provider.preset_from_model_id(model_id) {
                        return Ok(Self::from_preset(preset));
                    }
                }
                crate::bail!("Model not found for model_id: {}", model_id);
            }
        }

        impl ApiLlmPreset {
            pub fn all_models() -> Vec<Self> {
                vec![#(Self::#model_associated_consts_idents),*]
            }

            #(#model_associated_consts)*
        }
    };

    format_and_write(&output_path.join("models.rs"), code);
}
