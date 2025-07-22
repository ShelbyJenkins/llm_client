use super::*;

#[derive(Debug, serde::Deserialize)]
pub(super) struct DeCloudLlms(Vec<DeCloudLlm>);

impl DeCloudLlms {
    pub(super) fn into_macro_models(self, provider: &MacroCloudLlmProvider) -> MacroCloudLlms {
        let mut models: Vec<MacroCloudLlm> = Vec::new();
        for de_model in self.0 {
            models.push(MacroCloudLlm {
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
        MacroCloudLlms(models)
    }
}

#[derive(Debug, serde::Deserialize)]
struct DeCloudLlm {
    model_id: String,
    friendly_name: String,
    model_ctx_size: u64,
    inference_ctx_size: u64,
    cost_per_m_in_tokens: u64,
    cost_per_m_out_tokens: u64,
    tokens_per_message: u64,
    tokens_per_name: Option<i64>,
}

#[derive(Debug)]
pub(super) struct MacroCloudLlms(pub Vec<MacroCloudLlm>);

#[derive(Debug)]
pub(super) struct MacroCloudLlm {
    pub(super) provider: MacroCloudLlmProvider,
    pub(super) model_id: String,
    pub(super) friendly_name: String,
    pub(super) model_ctx_size: u64,
    pub(super) inference_ctx_size: u64,
    pub(super) cost_per_m_in_tokens: u64,
    pub(super) cost_per_m_out_tokens: u64,
    pub(super) tokens_per_message: u64,
    pub(super) tokens_per_name: Option<i64>,
}

impl MacroCloudLlm {
    pub(super) fn const_ident(&self) -> Ident {
        format_ident!(
            "{}",
            self.friendly_name
                .to_uppercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        )
    }

    pub(super) fn enum_ident(&self) -> Ident {
        format_ident!("{}", to_enum(&self.model_id))
    }

    pub(super) fn fn_name_ident(&self) -> Ident {
        format_ident!("{}", to_func(&self.model_id))
    }

    pub(super) fn model_enum_variant(&self) -> TokenStream {
        let variant_name = format_ident!("{}", self.provider.enum_variant());
        let enum_name = self.enum_ident();
        quote! { #variant_name(#enum_name) }
    }

    pub fn to_token_stream(self) -> TokenStream {
        let model_enum_variant = self.model_enum_variant();
        let const_ident = self.const_ident();
        let model_id = self.model_id;
        let friendly_name = self.friendly_name;
        let model_ctx_size = self.model_ctx_size;
        let inference_ctx_size = self.inference_ctx_size;
        let cost_per_m_in_tokens = self.cost_per_m_in_tokens;
        let cost_per_m_out_tokens = self.cost_per_m_out_tokens;
        let tokens_per_message = self.tokens_per_message;
        let tokens_per_name = match self.tokens_per_name {
            Some(i64) => {
                quote! { Some(#i64) }
            }
            None => {
                quote! { None }
            }
        };

        quote! {
            pub const #const_ident: CloudLlm = CloudLlm {
                provider_llm_id: CloudProviderLlmId::#model_enum_variant,
                model_base: LlmModelBase {
                    model_id: Cow::Borrowed(#model_id),
                    friendly_name: Cow::Borrowed(#friendly_name),
                    model_ctx_size: #model_ctx_size,
                    inference_ctx_size: #inference_ctx_size,
                },
                cost_per_m_in_tokens: #cost_per_m_in_tokens,
                cost_per_m_out_tokens: #cost_per_m_out_tokens,
                tokens_per_message: #tokens_per_message,
                tokens_per_name: #tokens_per_name,
            };
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let all_providers = MacroCloudLlmProvider::all();

    let mut model_associated_consts = Vec::new();
    let mut model_associated_consts_idents = Vec::new();

    for provider in all_providers {
        for model in provider.models().0 {
            model_associated_consts_idents.push(model.const_ident());
            model_associated_consts.push(model.to_token_stream());
        }
    }
    let model_associated_consts_len = model_associated_consts.len();
    let code = quote! {
        use super::*;

        impl CloudLlm {

            pub const ALL_MODELS: [CloudLlm; #model_associated_consts_len] = [ #(Self::#model_associated_consts_idents),* ];

            #(#model_associated_consts)*
        }
    };

    format_and_write(&output_path.join("models.rs"), code);
}
