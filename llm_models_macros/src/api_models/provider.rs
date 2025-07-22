use super::*;

#[derive(Debug, Clone)]
pub(super) enum MacroCloudLlmProvider {
    Anthropic,
    OpenAi,
    Perplexity,
    MistralAi,
}

impl MacroCloudLlmProvider {
    pub(super) fn all() -> Vec<Self> {
        vec![
            MacroCloudLlmProvider::Anthropic,
            MacroCloudLlmProvider::OpenAi,
            MacroCloudLlmProvider::Perplexity,
            MacroCloudLlmProvider::MistralAi,
        ]
    }

    pub(super) fn models(&self) -> MacroCloudLlms {
        let contents = match self {
            MacroCloudLlmProvider::Anthropic => anthropic::MODELS_JSON,
            MacroCloudLlmProvider::OpenAi => openai::MODELS_JSON,
            MacroCloudLlmProvider::Perplexity => perplexity::MODELS_JSON,
            MacroCloudLlmProvider::MistralAi => mistral::MODELS_JSON,
        };
        let models: DeCloudLlms = parse("models.json", contents).unwrap();
        models.into_macro_models(self)
    }

    pub(super) fn friendly_name(&self) -> &str {
        match self {
            MacroCloudLlmProvider::Anthropic => anthropic::FRIENDLY_NAME,
            MacroCloudLlmProvider::OpenAi => openai::FRIENDLY_NAME,
            MacroCloudLlmProvider::Perplexity => perplexity::FRIENDLY_NAME,
            MacroCloudLlmProvider::MistralAi => mistral::FRIENDLY_NAME,
        }
    }

    pub(super) fn default_model(&self) -> MacroCloudLlm {
        let model_id = match self {
            MacroCloudLlmProvider::Anthropic => anthropic::DEFAULT_MODEL,
            MacroCloudLlmProvider::OpenAi => openai::DEFAULT_MODEL,
            MacroCloudLlmProvider::Perplexity => perplexity::DEFAULT_MODEL,
            MacroCloudLlmProvider::MistralAi => mistral::DEFAULT_MODEL,
        };
        for model in self.models().0 {
            if model.model_id == model_id {
                return model;
            }
        }
        panic!("Default model not found for provider {:?}", self);
    }

    pub(super) fn enum_variant(&self) -> String {
        match self {
            MacroCloudLlmProvider::Anthropic => anthropic::ENUM_VARIANT,
            MacroCloudLlmProvider::OpenAi => openai::ENUM_VARIANT,
            MacroCloudLlmProvider::Perplexity => perplexity::ENUM_VARIANT,
            MacroCloudLlmProvider::MistralAi => mistral::ENUM_VARIANT,
        }
        .into()
    }

    pub(super) fn provider_enum_variant(&self) -> TokenStream {
        let variant_name = format_ident!("{}", self.enum_variant());
        let provider_enum_name = self.provider_enum_name();
        quote! { #variant_name(#provider_enum_name) }
    }

    pub(super) fn provider_enum_name(&self) -> Ident {
        format_ident!("{}LlmId", self.enum_variant())
    }

    fn enum_token_stream(&self) -> TokenStream {
        let enum_name = self.provider_enum_name();
        let mut provider_models_arms = Vec::new();
        let mut provider_model_ids_arms = Vec::new();
        let mut provider_models_const_idents = Vec::new();
        let mut model_enum_idents = Vec::new();
        let provider_friendly_name = self.friendly_name();
        let provider_default_model_const_ident = self.default_model().const_ident();

        for model in self.models().0 {
            let model_enum_ident = model.enum_ident();
            let model_const_ident = model.const_ident();
            let model_ident = model.model_id;
            provider_models_arms.push(quote! {
                Self::#model_enum_ident => CloudLlm::#model_const_ident
            });
            provider_model_ids_arms.push(quote! {
                Self::#model_enum_ident => #model_ident
            });
            model_enum_idents.push(model_enum_ident);

            provider_models_const_idents.push(model_const_ident);
        }

        quote! {
            #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
            pub enum #enum_name {
                #(#model_enum_idents),*
            }

            impl #enum_name {
                pub fn model(&self) -> CloudLlm {
                    match self {
                        #(#provider_models_arms),*
                    }
                }

                pub fn model_id(&self) -> &'static str {
                    match self {
                        #(#provider_model_ids_arms),*
                    }
                }

                pub fn all_provider_models() -> Vec<CloudLlm> {
                    vec![#(CloudLlm::#provider_models_const_idents),*]
                }

                pub fn provider_friendly_name(&self) -> &'static str {
                    #provider_friendly_name
                }

                pub fn default_model() -> CloudLlm {
                    CloudLlm::#provider_default_model_const_ident
                }
            }
        }
    }

    fn trait_token_stream(&self) -> TokenStream {
        let mut trait_setter_fns = Vec::new();
        let provider_enum_variant = self.enum_variant().to_owned();

        let trait_name = format_ident!("{provider_enum_variant}ModelTrait");

        for model in self.models().0 {
            let model_const_ident = model.const_ident();
            let fn_name_ident = model.fn_name_ident();
            trait_setter_fns.push(quote! {
                fn #fn_name_ident(mut self) -> crate::Result<Self>
                where
                Self: Sized,
                {
                    *self.model() = CloudLlm::#model_const_ident;
                    Ok(self)
                }

            });
        }

        quote! {
            pub trait #trait_name {
                fn model(&mut self) -> &mut CloudLlm;

                fn model_id_str(mut self, model_id: &str) -> crate::Result<Self>
                where
                    Self: Sized,
                {
                    *self.model() = CloudLlm::model_from_model_id(model_id)?;
                    Ok(self)
                }

                #(#trait_setter_fns)*
            }
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let all = MacroCloudLlmProvider::all();
    let mut provider_enum_variants = Vec::new();
    let mut provider_enums = Vec::new();
    let mut all_provider_models_arms = Vec::new();
    let mut provider_model_ids_arms = Vec::new();
    let mut provider_enum_names = Vec::new();
    let mut provider_friendly_name_arms = Vec::new();
    let mut model_arms = Vec::new();
    let mut provider_trait_impls = Vec::new();

    for provider in all {
        provider_enum_variants.push(provider.provider_enum_variant());
        provider_enums.push(provider.enum_token_stream());
        let enum_variant_ident = format_ident!("{}", provider.enum_variant());
        let provider_enum_name = provider.provider_enum_name();
        provider_trait_impls.push(provider.trait_token_stream());

        model_arms.push(quote! {
            Self::#enum_variant_ident(p) => p.model()
        });

        provider_friendly_name_arms.push(quote! {
            Self::#enum_variant_ident(p) => p.provider_friendly_name()
        });

        provider_model_ids_arms.push(quote! {
            Self::#enum_variant_ident(p) => p.model_id()
        });

        let mut provider_presets_const_idents = Vec::new();

        for model in provider.models().0 {
            provider_presets_const_idents.push(model.const_ident());
        }

        all_provider_models_arms.push(quote! {
            Self::#enum_variant_ident(_) => #provider_enum_name::all_provider_models()
        });
        provider_enum_names.push(provider_enum_name);
    }

    let code = quote! {
        use super::*;
        impl CloudLlm {
            pub fn model_from_model_id(model_id: &str) -> crate::Result<CloudLlm> {
                let model_id = model_id.to_lowercase();
                let models = Self::ALL_MODELS;
                for model in &models {
                if model.model_base.model_id.to_lowercase() == model_id {
                        return Ok((*model).to_owned());
                    }
                }
                for model in models {
                    if model_id.contains(&model.model_base.model_id.to_lowercase())
                        || model.model_base.model_id.contains(&model_id.to_lowercase())
                        || model
                            .model_base
                            .friendly_name
                            .to_lowercase()
                            .contains(&model_id)
                        {
                            return Ok(model);
                        }
                    }
                    crate::bail!("Model ID '{}' not found", model_id)
                }
        }

        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
        pub enum CloudProviderLlmId {
            #(#provider_enum_variants),*
        }
        impl CloudProviderLlmId {
            pub fn model(&self) -> CloudLlm {
                match self {
                    #(#model_arms),*
                }
            }

            pub fn model_id(&self) -> &'static str {
                match self {
                    #(#provider_model_ids_arms),*
                }
            }

            pub fn provider_friendly_name(&self) -> &'static str {
                match self {
                    #(#provider_friendly_name_arms),*
                }
            }
            pub fn all_models() -> Vec<CloudLlm> {
                [
                    #(#provider_enum_names::all_provider_models()),*
                ]
                .into_iter()
                .flatten()
                .collect()
            }

            pub fn all_provider_models(&self) -> Vec<CloudLlm> {
                match self {
                    #(#all_provider_models_arms),*
                }
            }
        }

        #(#provider_enums)*


        #(#provider_trait_impls)*
    };

    format_and_write(&output_path.join("providers.rs"), code);
}
