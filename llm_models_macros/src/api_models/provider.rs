use super::*;

#[derive(Debug, Clone)]
pub(super) enum MacroApiLlmProvider {
    Anthropic,
    OpenAi,
    Perplexity,
}

impl MacroApiLlmProvider {
    pub(super) fn all() -> Vec<Self> {
        vec![
            MacroApiLlmProvider::Anthropic,
            MacroApiLlmProvider::OpenAi,
            MacroApiLlmProvider::Perplexity,
        ]
    }

    pub(super) fn models(self) -> Vec<MacroApiLlmPreset> {
        let contents = match self {
            MacroApiLlmProvider::Anthropic => anthropic::DEFAULT_MODELS,
            MacroApiLlmProvider::OpenAi => openai::DEFAULT_MODELS,
            MacroApiLlmProvider::Perplexity => perplexity::DEFAULT_MODELS,
        };
        let models: DeApiLlmPresets = parse("models.json", contents);
        models.into_macro_models(self)
    }

    pub(super) fn friendly_name(&self) -> &str {
        match self {
            MacroApiLlmProvider::Anthropic => anthropic::FRIENDLY_NAME,
            MacroApiLlmProvider::OpenAi => openai::FRIENDLY_NAME,
            MacroApiLlmProvider::Perplexity => perplexity::FRIENDLY_NAME,
        }
    }

    pub(super) fn enum_variant(&self) -> &str {
        match self {
            MacroApiLlmProvider::Anthropic => anthropic::ENUM_VARIANT,
            MacroApiLlmProvider::OpenAi => openai::ENUM_VARIANT,
            MacroApiLlmProvider::Perplexity => perplexity::ENUM_VARIANT,
        }
    }

    fn variant_ident(&self) -> Ident {
        format_ident!("{}", self.enum_variant())
    }

    fn to_token_stream(self) -> TokenStream {
        let mut trait_setter_fns = Vec::new();

        let mut fn_name = self.enum_variant().to_owned();
        fn_name.get_mut(0..1).unwrap().make_ascii_uppercase();
        let trait_name = format_ident!("{fn_name}ModelTrait");

        let variant_ident = self.variant_ident();

        for model in self.models() {
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
                    *self.model() = ApiLlmModel::from_preset(ApiLlmPreset::#model_const_ident);
                    self
                }

            });
        }

        quote! {
            pub trait #trait_name {
                fn model(&mut self) -> &mut ApiLlmModel;

                fn model_id_str(mut self, model_id: &str) -> Result<Self, crate::Error>
                where
                    Self: Sized,
                {
                    *self.model() = ApiLlmModel::from_preset(ApiLlmProvider::#variant_ident.preset_from_model_id(model_id)?);
                    Ok(self)
                }

                #(#trait_setter_fns)*
            }
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let all = MacroApiLlmProvider::all();
    let mut variant_idents = Vec::new();
    let mut friendly_name_arms = Vec::new();
    let mut provider_presets_arms = Vec::new();
    let mut provider_trait_impls = Vec::new();

    for provider in all {
        let ident = format_ident!("{}", provider.enum_variant());

        let friendly_name = provider.friendly_name();
        friendly_name_arms.push(quote! {
            Self::#ident => #friendly_name
        });

        let mut provider_presets_const_idents = Vec::new();

        provider_trait_impls.push(provider.clone().to_token_stream());

        for model in provider.models() {
            provider_presets_const_idents.push(model.const_ident());
        }

        provider_presets_arms.push(quote! {
            Self::#ident => vec![#(ApiLlmPreset::#provider_presets_const_idents),*]
        });

        variant_idents.push(ident);
    }

    let code = quote! {
        use super::*;

        #[derive(Debug, Clone)]
        pub enum ApiLlmProvider {
            #(#variant_idents),*
        }

        impl ApiLlmProvider {
            pub fn all_providers() -> Vec<Self> {
                vec![#(Self::#variant_idents),*]
            }

            pub fn all_provider_presets(&self) -> Vec<ApiLlmPreset> {
                match self {
                    #(#provider_presets_arms),*
                }
            }

            pub fn preset_from_model_id(&self, model_id: &str) -> Result<ApiLlmPreset, crate::Error> {
                let model_id = model_id.to_lowercase();
                let presets = self.all_provider_presets();
                for preset in &presets {
                    if preset.model_id.to_lowercase() == model_id {
                        return Ok(preset.to_owned());
                    }
                }
                for preset in presets {
                    if model_id.contains(&preset.model_id.to_lowercase())
                        || preset.model_id.contains(&model_id.to_lowercase())
                        || preset.friendly_name.to_lowercase().contains(&model_id)
                    {
                        return Ok(preset);
                    }
                }
                crate::bail!("Model ID '{}' not found", model_id)
            }

            pub fn friendly_name(&self) -> &str {
                match self {
                    #(#friendly_name_arms),*
                }
            }
        }

        #(#provider_trait_impls)*
    };

    format_and_write(&output_path.join("providers.rs"), code);
}
