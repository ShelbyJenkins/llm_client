use super::*;

#[derive(Debug)]
pub(super) struct MacroPresetOrganizations(pub(super) Vec<MacroPresetOrganization>);

impl MacroPresetOrganizations {
    pub(super) fn new() -> Self {
        let entries = std::fs::read_dir(&*PATH_TO_ORGS_DATA_DIR).unwrap_or_else(|_| {
            panic!(
                "Failed to read preset directory: {}",
                PATH_TO_ORGS_DATA_DIR.display()
            )
        });

        let preset_org_paths = entries
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        Some(e)
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>();
        if preset_org_paths.is_empty() {
            panic!(
                "No organization directories found in preset directory: {}",
                PATH_TO_ORGS_DATA_DIR.display()
            );
        }

        let mut organizations = Vec::new();
        for preset_org_path in preset_org_paths {
            let org_path = preset_org_path.path().join("organization.json");
            let org: DePresetOrganization = open_and_parse("organization.json", &org_path);

            let org = MacroPresetOrganization {
                friendly_name: org.friendly_name,
                hf_account: org.hf_account,
                preset_org_path: preset_org_path.path(),
            };
            organizations.push(org);
        }
        Self(organizations)
    }
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct DePresetOrganization {
    friendly_name: String,
    hf_account: String,
}

#[derive(Debug, Clone)]
pub(super) struct MacroPresetOrganization {
    pub(super) friendly_name: String,
    hf_account: String,
    preset_org_path: std::path::PathBuf,
}

impl MacroPresetOrganization {
    pub(super) fn const_ident(&self) -> Ident {
        format_ident!(
            "{}",
            self.friendly_name
                .to_uppercase()
                .replace(|c: char| !c.is_alphanumeric(), "_")
        )
    }

    pub(super) fn load_models(&self) -> Vec<MacroGgufPreset> {
        let mut models = Vec::new();
        let preset_path = self.preset_org_path.clone();
        let entries = fs::read_dir(&preset_path).unwrap_or_else(|_| {
            panic!("Failed to read preset directory: {}", preset_path.display())
        });

        let preset_model_paths = entries
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    if e.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        Some(e)
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>();
        if preset_model_paths.is_empty() {
            panic!(
                "No model directories found in preset directory: {}",
                preset_path.display()
            );
        }

        for preset_path in preset_model_paths {
            models.push(MacroGgufPreset::new(self.clone(), &preset_path.path()));
        }
        models
    }

    pub fn to_token_stream(self) -> TokenStream {
        let const_ident = self.const_ident();
        let friendly_name = self.friendly_name;
        let hf_account = self.hf_account;
        quote! {
            pub const #const_ident: LocalLlmOrganization = LocalLlmOrganization {
                friendly_name: #friendly_name,
                hf_account: #hf_account,
            };
        }
    }
}

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let organizations = MacroPresetOrganizations::new();
    let mut org_associated_consts = Vec::new();
    let mut org_associated_consts_idents = Vec::new();

    for org in organizations.0 {
        org_associated_consts_idents.push(org.const_ident());
        org_associated_consts.push(org.to_token_stream());
    }

    let code = quote! {
        #[derive(Debug, Clone, PartialEq)]
        pub struct LocalLlmOrganization {
            pub friendly_name: &'static str,
            pub hf_account: &'static str,
        }

        impl LocalLlmOrganization {
            pub fn all_organizations() -> Vec<Self> {
                vec![#(Self::#org_associated_consts_idents),*]
            }

            #(#org_associated_consts)*
        }
    };

    format_and_write(&output_path.join("organizations.rs"), code);
}
