use super::{model::DeGgufPreset, *};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeQuantFileNames {
    pub q8: Option<String>,
    pub q7: Option<String>,
    pub q6: Option<String>,
    pub q5: Option<String>,
    pub q4: Option<String>,
    pub q3: Option<String>,
    pub q2: Option<String>,
    pub q1: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MacroQuantFile {
    pub q_lvl: u8,
    pub fname: String,
    pub total_bytes: u64,
}

impl MacroQuantFile {
    pub fn get_quants(preset: &DeGgufPreset) -> Vec<MacroQuantFile> {
        let blobs_info: serde_json::Value = hf_api()
            .model(preset.gguf_repo_id.clone().into())
            .info_request()
            .query("blobs", "true")
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        let hf_info: HuggingFaceRepoInfo = serde_json::from_value(blobs_info.clone()).unwrap();

        let mut quants = Vec::new();
        for q_lvl in 1..=8 {
            let quant_filename = match q_lvl {
                1 => preset.quant_file_names.q1.clone(),
                2 => preset.quant_file_names.q2.clone(),
                3 => preset.quant_file_names.q3.clone(),
                4 => preset.quant_file_names.q4.clone(),
                5 => preset.quant_file_names.q5.clone(),
                6 => preset.quant_file_names.q6.clone(),
                7 => preset.quant_file_names.q7.clone(),
                8 => preset.quant_file_names.q8.clone(),
                _ => panic!("Invalid quantization level"),
            };

            let fname = if let Some(fname) = quant_filename {
                fname.to_owned()
            } else {
                continue;
            };

            let total_bytes = match hf_info.get_file(&fname).map(|f| f.size) {
                Some(size) => size as u64,
                None => {
                    panic!(
                        "Could not find quant file {} in repo {}",
                        fname, preset.gguf_repo_id
                    )
                }
            };

            quants.push(MacroQuantFile {
                q_lvl,
                fname,
                total_bytes,
            });
        }
        quants
    }

    pub fn to_token_stream(quants: &Vec<Self>) -> TokenStream {
        let mut preset_quants = Vec::new();
        for quant in quants {
            let q_lvl = quant.q_lvl;
            let fname = &quant.fname;
            let total_bytes = quant.total_bytes;
            preset_quants.push(quote! {
                GgufPresetQuant {
                    q_lvl: #q_lvl,
                    fname: Cow::Borrowed(#fname),
                    total_bytes: #total_bytes,

                }
            });
        }
        quote! {
           &[ #(#preset_quants),*]
        }
    }
}
