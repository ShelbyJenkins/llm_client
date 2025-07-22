use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufPreset {
    pub model_base: LlmModelBase,
    pub organization: LocalLlmOrganization,
    pub model_repo_id: Cow<'static, str>,
    pub gguf_repo_id: Cow<'static, str>,
    pub number_of_parameters: f64,
    pub tokenizer_file_name: Option<Cow<'static, str>>,
    pub config: GgufPresetConfig,
    pub quants: Cow<'static, [GgufPresetQuant]>,
    pub preset_llm_id: GgufPresetId,
}

impl GgufPreset {
    pub fn hf_repo_link(&self) -> String {
        format!("https://huggingface.co/{}", self.gguf_repo_id)
    }

    pub fn tokenizer_path(&self) -> crate::Result<Option<PathBuf>> {
        tokenizer_path(&self.tokenizer_file_name)
    }

    pub fn select_quant_for_q_lvl(&self, q_lvl: u8) -> Option<GgufPresetQuant> {
        self.quants
            .iter()
            .find(|quant| quant.q_lvl == q_lvl)
            .map(|quant| quant)
            .cloned()
    }

    pub fn select_quant_for_available_memory(
        &self,
        ctx_size: Option<u64>,
        use_memory_bytes: u64,
    ) -> Result<GgufPresetQuant, crate::Error> {
        let ctx_memory_size_bytes = estimate_context_size_bytes(
            ctx_size,
            self.config.embedding_length,
            self.config.head_count,
            self.config.head_count_kv.unwrap(),
            self.config.block_count,
            None,
        );
        let initial_q_lvls = estimate_max_quantization_level(
            self.number_of_parameters,
            use_memory_bytes,
            ctx_memory_size_bytes,
        )?;
        let mut q_lvls = initial_q_lvls;
        loop {
            if let Some(quant) = self.select_quant_for_q_lvl(q_lvls) {
                return Ok(quant);
            }
            if q_lvls == 1 {
                crate::bail!(
                    "No model file found from quantization levels: {initial_q_lvls}-{q_lvls}",
                );
            } else {
                q_lvls -= 1;
            }
        }
    }

    pub fn quant_file_name_for_q_lvl(&self, q_lvls: u8) -> Option<String> {
        self.quants
            .iter()
            .find(|quant| quant.q_lvl == q_lvls)
            .map(|quant| quant.fname.to_string())
    }
}

impl Default for GgufPreset {
    fn default() -> Self {
        GgufPreset::LLAMA_3_2_1B_INSTRUCT
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufPresetQuant {
    pub q_lvl: u8,
    pub fname: Cow<'static, str>,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GgufPresetConfig {
    pub context_length: u64,
    pub embedding_length: u64,
    pub feed_forward_length: Option<u64>,
    pub head_count: u64,
    pub head_count_kv: Option<u64>,
    pub block_count: u64,
    pub torch_dtype: Cow<'static, str>,
    pub vocab_size: u64,
    pub architecture: Cow<'static, str>,
    pub model_size_bytes: Option<u64>,
}

static TOKENIZERS_DIR: std::sync::LazyLock<PathBuf> = std::sync::LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("llm")
        .join("local")
        .join("gguf")
        .join("tokenizers")
});

fn tokenizer_path(file_name: &Option<Cow<'static, str>>) -> crate::Result<Option<PathBuf>> {
    if let Some(file_name) = file_name {
        let tokenizer_path = TOKENIZERS_DIR.join(file_name.to_string());
        if tokenizer_path.exists() {
            Ok(Some(tokenizer_path))
        } else {
            crate::bail!("Tokenizer file not found: {:?}", tokenizer_path)
        }
    } else {
        Ok(None)
    }
}
