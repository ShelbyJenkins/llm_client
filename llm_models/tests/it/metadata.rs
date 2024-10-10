use llm_models::local_model::{gguf::GgufLoader, GgufPresetTrait};

#[test]
fn test_base_generation_prefix() {
    let model = GgufLoader::default()
        .llama3_1_8b_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();
    assert_eq!(
        Some("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        model.chat_template.base_generation_prefix.as_deref()
    );
    let model = GgufLoader::default()
        .mistral7b_instruct_v0_3()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();
    assert_eq!(
        Some(""),
        model.chat_template.base_generation_prefix.as_deref()
    );
    let model = GgufLoader::default()
        .phi3_5_mini_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();
    assert_eq!(
        Some("<|assistant|>\n"),
        model.chat_template.base_generation_prefix.as_deref()
    );
}
