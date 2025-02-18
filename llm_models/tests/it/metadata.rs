use llm_models::{GgufLoader, GgufPresetTrait};

#[test]
fn test_base_generation_prefix() {
    let model = GgufLoader::default()
        .llama_3_2_1b_instruct()
        .preset_with_memory_gb(48)
        .load()
        .unwrap();
    println!("{:#?}", model.chat_template.base_generation_prefix);
    assert_eq!(
        Some("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        model.chat_template.base_generation_prefix.as_deref()
    );

    let model = GgufLoader::default()
        .phi_3_5_mini_instruct()
        .preset_with_memory_gb(48)
        .load()
        .unwrap();
    println!("{:#?}", model.chat_template.base_generation_prefix);
    assert_eq!(
        Some("<|assistant|>\n"),
        model.chat_template.base_generation_prefix.as_deref()
    );
}
