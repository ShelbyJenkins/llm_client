use llm_models::{
    local_model::gguf::preset::LlmPreset, GgufLoader, GgufPresetTrait, LocalLlmModel,
};

#[test]
fn load_from_vram() {
    let model = GgufLoader::default()
        .phi4()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .phi3_5_moe_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .llama3_1_51b_nemotron_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);
}

#[test]
fn load_from_q_level() {
    let model: LocalLlmModel = GgufLoader::default()
        .llama3_1_8b_instruct()
        .preset_with_quantization_level(8)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .phi3_5_mini_instruct()
        .preset_with_quantization_level(8)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .mistral_nemo_instruct2407()
        .preset_with_quantization_level(8)
        .load()
        .unwrap();

    println!("{:#?}", model);
}

#[test]
fn models_macros_test() {
    // Add get all variants macro
    let variants = vec![
        LlmPreset::Llama3_1_70bNemotronInstruct,
        LlmPreset::MistralNemoMinitron8bInstruct,
        LlmPreset::Llama3_1_8bInstruct,
        LlmPreset::Mistral7bInstructV0_3,
        LlmPreset::Mixtral8x7bInstructV0_1,
        LlmPreset::MistralNemoInstruct2407,
        LlmPreset::MistralSmallInstruct2409,
        LlmPreset::Llama3_1_70bNemotronInstruct,
        LlmPreset::Llama3_1_51bNemotronInstruct,
        LlmPreset::Phi3Medium4kInstruct,
        LlmPreset::Phi3Mini4kInstruct,
        LlmPreset::Phi3_5MiniInstruct,
        LlmPreset::Phi3_5MoeInstruct,
        LlmPreset::Phi4,
        LlmPreset::Llama3_2_1bInstruct,
        LlmPreset::Llama3_2_3bInstruct,
        LlmPreset::SuperNovaMedius13b,
        LlmPreset::Granite3_2bInstruct,
        LlmPreset::Granite3_8bInstruct,
        LlmPreset::Qwen2_5_32bInstruct,
        LlmPreset::Qwen2_5_14bInstruct,
        LlmPreset::Qwen2_5_7bInstruct,
        LlmPreset::Qwen2_5_3bInstruct,
        LlmPreset::StableLm2_12bChat,
    ];
    for variant in variants {
        println!("{:#?}", variant.model_id());
        println!("{:#?}", variant.gguf_repo_id());
        println!("{:#?}", variant.config_json());
        println!("{:#?}", variant.number_of_parameters());
        for i in 1..=8 {
            println!("{:#?}", variant.f_name_for_q_bits(i));
        }
        println!("{:#?}", variant.get_data());
    }
}
