use llm_models::local_model::{
    gguf::{preset::LlmPreset, GgufLoader},
    GgufPresetTrait, LocalLlmModel,
};

#[test]
fn load_from_vram() {
    // let model = GgufLoader::default()
    //     .mistral_small_instruct2409()
    //     .preset_with_available_vram_gb(48)
    //     .load()
    //     .unwrap();

    // println!("{:#?}", model);

    let model: LocalLlmModel = GgufLoader::default()
        .llama3_1_8b_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .phi3_5_mini_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);

    let model = GgufLoader::default()
        .mistral_nemo_instruct2407()
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
    let variants = vec![
        LlmPreset::Llama3_1_8bInstruct,
        LlmPreset::Mistral7bInstructV0_3,
        LlmPreset::Mixtral8x7bInstructV0_1,
        LlmPreset::MistralNemoInstruct2407,
        LlmPreset::MistralSmallInstruct2409,
        LlmPreset::Phi3Medium4kInstruct,
        LlmPreset::Phi3Mini4kInstruct,
        LlmPreset::Phi3_5MiniInstruct,
    ];
    for variant in variants {
        println!("{:#?}", variant.model_id());
        println!("{:#?}", variant.gguf_repo_id());
        println!("{:#?}", variant.config_json());
        println!("{:#?}", variant.number_of_parameters());
        for i in 1..=8 {
            println!("{:#?}", variant.f_name_for_q_bits(i));
        }
    }
}
