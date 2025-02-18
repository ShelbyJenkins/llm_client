use llm_models::{gguf_presets::GgufPreset, GgufLoader, GgufPresetTrait, LocalLlmModel};

#[test]
fn load_from_vram() {
    let model = GgufLoader::default()
        .llama_3_2_1b_instruct()
        .preset_with_memory_gb(48)
        .load()
        .unwrap();

    println!("{:#?}", model);

    // let model = GgufLoader::default()
    //     .phi_4()
    //     .preset_with_memory_gb(48)
    //     .load()
    //     .unwrap();

    // println!("{:#?}", model);

    // let model = GgufLoader::default()
    //     .phi_3_5_moe_instruct()
    //     .preset_with_memory_gb(48)
    //     .load()
    //     .unwrap();

    // println!("{:#?}", model);

    // let model = GgufLoader::default()
    //     .mistral_small_24b_instruct_2501()
    //     .preset_with_memory_gb(48)
    //     .load()
    //     .unwrap();

    // println!("{:#?}", model);

    // let model = GgufLoader::default()
    //     .llama_3_1_nemotron_51b_instruct()
    //     .preset_with_memory_gb(48)
    //     .load()
    //     .unwrap();

    // println!("{:#?}", model);
}

#[test]
fn load_from_q_level() {
    let model: LocalLlmModel = GgufLoader::default()
        .llama_3_2_1b_instruct()
        .preset_with_quantization_level(8)
        .load()
        .unwrap();

    println!("{:#?}", model);
}

#[test]
fn models_macros_test() {
    // Add get all variants macro
    let variants = GgufPreset::all_models();
    for variant in variants {
        println!("{:#?}", variant.model_id);
        println!("{:#?}", variant.gguf_repo_id);
        println!("{:#?}", variant.config);
        println!("{:#?}", variant.number_of_parameters);
        for i in 1..=8 {
            println!("{:#?}", variant.quant_file_name_for_q_bit(i));
        }
    }
}
