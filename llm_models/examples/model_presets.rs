use llm_models::*;

fn main() {
    // Using
    let _model: LocalLlmModel = GgufLoader::default()
        .llama3_1_8b_instruct()
        .preset_with_available_vram_gb(48)
        .load()
        .unwrap();

    // model.local_model_path can now be used to load the model into the inference engine.
}
