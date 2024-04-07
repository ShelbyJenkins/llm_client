use chrono::Utc;
use dotenv::dotenv;
use std::{
    env,
    fs::{self, File},
    io::{self, Read, Write},
    path::PathBuf,
};
// See llm_client/src/bin/model_loader_cli.rs for cli instructions
// Downloads to Path: "/root/.cache/huggingface/hub/

pub async fn check_requested_model_against_given_model(
    model_id: &str,
    model_filename: &str,
    given_path: PathBuf,
) -> bool {
    let requested_path = download_model(model_id, model_filename, None)
        .await
        .unwrap();

    given_path == requested_path
}

pub async fn download_model(
    model_id: &str,
    model_filename: &str,
    model_token: Option<String>,
) -> Option<PathBuf> {
    let model_token = model_token.or_else(|| {
        let model_token = env::var("HUGGING_FACE_TOKEN").ok();
        if model_token.is_none() {
            dotenv().ok(); // Load .env file
            dotenv::var("HUGGING_FACE_TOKEN").ok()
        } else {
            model_token
        }
    });

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(true)
        .with_token(model_token)
        .build()
        .unwrap();

    let local_model_filename = api
        .model(model_id.into())
        .get(model_filename)
        .await
        .unwrap();

    // println!("Downloaded model: {} ", model_id);

    let config_json = api.model(model_id.into()).get("config.json").await.unwrap();
    let readme_md = api.model(model_id.into()).get("README.md").await.unwrap();

    let local_path;
    if let Ok(absolute_path) = local_model_filename.canonicalize() {
        local_path = Some(absolute_path.clone());
        // println!("Local model path: {:?}", absolute_path.display());
    } else {
        local_path = None;
        // println!(
        //     "Could not get the absolute path for: {:?}",
        //     local_model_filename.display()
        // );
    }

    let model_info = api.model(model_id.to_string()).info().await;
    if let Ok(model_info) = model_info {
        // println!("Model info: {:?}", model_info);
        save_model_info(
            model_id,
            model_info,
            &local_path,
            model_filename.to_string(),
            config_json,
            readme_md,
        )
        .unwrap();
    } else {
        println!("Could not get model info: {:?}", model_info);
    }

    local_path
}

fn save_model_info(
    model_id: &str,
    model_info: hf_hub::api::RepoInfo,
    local_path: &Option<PathBuf>,
    model_filename: String,
    config_json: PathBuf,
    readme_md: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let models_dir = manifest_dir.join("hf_downloaded_models");
    fs::create_dir_all(&models_dir)?;
    let model_id_dir = models_dir.join(model_id);
    fs::create_dir_all(&model_id_dir)?;

    let current_date = Utc::now().format("%Y/%m/%d %H:%M").to_string();
    let mut model_info_yaml_data =
        serde_yaml::to_string(&format!("Model last loaded on: {}", current_date))?;
    if let Some(local_path) = local_path {
        model_info_yaml_data.push_str(&format!("\nlocal_path: {}", local_path.display()));
    }
    model_info_yaml_data.push_str(&format!("\nlocal_model_filename: {}", model_filename));

    model_info_yaml_data.push_str("\nModel info from hf_hub:");
    model_info_yaml_data.push_str(&model_info.sha);
    for sibling in model_info.siblings {
        model_info_yaml_data.push_str(&format!("\nrfilename: {}", sibling.rfilename));
    }

    // Save the YAML data to a file named `local_model_filename`
    let yaml_file_path = model_id_dir.join(&model_filename);
    let mut yaml_file = File::create(yaml_file_path)?;
    yaml_file.write_all(model_info_yaml_data.as_bytes())?;

    // Save the README.md file
    let readme_file_path = model_id_dir.join("README.md");
    copy_file(&readme_md, &readme_file_path)?;

    // Save the config.json file
    let config_file_path = model_id_dir.join("config.json");
    copy_file(&config_json, &config_file_path)?;

    Ok(())
}

fn copy_file(src: &PathBuf, dest: &PathBuf) -> io::Result<()> {
    let mut src_file = File::open(src)?;
    let mut dest_file = File::create(dest)?;
    let mut contents = Vec::new();
    src_file.read_to_end(&mut contents)?;
    dest_file.write_all(&contents)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_get() {
        let model_id = "julien-c/dummy-unknown";
        let model_filename = "config.json";

        let downloaded_path = download_model(model_id, model_filename, None).await;
        if let Some(downloaded_path) = downloaded_path.clone() {
            assert!(downloaded_path.exists());
            assert!(
                check_requested_model_against_given_model(
                    model_id,
                    model_filename,
                    downloaded_path
                )
                .await
            );
        } else {
            panic!("Could not download model");
        }
    }
}
