use super::*;

pub(super) fn generate(output_path: &std::path::PathBuf) {
    let organizations = MacroPresetOrganizations::new();
    let input_dir = PATH_TO_ORGS_DATA_DIR.join("tokenizers");
    let output_dir = output_path.join("tokenizers");

    // If output directory doesn't exist, create it
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).unwrap();
    }

    // Create a set of all files in the output directory before we start
    let mut existing_files: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(&output_dir) {
        for entry in entries.flatten() {
            if let Ok(path) = entry.path().strip_prefix(&output_dir) {
                existing_files.insert(path.to_path_buf());
            }
        }
    }

    let mut pairs = std::collections::HashSet::new();

    for org in organizations.0 {
        for model in org.load_models() {
            if let (Some(input_path), Some(output_path)) = (
                model.input_tokenizer_path.as_ref(),
                model.output_tokenizer_path.as_ref(),
            ) {
                let pair = (input_path.clone(), output_path.clone());

                // Only copy if we haven't seen this pair before
                if !pairs.contains(&pair) {
                    pairs.insert(pair);

                    // Construct the source and destination paths
                    let source = input_dir.join(input_path);
                    let destination = output_dir.join(output_path);

                    // Only copy if files are different or destination doesn't exist
                    if !destination.exists() || !files_are_identical(&source, &destination) {
                        std::fs::copy(source, destination).unwrap();
                    }
                }
            }
        }
    }

    // Remove files that are no longer needed
    let used_paths: std::collections::HashSet<&std::path::PathBuf> =
        pairs.iter().map(|(_, output)| output).collect();
    for file in existing_files {
        if !used_paths.contains(&file) {
            let path_to_remove = output_dir.join(&file);
            let _ = std::fs::remove_file(path_to_remove);
        }
    }
}

fn files_are_identical(path1: &std::path::Path, path2: &std::path::Path) -> bool {
    // Quick size check first
    let metadata1 = std::fs::metadata(path1).unwrap();
    let metadata2 = std::fs::metadata(path2).unwrap();

    if metadata1.len() != metadata2.len() {
        return false;
    }

    // Compare contents if sizes match
    let contents1 = std::fs::read(path1).unwrap();
    let contents2 = std::fs::read(path2).unwrap();
    contents1 == contents2
}
