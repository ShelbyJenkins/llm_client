use std::{fs::File, io::Read, path::PathBuf};

pub fn load_content_path(content_path: &PathBuf) -> String {
    match File::open(content_path) {
        Ok(mut file) => {
            let mut custom_prompt = String::new();
            match file.read_to_string(&mut custom_prompt) {
                Ok(_) => {
                    if custom_prompt.trim().is_empty() {
                        panic!("content_path '{}' is empty.", content_path.display())
                    }
                    custom_prompt
                }
                Err(e) => panic!("Failed to read file: {}", e),
            }
        }
        Err(e) => panic!("Failed to open file: {}", e),
    }
}
