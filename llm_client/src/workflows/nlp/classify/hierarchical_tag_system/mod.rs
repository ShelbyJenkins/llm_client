pub mod generation;
pub mod tag;

use generation::TagCollectionDescriber;
use serde::{Deserialize, Serialize};
pub use tag::Tag;

const OUTPUT_DIR: &str = "generations";
const DEFAULT_COLLECTION_NAME: &str = "default_collection";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagCollection {
    pub collection_name: Option<String>,
    pub from_text_file_path: Option<std::path::PathBuf>,
    pub from_string: Option<String>,
    pub output_dir_path: std::path::PathBuf,
    pub tag_path_seperator: String,
    contents: String,
    root_tag: Option<Tag>,
}

impl Default for TagCollection {
    fn default() -> Self {
        TagCollection {
            collection_name: None,
            from_text_file_path: None,
            from_string: None,
            output_dir_path: OUTPUT_DIR.into(),
            tag_path_seperator: ":".into(),
            contents: String::new(),
            root_tag: None,
        }
    }
}

impl TagCollection {
    pub fn new() -> TagCollection {
        TagCollection::default()
    }

    pub fn collection_name<S: AsRef<str>>(mut self, collection_name: S) -> Self {
        self.collection_name = Some(collection_name.as_ref().to_owned());
        self
    }

    pub fn from_text_file_path<P: Into<std::path::PathBuf>>(
        mut self,
        from_text_file_path: P,
    ) -> Self {
        self.from_text_file_path = Some(from_text_file_path.into());
        self
    }

    pub fn from_string<S: AsRef<str>>(mut self, from_string: S) -> Self {
        self.from_string = Some(from_string.as_ref().to_owned());
        self
    }

    pub fn output_dir_path<P: AsRef<std::path::PathBuf>>(mut self, output_dir_path: P) -> Self {
        self.output_dir_path = output_dir_path.as_ref().to_owned();
        self
    }

    pub fn tag_path_seperator<S: AsRef<str>>(mut self, tag_path_seperator: S) -> Self {
        self.tag_path_seperator = tag_path_seperator.as_ref().to_owned();
        self
    }

    pub fn load(self) -> crate::Result<Self> {
        match (
            &self.collection_name,
            &self.from_string,
            &self.from_text_file_path,
        ) {
            (_, Some(_), Some(_)) => {
                crate::bail!("Only one of from_string, or from_text_file_path can be provided.")
            }
            (Some(_), None, None) => self.load_from_collection_name(),
            (_, Some(_), None) => self.load_from_string(),
            (_, None, Some(_)) => self.load_from_text_file(),
            (None, None, None) => crate::bail!(
                "collection_name, from_string, or from_text_file_path must be provided."
            ),
        }
    }

    pub fn get_root_tag(&self) -> crate::Result<Tag> {
        if let Some(root_tag) = &self.root_tag {
            Ok(root_tag.clone())
        } else {
            crate::bail!("Root tag is not available.")
        }
    }

    pub async fn populate_descriptions(
        &mut self,
        llm_client: &crate::LlmClient,
        criteria: &str,
    ) -> crate::Result<()> {
        TagCollectionDescriber::run(llm_client, criteria, self).await?;

        self.save_as_json()?;
        Ok(())
    }

    fn load_from_string(mut self) -> crate::Result<Self> {
        let from_string = if let Some(from_string) = &self.from_string {
            from_string.to_owned()
        } else {
            crate::bail!("from_string is not provided.");
        };
        let collection_name = if let Some(collection_name) = &self.collection_name {
            collection_name.to_owned()
        } else {
            let snippet = from_string
                .chars()
                .filter(|c| !c.is_whitespace())
                .take(12)
                .collect::<String>();
            let collection_name = format!("{DEFAULT_COLLECTION_NAME}_{}", snippet);
            self.collection_name = Some(collection_name.clone());
            collection_name
        };
        if let Some(loaded_collection) = self.load_from_json(collection_name.as_ref())? {
            if from_string == loaded_collection.contents {
                return Ok(loaded_collection);
            }
        }
        self.build_from_contents(&from_string)?;
        self.save_as_json()?;
        Ok(self)
    }

    fn load_from_text_file(mut self) -> crate::Result<Self> {
        let from_text_file_path = if let Some(from_text_file_path) = &self.from_text_file_path {
            from_text_file_path.to_owned()
        } else {
            crate::bail!("from_text_file_path is not provided.");
        };
        let from_text_file = match std::fs::read_to_string(&from_text_file_path) {
            Ok(contents) => contents,
            Err(e) => crate::bail!("Error reading file: {}", e),
        };

        let collection_name = if let Some(collection_name) = &self.collection_name {
            collection_name.to_owned()
        } else {
            // or return anyhow::bail!("Error getting file stem.");
            let collection_name = std::path::Path::new(&from_text_file_path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(|s| s.to_owned())
                .ok_or_else(|| anyhow::anyhow!("Error getting file stem."))?;
            self.collection_name = Some(collection_name.clone());
            collection_name
        };
        if let Some(loaded_collection) = self.load_from_json(collection_name.as_ref())? {
            if from_text_file == loaded_collection.contents {
                return Ok(loaded_collection);
            }
        }
        self.build_from_contents(&from_text_file)?;
        self.save_as_json()?;
        Ok(self)
    }

    fn load_from_collection_name(mut self) -> crate::Result<Self> {
        let collection_name = self
            .collection_name
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("collection_name is not provided."))?;
        if let Some(loaded_collection) = self.load_from_json(collection_name.as_ref())? {
            Ok(loaded_collection)
        } else {
            self.root_tag = Some(Tag::default());
            self.save_as_json()?;
            Ok(self)
        }
    }

    fn build_from_contents(&mut self, contents: &str) -> crate::Result<()> {
        self.contents = contents.to_string();
        let mut root_tag = Tag::default();
        for line in contents.lines() {
            let parts: Vec<&str> = line.split(&self.tag_path_seperator).collect();
            root_tag.add_tag_recursive(&parts, 0, &self.tag_path_seperator);
        }

        self.root_tag = Some(root_tag);
        Ok(())
    }

    fn load_from_json(&self, collection_name: &str) -> crate::Result<Option<Self>> {
        let mut file_path = self.output_dir_path.clone();
        file_path.push(format!("{}.json", collection_name));

        match std::fs::read_to_string(&file_path) {
            Ok(json_content) => {
                let loaded_data: Self = serde_json::from_str(&json_content)?;
                Ok(Some(loaded_data))
            }
            Err(_) => Ok(None),
        }
    }

    fn save_as_json(&self) -> crate::Result<()> {
        let collection_name = if let Some(collection_name) = &self.collection_name {
            collection_name.to_owned()
        } else {
            crate::bail!("collection_name is not provided.");
        };
        let filename = format!("{collection_name}.json");

        // Create a PathBuf for the directory
        let mut file_path = std::path::PathBuf::from(OUTPUT_DIR);

        // Create the directory if it doesn't exist
        std::fs::create_dir_all(&file_path)?;

        // Add the filename to the directory path
        file_path.push(&filename);

        // Remove the file if it already exists
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
        }

        // Serialize the struct to JSON
        let json_content = serde_json::to_string_pretty(self)?;

        // Write the JSON to file
        std::fs::write(&file_path, json_content)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn create_sample_tag_collection() -> TagCollection {
        let input = "\
    terrestrial
    terrestrial:arid
    terrestrial:soil
    aquatic
    aquatic:fresh water
    aquatic:fresh water:lake
    bed
    bed:hotel:queen
    bed:hotel:king
    bed:home:double:bunk
    host-associated:animal
    host:digestive
    tract:mouth
    salinity:low
    salinity
    age group:infant
    age  group:old age:senior
    age-group:young
    age-group:young: new born
    ";
        TagCollection::default()
            .from_string(input)
            .tag_path_seperator(":")
            .load()
            .unwrap()
    }

    #[test]
    #[ignore]
    fn test_tag_collection_creation_from_string() -> crate::Result<()> {
        let tag_collection = create_sample_tag_collection();
        let tags = tag_collection.get_root_tag()?;
        for tag in tags.get_tags() {
            println!("{}", tag.display_child_tags());
        }
        println!("{}", tags.display_all_tags());
        println!("{}", tags.display_child_tags());
        println!("{}", tags.display_all_tags_with_nested_paths());
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_tag_collection_creation_from_file() -> crate::Result<()> {
        let tag_collection = TagCollection::default()
            .from_text_file_path("/workspaces/test/bacdive_hierarchy.txt")
            .tag_path_seperator(":")
            .load()?;
        let tags = tag_collection.get_root_tag()?;

        println!("{}", tags.display_all_tags_with_paths());
        println!("{}", tags.display_all_tags_with_nested_paths());
        assert!(tags.get_tag("host::arthropoda::tick").is_some());
        assert!(tags
            .get_tag("infection::patient::{blood-culture}")
            .is_some());
        assert!(tags.get_tag("root::host::arthropoda::tick").is_some());
        assert!(tags
            .get_tag("root::infection::patient::{blood-culture}")
            .is_some());
        Ok(())
    }
}
