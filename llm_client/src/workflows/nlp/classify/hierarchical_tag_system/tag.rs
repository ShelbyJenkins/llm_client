use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::generation::TagDescription;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tag {
    pub name: Option<String>,
    pub full_path: Option<String>,
    pub description: Option<TagDescription>,
    pub tags: HashMap<String, Tag>,
}

impl Tag {
    pub fn new() -> Tag {
        Tag::default()
    }

    pub fn add_tag(&mut self, tag: Tag) {
        self.tags.insert(tag.tag_name(), tag);
    }

    pub fn add_tags(&mut self, tag: Vec<Tag>) {
        for tag in tag {
            self.tags.insert(tag.tag_name(), tag);
        }
    }

    pub fn clear_tags(&mut self) {
        self.tags.clear();
    }

    pub fn remove_tag(&mut self, name: &str) -> crate::Result<Tag> {
        let tag = match self.tags.remove(name) {
            Some(tag) => tag,
            None => crate::bail!("Tag not found."),
        };
        Ok(tag)
    }

    pub fn tag_name(&self) -> String {
        self.name.as_ref().unwrap().to_owned()
    }

    pub fn tag_path(&self) -> String {
        self.full_path.as_ref().unwrap().to_owned()
    }

    pub fn get_tags(&self) -> Vec<&Tag> {
        let mut tags: Vec<&Tag> = self.tags.values().collect();
        tags.sort_by(|a, b| a.name.cmp(&b.name));
        tags
    }

    pub fn get_tag_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tags.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn display_child_tags(&self) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            output.push_str(&child_tag.tag_name());
            output.push('\n');
        }
        output
    }

    pub fn display_child_tag_descriptions(&self) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            output.push_str(&child_tag.format_tag_description());
            output.push('\n');
        }
        output
    }

    fn format_tag_description(&self) -> String {
        if let Some(description) = &self.description {
            indoc::formatdoc! {"
            Parent Classification '{}' 
            {}
            ",
            self.tag_name(),
            description.is_applicable,
            }
        } else {
            indoc::formatdoc! {"
            Classification '{}' 
            ",
            self.tag_name(),
            }
        }
    }

    pub fn display_all_tags(&self) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            child_tag.format_tag_for_llm(&mut output, 0);
        }
        output
    }

    pub fn display_all_tags_with_paths(&self) -> String {
        let mut result = String::new();
        self.collect_tags_with_paths(&mut result, Vec::new());
        result.trim_end().to_string()
    }

    fn collect_tags_with_paths(&self, result: &mut String, mut current_path: Vec<String>) {
        if let Some(name) = &self.name {
            current_path.push(name.clone());
        }

        let mut leaf_tags = Vec::new();

        for (_, child_tag) in &self.tags {
            if child_tag.tags.is_empty() {
                if let Some(name) = &child_tag.name {
                    leaf_tags.push(name.clone());
                }
            } else {
                child_tag.collect_tags_with_paths(result, current_path.clone());
            }
        }

        if !leaf_tags.is_empty() {
            result.push_str(&current_path.join("::"));
            result.push_str("::{");
            result.push_str(&leaf_tags.join(", "));
            result.push_str("}\n");
        }
    }

    pub(super) fn add_tag_recursive(&mut self, parts: &[&str], depth: usize, path_sep: &str) {
        if depth < parts.len() {
            let current_path = parts[..=depth].join(path_sep);
            let tag = self.new_tag(&current_path, path_sep);

            if depth < parts.len() - 1 {
                tag.add_tag_recursive(parts, depth + 1, path_sep);
            }
        }
    }

    fn new_tag(&mut self, full_path: &str, path_sep: &str) -> &mut Tag {
        let parts: Vec<&str> = full_path.split(path_sep).collect();
        let name = parts
            .last()
            .unwrap()
            .to_string()
            .split_whitespace()
            .map(str::trim)
            .collect::<Vec<&str>>()
            .join("-")
            .to_lowercase();
        let full_path = parts
            .iter()
            .map(|&s| {
                s.split_whitespace()
                    .map(str::trim)
                    .collect::<Vec<&str>>()
                    .join("-")
                    .to_lowercase()
            })
            .collect::<Vec<String>>()
            .join(":");

        let tag = Tag {
            description: None,
            name: Some(name.to_owned()),
            full_path: Some(full_path.to_owned()),
            tags: HashMap::new(),
        };

        self.tags.entry(tag.tag_name()).or_insert_with(|| tag)
    }

    fn format_tag_for_llm(&self, output: &mut String, depth: usize) {
        for _ in 0..depth {
            output.push_str("  ");
        }
        output.push_str(&self.tag_name());
        if self.tags.is_empty() && !output.is_empty() {
            output.push(';');
            output.push('\n');
        }
        if !self.get_tags().is_empty() {
            output.push(':');
            output.push('\n');
            for child_tag in self.get_tags() {
                child_tag.format_tag_for_llm(output, depth + 1);
            }
        }
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        crate::i_nln(f, format_args!("{}", self.display_all_tags()))?;
        Ok(())
    }
}
