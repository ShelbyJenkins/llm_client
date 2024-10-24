use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::tag_describer::TagDescription;

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

    pub fn tag_name(&self) -> String {
        self.name.as_ref().unwrap().to_owned()
    }

    pub fn tag_path(&self) -> String {
        format!(
            "root::{}",
            self.full_path
                .as_ref()
                .unwrap()
                .to_owned()
                .replace(":", "::")
        )
    }

    pub fn get_tags(&self) -> Vec<&Tag> {
        let mut tags: Vec<&Tag> = self.tags.values().collect();
        tags.sort_by(|a, b| a.name.cmp(&b.name));
        tags
    }

    pub fn get_tag(&self, tag_path: &str) -> Option<&Tag> {
        // Try the various parts of the tag path
        let potential_parts: Vec<&str> = tag_path.split_whitespace().collect();
        for part in potential_parts {
            let part_attempt = part.strip_prefix("root::").unwrap_or(part);

            // Split the path
            let tag_paths: Vec<&str> = part_attempt.split("::").collect();

            for i in (0..tag_paths.len()).rev() {
                // println!("trying: {:?}", &parts[..=i]);
                if let Some(tag) = self.get_tag_recursive(&tag_paths[..=i]) {
                    return Some(tag);
                }
            }
        }

        None
    }

    pub fn display_child_tags(&self) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            output.push_str(&child_tag.tag_name());
            output.push('\n');
        }
        output
    }

    pub fn display_immediate_child_paths(&self) -> String {
        let mut output = String::new();
        for tag in self.get_tags() {
            if !output.is_empty() {
                output.push_str("\n");
            }
            output.push_str(&tag.tag_path());
        }
        output
    }

    pub fn display_tag_criteria(&self, entity: &str) -> String {
        if let Some(description) = &self.description {
            indoc::formatdoc! {"'{}' is applicable if: '{entity}' {}",
            self.tag_name(),
            description.is_applicable.trim(),
            }
        } else {
            indoc::formatdoc! {"
            Classification '{}' is applicable if it applies to '{entity}'
            ",
            self.tag_name(),
            }
        }
    }

    pub fn display_all_tags_with_paths(&self) -> String {
        let mut result = String::new();
        self.collect_tags_with_paths(&mut result);
        result.trim_end().to_string()
    }

    pub fn display_all_tags_with_nested_paths(&self) -> String {
        let mut result = String::new();
        self.collect_tags_with_nested_paths(&mut result, true);
        result.trim_end().to_string()
    }

    fn collect_tags_with_paths(&self, result: &mut String) {
        if let Some(full_path) = &self.full_path {
            // Convert single ":" to "::" and prepend "root"
            let formatted_path = format!("root::{}", full_path.replace(":", "::"));
            result.push_str(&formatted_path);
            result.push('\n');
        }

        // Recursively process child tags
        for (_, child_tag) in &self.tags {
            child_tag.collect_tags_with_paths(result);
        }
    }

    fn collect_tags_with_nested_paths(&self, result: &mut String, is_root: bool) {
        let mut leaf_tags = Vec::new();
        let mut nested_tags = Vec::new();

        for (_, child_tag) in &self.tags {
            if child_tag.tags.is_empty() {
                if let Some(name) = &child_tag.name {
                    leaf_tags.push(name.clone());
                }
            } else {
                nested_tags.push(child_tag);
            }
        }

        if !leaf_tags.is_empty() {
            if let Some(full_path) = &self.full_path {
                let display_path = if is_root {
                    format!("root::{}", full_path)
                } else {
                    format!("root::{}", full_path)
                };
                result.push_str(&display_path);
                result.push_str("::{");
                result.push_str(&leaf_tags.join(", "));
                result.push_str("}\n");
            }
        }

        for nested_tag in nested_tags {
            nested_tag.collect_tags_with_nested_paths(result, false);
        }
    }

    fn get_tag_recursive(&self, path_parts: &[&str]) -> Option<&Tag> {
        if path_parts.is_empty() {
            return Some(self);
        }

        let current = path_parts[0];
        let remaining = &path_parts[1..];

        // Handle curly braces
        let tags_to_try = if current.starts_with('{') && current.ends_with('}') {
            current[1..current.len() - 1]
                .split(',')
                .map(|s| s.trim())
                .collect::<Vec<_>>()
        } else {
            vec![current]
        };

        for &tag in &tags_to_try {
            // Try case-insensitive exact match
            let matching_child = self
                .tags
                .iter()
                .find(|(child_name, _)| child_name.to_lowercase() == tag.to_lowercase());

            if let Some((_, child_tag)) = matching_child {
                if remaining.is_empty() {
                    return Some(child_tag);
                }
                if let Some(found) = child_tag.get_tag_recursive(remaining) {
                    return Some(found);
                }
            }
        }

        // If no match found and we have remaining parts, try skipping this part
        if !remaining.is_empty() {
            return self.get_tag_recursive(remaining);
        }

        None
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
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::i_nln(f, format_args!("{}", self.tag_path()))?;
        Ok(())
    }
}
