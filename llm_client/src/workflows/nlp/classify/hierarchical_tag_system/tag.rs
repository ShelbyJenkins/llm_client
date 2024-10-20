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

    pub fn get_tag_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tags.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_tag(&self, tag_path: &str) -> Option<&Tag> {
        // Remove 'root::' prefix if present
        let tag_path = tag_path.strip_prefix("root::").unwrap_or(tag_path);
        // Extract the actual tag path by splitting on whitespace and taking the first part
        let actual_path = tag_path.split_whitespace().next().unwrap_or(tag_path);
        // Split the path
        let parts: Vec<&str> = actual_path.split("::").collect();

        for i in (0..parts.len()).rev() {
            // println!("trying: {:?}", &parts[..=i]);
            if let Some(tag) = self.get_tag_recursive(&parts[..=i]) {
                return Some(tag);
            }
        }

        None
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

    pub fn display_child_tags(&self) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            output.push_str(&child_tag.tag_name());
            output.push('\n');
        }
        output
    }

    pub fn display_child_tags_comma(&self) -> String {
        let mut names = String::new();
        for child_tag in self.get_tags() {
            names.push_str(&format!("'{}', ", child_tag.tag_name()));
        }
        names
    }

    pub fn display_child_tag_descriptions(&self, entity: &str) -> String {
        let mut output = String::new();
        for child_tag in self.get_tags() {
            output.push_str(&child_tag.format_tag_criteria(entity));
            output.push('\n');
        }
        output
    }

    pub fn format_tag_criteria(&self, entity: &str) -> String {
        if let Some(description) = &self.description {
            if description.is_parent_tag {
                indoc::formatdoc! {"Classification '{}' is applicable if '{entity}' {}",
                self.tag_name(),
                description.is_applicable.trim(),
                }
            } else {
                indoc::formatdoc! {"Classification '{}' is applicable if '{entity}' {} Specifically, of type '{}'",
                self.tag_name(),
                description.is_applicable.trim(),
                self.tag_name(),
                }
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
        self.collect_tags_with_paths(&mut result);
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

    pub fn display_all_tags_with_nested_paths(&self) -> String {
        let mut result = String::new();
        self.collect_tags_with_nested_paths(&mut result, vec!["root".to_string()]);
        result.trim_end().to_string()
    }

    fn collect_tags_with_nested_paths(&self, result: &mut String, mut current_path: Vec<String>) {
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
                child_tag.collect_tags_with_nested_paths(result, current_path.clone());
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
        crate::i_nln(f, format_args!("{}", self.display_all_tags_with_paths()))?;
        Ok(())
    }
}
