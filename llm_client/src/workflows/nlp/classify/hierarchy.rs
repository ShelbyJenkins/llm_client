use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Tag {
    pub name: Option<String>,
    pub full_path: Option<String>,
    tags: HashMap<String, Tag>,
}

impl Tag {
    pub fn new() -> Tag {
        Tag::default()
    }

    pub fn new_collection_from_string(input: &str) -> Tag {
        let mut root = Tag {
            name: None,
            full_path: None,
            tags: HashMap::new(),
        };
        for line in input.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            root.add_tag_recursive(&parts, 0);
        }

        root
    }

    pub fn new_collection_from_text_file<P: AsRef<std::path::Path>>(path: P) -> Tag {
        match std::fs::read_to_string(path) {
            Ok(contents) => Tag::new_collection_from_string(&contents),
            Err(e) => panic!("Error reading file: {}", e),
        }
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

    pub fn get_tags_owned(&self) -> Vec<Tag> {
        let mut tags: Vec<Tag> = self.tags.values().cloned().collect();
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
        if let Some(path) = &self.full_path {
            result.push_str(path);
            result.push('\n');
        }

        for child_tag in self.get_tags() {
            child_tag.collect_tags_with_paths(result);
        }
    }

    fn add_tag_recursive(&mut self, parts: &[&str], depth: usize) {
        if depth < parts.len() {
            let current_path = parts[..=depth].join(":");
            let tag = self.new_tag(&current_path);

            if depth < parts.len() - 1 {
                tag.add_tag_recursive(parts, depth + 1);
            }
        }
    }

    fn new_tag(&mut self, full_path: &str) -> &mut Tag {
        let parts: Vec<&str> = full_path.split(':').collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    fn create_sample_tag_collection() -> Tag {
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
        Tag::new_collection_from_string(input)
    }

    #[test]
    #[ignore]
    fn test_tag_collection_creation_from_file() {
        let tags = Tag::new_collection_from_text_file("/workspaces/test/bacdive_hierarchy.txt");
        for tag in tags.get_tags() {
            println!("{}", tag.display_child_tags());
        }
        println!("{}", tags.display_all_tags());
        println!("{}", tags.display_child_tags());
        println!("{}", tags.display_all_tags_with_paths());
    }
}
