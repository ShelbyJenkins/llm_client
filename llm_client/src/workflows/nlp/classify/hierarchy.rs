use std::collections::HashMap;

use llm_utils::grammar::Grammar;

use crate::{primitives::ExactStringPrimitive, PrimitiveTrait};

#[derive(Debug, Clone)]
struct TagCollection {
    tags: HashMap<String, Tag>,
    tag_grammar: ExactStringPrimitive,
}

impl TagCollection {
    fn new() -> Self {
        Self {
            tags: HashMap::new(),
            tag_grammar: ExactStringPrimitive::default(),
        }
    }
    pub fn create_from_string(input: &str) -> Self {
        let mut tags = HashMap::new();
        let mut tag_grammar = ExactStringPrimitive::default();
        for line in input.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            let mut current_tag = Tag::new(&parts[0]);

            for &part in &parts[1..] {
                current_tag.add_child_tag(part);
            }
            tag_grammar.add_string_to_allowed(current_tag.name.clone());
            tags.insert(current_tag.name.clone(), current_tag);
        }

        Self { tags, tag_grammar }
    }

    fn add_tag(&mut self, name: &str) -> &mut Tag {
        let name = name.trim().to_lowercase();
        self.tag_grammar.add_string_to_allowed(&name);
        self.tags
            .entry(name.to_owned())
            .or_insert_with(|| Tag::new(&name))
    }

    fn remove_tag(&mut self, name: &str) -> crate::Result<()> {
        match self.tags.remove(name) {
            Some(_) => (),
            None => crate::bail!("Tag not found."),
        }
        self.tag_grammar.remove_string_from_allowed(&name);
        Ok(())
    }

    fn get_tag(&self, name: &str) -> Option<&Tag> {
        self.tags.get(name)
    }

    fn get_tags(&self) -> Vec<&Tag> {
        self.tags.values().collect()
    }

    fn get_tag_names(&self) -> Vec<&str> {
        self.tags.keys().map(|s| s.as_str()).collect()
    }

    fn grammar(&self) -> Grammar {
        self.tag_grammar.grammar()
    }
}

#[derive(Debug, Clone)]
pub struct Tag {
    pub name: String,
    tags: TagCollection,
}

impl Tag {
    fn new(name: &str) -> Self {
        Tag {
            name: name.trim().to_lowercase(),
            tags: TagCollection::new(),
        }
    }

    fn add_child_tag(&mut self, name: &str) -> &mut Self {
        self.tags.add_tag(name)
    }

    pub fn remove_child_tag(&mut self, name: &str) -> crate::Result<()> {
        self.tags.remove_tag(name)
    }

    pub fn get_child_tags(&self) -> Vec<&Tag> {
        self.tags.get_tags()
    }

    pub fn get_child_tag_names(&self) -> Vec<&str> {
        self.tags.get_tag_names()
    }

    pub fn grammar(&self) -> Grammar {
        self.tags.grammar()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_tag_collection() -> TagCollection {
        let input = "\
terrestrial
terrestrial:soil
terrestrial:arid
aquatic
aquatic:fresh water
aquatic:fresh water:lake
bed
host-associated:animal
host:digestive
tract:mouth
salinity:low
salinity
age group:infant";
        TagCollection::create_from_string(input)
    }

    #[test]
    fn test_tag_collection_creation() {
        let tags = create_sample_tag_collection();

        // Test root tags
        assert!(tags.get_tag("terrestrial").is_some());
        assert!(tags.get_tag("aquatic").is_some());
        assert!(tags.get_tag("host-associated").is_some());
        assert!(tags.get_tag("salinity").is_some());
        assert!(tags.get_tag("age group").is_some());

        // Test non-existent tags
        assert!(tags.get_tag("non-existent").is_none());

        // Test tag names
        assert_eq!(tags.get_tag("terrestrial").unwrap().name, "terrestrial");

        // Test number of root tags
        assert_eq!(tags.get_tags().len(), 7);
    }

    #[test]
    fn test_child_tags() {
        let tags = create_sample_tag_collection();

        // Test child tags
        let terrestrial = tags.get_tag("terrestrial").unwrap();
        assert_eq!(terrestrial.get_child_tag_names(), vec!["soil", "arid"]);

        let aquatic = tags.get_tag("aquatic").unwrap();
        assert_eq!(aquatic.get_child_tag_names(), vec!["fresh water"]);

        let fresh_water = aquatic.tags.get_tag("fresh water").unwrap();
        assert_eq!(fresh_water.get_child_tag_names(), vec!["lake"]);
    }
}
