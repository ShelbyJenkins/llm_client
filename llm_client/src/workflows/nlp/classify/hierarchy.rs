use std::collections::HashMap;

use llm_utils::grammar::Grammar;

use crate::{primitives::ExactStringPrimitive, PrimitiveTrait};

pub fn create_from_string(input: &str) -> TagCollection {
    let mut tag_collection = TagCollection::new();
    for line in input.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        let current_tag = tag_collection.new_tag(parts[0]);

        for &part in &parts[1..] {
            current_tag.add_child_tag(part);
        }
    }

    tag_collection
}

#[derive(Debug, Clone)]
pub struct TagCollection {
    tags: HashMap<String, Tag>,
    tag_grammar: ExactStringPrimitive,
}

impl TagCollection {
    pub fn new() -> Self {
        Self {
            tags: HashMap::new(),
            tag_grammar: ExactStringPrimitive::default(),
        }
    }

    pub fn create_from_string(input: &str) -> TagCollection {
        let mut tag_collection = TagCollection::new();

        for line in input.lines() {
            Self::add_tag_recursive(&mut tag_collection, line);
        }

        tag_collection
    }

    fn add_tag_recursive(tag_collection: &mut TagCollection, tag_string: &str) {
        match tag_string.split_once(':') {
            Some((parent, child)) => {
                let parent_tag = tag_collection.new_tag(parent);
                Self::add_tag_recursive(&mut parent_tag.tags, child);
            }
            None => {
                tag_collection.new_tag(tag_string);
            }
        }
    }

    pub fn new_tag(&mut self, name: &str) -> &mut Tag {
        let name = name
            .split_whitespace()
            .map(str::trim)
            .collect::<Vec<&str>>()
            .join("-")
            .to_lowercase();

        self.add_tag(Tag::new(&name))
    }

    pub fn add_tag(&mut self, tag: Tag) -> &mut Tag {
        if !self.tags.contains_key(&tag.name) {
            self.tag_grammar.add_string_to_allowed(&tag.name);
        }

        self.tags.entry(tag.name.clone()).or_insert_with(|| tag)
    }

    pub fn remove_tag(&mut self, name: &str) -> crate::Result<Tag> {
        let tag = match self.tags.remove(name) {
            Some(tag) => tag,
            None => crate::bail!("Tag not found."),
        };
        self.tag_grammar.remove_string_from_allowed(&name);
        Ok(tag)
    }

    pub fn get_tag(&self, name: &str) -> Option<&Tag> {
        self.tags.get(name)
    }

    pub fn get_tags(&self) -> Vec<&Tag> {
        let mut tags: Vec<&Tag> = self.tags.values().collect();
        tags.sort_by(|a, b| a.name.cmp(&b.name));
        tags
    }

    pub fn get_tag_names(&self) -> Vec<&str> {
        self.get_tags()
            .iter()
            .map(|tag| tag.name.as_str())
            .collect()
    }

    pub fn grammar(&self) -> Grammar {
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
        self.tags.new_tag(name)
    }

    pub fn remove_child_tag(&mut self, name: &str) -> crate::Result<Tag> {
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
        assert!(tags.get_tag("age-group").is_some());

        // Test non-existent tags
        assert!(tags.get_tag("non-existent").is_none());

        // Test tag names
        assert_eq!(tags.get_tag("terrestrial").unwrap().name, "terrestrial");

        // Test number of root tags
        assert_eq!(tags.get_tags().len(), 8);
    }

    #[test]
    fn test_white_space() {
        let tags = create_sample_tag_collection();

        // Test white space
        assert!(tags.get_tag("age-group").is_some());
        assert!(tags.get_tag("age group").is_none());
        let age_group = tags.get_tag("age-group").unwrap();
        assert_eq!(
            age_group.get_child_tag_names(),
            vec!["infant", "old-age", "young"]
        );
        let young = age_group.tags.get_tag("young").unwrap();
        assert_eq!(young.get_child_tag_names(), vec!["new-born"]);
    }

    #[test]
    fn test_child_tags() {
        let tags = create_sample_tag_collection();

        // Test child tags
        let terrestrial = tags.get_tag("terrestrial").unwrap();
        assert_eq!(terrestrial.get_child_tag_names(), vec!["arid", "soil"]);

        let aquatic = tags.get_tag("aquatic").unwrap();
        assert_eq!(aquatic.get_child_tag_names(), vec!["fresh-water"]);

        let fresh_water = aquatic.tags.get_tag("fresh-water").unwrap();
        assert_eq!(fresh_water.get_child_tag_names(), vec!["lake"]);
        let bed = tags.get_tag("bed").unwrap();
        assert_eq!(bed.get_child_tag_names(), vec!["home", "hotel",]);
        let hotel = bed.tags.get_tag("hotel").unwrap();
        assert_eq!(hotel.get_child_tag_names(), vec!["king", "queen"]);
        let home = bed.tags.get_tag("home").unwrap();
        assert_eq!(home.get_child_tag_names(), vec!["double"]);
        let double = home.tags.get_tag("double").unwrap();
        assert_eq!(double.get_child_tag_names(), vec!["bunk"]);
    }
}
