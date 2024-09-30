use std::collections::HashMap;

#[derive(Debug)]
struct TagSystem {
    root_tags: HashMap<String, Tag>,
}

impl TagSystem {
    pub fn new() -> Self {
        TagSystem {
            root_tags: HashMap::new(),
        }
    }

    pub fn create_from_string(&mut self, input: &str) {
        for line in input.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            let mut current_tag = self.create_parent_tag(parts[0]);

            for &part in &parts[1..] {
                current_tag = current_tag.add_child_tag(part);
            }
        }
    }

    pub fn create_parent_tag(&mut self, name: &str) -> &mut Tag {
        self.root_tags
            .entry(name.to_owned())
            .or_insert_with(|| Tag::new(name))
    }

    pub fn get_tag(&self, path: &str) -> Option<&Tag> {
        let parts: Vec<&str> = path.split(':').collect();
        let mut current_tag = self.root_tags.get(parts[0])?;

        for &part in &parts[1..] {
            current_tag = current_tag.child_tags.get(part)?;
        }

        Some(current_tag)
    }

    pub fn get_tags(&self) -> Vec<&Tag> {
        self.root_tags.values().collect()
    }
}

#[derive(Debug)]
struct Tag {
    name: String,
    child_tags: HashMap<String, Tag>,
}

impl Tag {
    fn new(name: &str) -> Self {
        Tag {
            name: name.to_lowercase(),
            child_tags: HashMap::new(),
        }
    }

    fn add_child_tag(&mut self, name: &str) -> &mut Self {
        self.child_tags.insert(name.to_owned(), Tag::new(name));
        self.child_tags.get_mut(name).unwrap()
    }

    pub fn get_tags(&self) -> Vec<&Tag> {
        self.child_tags.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_system() {
        let mut tag_system = TagSystem::new();
        let input = "\
terrestrial
terrestrial:soil
terrestrial:arid
aquatic
aquatic:fresh water
aquatic:fresh water:lake bed
host-associated:animal host:digestive tract:mouth
salinity:low salinity
age group:infant";

        tag_system.create_from_string(input);
        let tags = tag_system.get_tags();
        println!("{:?}", tags);
        for t in tags {
            println!("{:?}", t.get_tags());
        }

        // Test root tags
        assert!(tag_system.get_tag("terrestrial").is_some());
        assert!(tag_system.get_tag("aquatic").is_some());
        assert!(tag_system.get_tag("host-associated").is_some());
        assert!(tag_system.get_tag("salinity").is_some());
        assert!(tag_system.get_tag("age group").is_some());

        // Test nested tags
        assert!(tag_system.get_tag("terrestrial:soil").is_some());
        assert!(tag_system.get_tag("aquatic:fresh water:lake bed").is_some());
        assert!(tag_system
            .get_tag("host-associated:animal host:digestive tract:mouth")
            .is_some());

        // Test non-existent tags
        assert!(tag_system.get_tag("non-existent").is_none());
        assert!(tag_system.get_tag("terrestrial:non-existent").is_none());

        // Test tag names
        assert_eq!(
            tag_system.get_tag("terrestrial").unwrap().name,
            "terrestrial"
        );
        assert_eq!(
            tag_system.get_tag("aquatic:fresh water").unwrap().name,
            "fresh water"
        );
        assert_eq!(
            tag_system.get_tag("age group:infant").unwrap().name,
            "infant"
        );
    }
}
