use super::local_content::load_content_path;
use super::TextConcatenator;
use std::{collections::HashMap, path::PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub enum PromptMessageType {
    System,
    User,
    Assistant,
}

impl PromptMessageType {
    pub fn as_str(&self) -> &str {
        match self {
            PromptMessageType::System => "system",
            PromptMessageType::User => "user",
            PromptMessageType::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PromptMessage {
    content: std::cell::RefCell<Vec<String>>,
    pub built_message_hashmap: std::cell::RefCell<HashMap<String, String>>,
    pub built_message_string: std::cell::RefCell<Option<String>>,
    pub message_type: PromptMessageType,
    pub concatenator: TextConcatenator,
}

impl PromptMessage {
    pub fn new(message_type: PromptMessageType, concatenator: &TextConcatenator) -> Self {
        Self {
            content: Vec::new().into(),
            built_message_hashmap: HashMap::new().into(),
            built_message_string: None.into(),
            message_type,
            concatenator: concatenator.clone(),
        }
    }
    // Setter functions
    pub fn set_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        if self
            .content_ref()
            .first()
            .map_or(true, |first| first != content.as_ref())
        {
            self.built_message_hashmap_mut().clear();
            *self.built_message_string_mut() = None;
            *self.content_mut() = vec![content.as_ref().to_owned()];
        }

        self
    }

    pub fn set_content_from_path(&self, content_path: &PathBuf) -> &Self {
        self.set_content(load_content_path(content_path))
    }

    pub fn prepend_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        if self
            .content_ref()
            .first()
            .map_or(true, |first| first != content.as_ref())
        {
            self.built_message_hashmap_mut().clear();
            *self.built_message_string_mut() = None;
            self.content_mut().insert(0, content.as_ref().to_owned());
        }

        self
    }

    pub fn prepend_content_from_path(&self, content_path: &PathBuf) -> &Self {
        self.prepend_content(load_content_path(content_path))
    }

    pub fn append_content<T: AsRef<str>>(&self, content: T) -> &Self {
        if content.as_ref().is_empty() {
            return self;
        }

        if self
            .content_ref()
            .last()
            .map_or(true, |last| last != content.as_ref())
        {
            self.built_message_hashmap_mut().clear();
            *self.built_message_string_mut() = None;
            self.content_mut().push(content.as_ref().to_owned());
        }

        self
    }

    pub fn append_content_from_path(&self, content_path: &PathBuf) -> &Self {
        self.append_content(load_content_path(content_path))
    }

    // Getter functions
    pub fn get_built_message_string(&self) -> Option<String> {
        if self.built_message_string_ref().is_none() {
            self.build();
        }
        self.built_message_string_ref().clone()
    }

    // Builder functions
    pub fn requires_build(&self) -> bool {
        !self.content_ref().is_empty() && self.built_message_hashmap_ref().is_empty()
    }

    pub fn build(&self) {
        if let Some(built_message_string) = self.build_prompt_string() {
            *self.built_message_string_mut() = Some(built_message_string.clone());
            *self.built_message_hashmap_mut() = HashMap::from([
                ("role".to_string(), self.message_type.as_str().to_owned()),
                ("content".to_string(), built_message_string.to_owned()),
            ]);
        }
    }

    fn build_prompt_string(&self) -> Option<String> {
        if self.content_ref().is_empty() {
            return None;
        };
        let mut built_message_string = String::new();

        for c in self.content_ref().iter() {
            if c.as_str().is_empty() {
                continue;
            }
            if !built_message_string.is_empty() {
                built_message_string.push_str(self.concatenator.as_str());
            }
            built_message_string.push_str(c.as_str());
        }
        if built_message_string.is_empty() {
            return None;
        }
        Some(built_message_string)
    }

    // Helper functions
    fn content_ref(&self) -> std::cell::Ref<Vec<String>> {
        self.content.borrow()
    }

    fn content_mut(&self) -> std::cell::RefMut<Vec<String>> {
        self.content.borrow_mut()
    }

    fn built_message_hashmap_ref(&self) -> std::cell::Ref<HashMap<String, String>> {
        self.built_message_hashmap.borrow()
    }

    fn built_message_hashmap_mut(&self) -> std::cell::RefMut<HashMap<String, String>> {
        self.built_message_hashmap.borrow_mut()
    }

    fn built_message_string_ref(&self) -> std::cell::Ref<Option<String>> {
        self.built_message_string.borrow()
    }

    fn built_message_string_mut(&self) -> std::cell::RefMut<Option<String>> {
        self.built_message_string.borrow_mut()
    }
}

pub(crate) fn build_messages(messages: &mut [PromptMessage]) -> Vec<HashMap<String, String>> {
    let mut prompt_messages: Vec<HashMap<String, String>> = Vec::new();
    let mut last_message_type = None;
    for (i, message) in messages.iter_mut().enumerate() {
        let message_type = &message.message_type;
        // Rule 1: System message can only be the first message
        if *message_type == PromptMessageType::System && i != 0 {
            panic!("System message can only be the first message.");
        }
        // Rule 2: First message must be either System or User
        if i == 0
            && *message_type != PromptMessageType::System
            && *message_type != PromptMessageType::User
        {
            panic!("Conversation must start with either a System or User message.");
        }
        // Rule 3: Ensure alternating User/Assistant messages after the first message
        if i > 0 {
            match (last_message_type, message_type) {
                (Some(PromptMessageType::User), PromptMessageType::Assistant) => {},
                (Some(PromptMessageType::Assistant), PromptMessageType::User) => {},
                (Some(PromptMessageType::System), PromptMessageType::User) => {},
                _ => panic!("Messages must alternate between User and Assistant after the first message (which can be System)."),
            }
        }
        last_message_type = Some(message_type.clone());
        if message.requires_build() {
            message.build();
        }
        if message.built_message_hashmap_ref().is_empty() {
            eprintln!("message.built_content is empty and skipped");
            continue;
        }
        prompt_messages.push(message.built_message_hashmap_ref().clone());
    }
    prompt_messages
}

impl std::fmt::Display for PromptMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message_type = match self.message_type {
            PromptMessageType::System => "System",
            PromptMessageType::User => "User",
            PromptMessageType::Assistant => "Assistant",
        };
        let message = match self.build_prompt_string() {
            Some(built_message_string) => {
                if built_message_string.len() > 300 {
                    format!(
                        "{}...",
                        built_message_string.chars().take(300).collect::<String>()
                    )
                } else {
                    built_message_string
                }
            }
            None => "debug message: empty or unbuilt".to_string(),
        };

        writeln!(f, "\x1b[1m{message_type}\x1b[0m:\n{:?}", message)
    }
}
