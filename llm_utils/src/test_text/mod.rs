use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, sync::LazyLock};

macro_rules! generate_splitting_structs {
    ($($file:ident),+) => {
        pub struct Splitting {
            $(
                pub $file: $file,
            )+
        }
        $(
            #[allow(non_camel_case_types)]
            pub struct $file {
                pub content: String,
                pub cases: Vec<String>,

            }
            impl $file {
                fn load_splitting_texts() -> Self {
                    let file_name = format!("{}.json", stringify!($file).to_lowercase());
                    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                    let file_path = PathBuf::from(cargo_manifest_dir)
                        .join("src")
                        .join("test_text")
                        .join("splitting")
                        .join(file_name);
                    let content = fs::read_to_string(&file_path).expect("Failed to read file");
                    let data: serde_json::Value =
                        serde_json::from_str(&content).expect("Failed to parse JSON");


                    Self {
                        content: data["content"].as_str().unwrap().to_string(),
                        cases: data["cases"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|s| s.as_str().unwrap().to_string())
                            .collect(),
                    }
                }
            }
        )+
        impl Splitting {
            fn load_splitting_texts() -> Self {
                Self {
                    $(
                        $file: $file::load_splitting_texts(),
                    )+
                }
            }
        }
    };
}

generate_splitting_structs!(
    graphemes_unicode,
    sentences_rule_1,
    sentences_rule_2,
    sentences_rule_3,
    sentences_rule_4,
    sentences_unicode,
    single_eol,
    two_plus_eol,
    words_unicode,
    joining
);

#[derive(Serialize, Deserialize, Clone)]
pub struct ChunkingTestCase {
    pub first: Option<String>,
    pub last: Option<String>,
}

impl ChunkingTestCase {
    pub fn first(&self) -> &str {
        self.first.as_ref().unwrap()
    }
    pub fn last(&self) -> &str {
        self.last.as_ref().unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChunkingTestCases {
    pub case_64: Option<ChunkingTestCase>,
    pub case_128: Option<ChunkingTestCase>,
    pub case_256: Option<ChunkingTestCase>,
    pub case_512: Option<ChunkingTestCase>,
    pub case_768: Option<ChunkingTestCase>,
    pub case_1024: Option<ChunkingTestCase>,
    pub case_1536: Option<ChunkingTestCase>,
    pub case_2048: Option<ChunkingTestCase>,
}

impl ChunkingTestCases {
    pub fn case(&self, case: u32) -> ChunkingTestCase {
        match case {
            64 => self.case_64.clone().unwrap(),
            128 => self.case_128.clone().unwrap(),
            256 => self.case_256.clone().unwrap(),
            512 => self.case_512.clone().unwrap(),
            768 => self.case_768.clone().unwrap(),
            1024 => self.case_1024.clone().unwrap(),
            1536 => self.case_1536.clone().unwrap(),
            2048 => self.case_2048.clone().unwrap(),
            _ => panic!("Invalid case"),
        }
    }
}

macro_rules! generate_chunking_structs {
    ($($file:ident),+) => {
        pub struct Chunking {
            $(
                pub $file: $file,
            )+
        }
        $(
            #[allow(non_camel_case_types)]
            #[derive(Serialize, Deserialize)]
            pub struct $file {
                pub content: String,
                pub test_cases: ChunkingTestCases,

            }
            impl $file {
                fn load_chunking_texts() -> Self {
                    let file_name = format!("{}.json", stringify!($file).to_lowercase());
                    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                    let file_path = PathBuf::from(cargo_manifest_dir)
                        .join("src")
                        .join("test_text")
                        .join("chunking")
                        .join(file_name);
                    let content = fs::read_to_string(&file_path).expect("Failed to read file");
                    serde_json::from_str(&content).expect("Failed to parse JSON")
                }

            }
        )+
        impl Chunking {
            fn load_chunking_texts() -> Self {
                Self {
                    $(
                        $file: $file::load_chunking_texts(),
                    )+
                }
            }
        }
    };
}

generate_chunking_structs!(chunking_tiny, chunking_small);

macro_rules! generate_text_structs {
    ($($file:ident),+) => {
        pub struct Text {
            $(
                pub $file: $file,
            )+
        }
        $(
            #[allow(non_camel_case_types)]
            pub struct $file {
                pub content: String,
            }
            impl $file {
                fn load_text() -> Self {
                    let file_name = format!("{}.txt", stringify!($file).to_lowercase());
                    let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
                    let file_path = PathBuf::from(cargo_manifest_dir)
                        .join("src")
                        .join("test_text")
                        .join("text")
                        .join(file_name);
                    let content = fs::read_to_string(&file_path).expect("Failed to read file");
                    Self { content }
                }
            }
        )+
        impl Text {
            fn load_text() -> Self {
                Self {
                    $(
                        $file: $file::load_text(),
                    )+
                }
            }
        }
    };
}

generate_text_structs!(
    smollest,
    tiny,
    small,
    medium,
    long,
    really_long,
    html_short,
    many_subjects,
    one_subject_many_topics,
    one_subject_one_topic
);

pub static SPLIT_TESTS: LazyLock<Splitting> = LazyLock::new(Splitting::load_splitting_texts);
pub static CHUNK_TESTS: LazyLock<Chunking> = LazyLock::new(Chunking::load_chunking_texts);
pub static TEXT: LazyLock<Text> = LazyLock::new(Text::load_text);
