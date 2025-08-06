pub mod local;
pub mod tokenizer;

// #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
// pub enum ModelSpec {
//     HfRepo {
//         name_space: String,
//         repo_name: String,
//     },
//     LocalRepo {
//         dir: StorageLocation,
//         base_name: String,
//     },
//     HfFile(Url),
//     LocalFile(ExistingFile),
// }

// #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
// pub enum ModelSmartSelectSpec {
//     AvailableMemory(u64),
//     QuantizationLevel(u8),
// }
