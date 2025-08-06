use std::{
    collections::BTreeMap,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    fs::file_path::ExistingFile,
    hf::id::{HfRFilename, HfRepoId},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceLocator {
    HfRepoId(HfRepoId),
    FilePath(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileLocator {
    HfRFilename(HfRFilename),
    FilePath(String),
}

impl FileLocator {
    pub fn as_str(&self) -> &str {
        match self {
            FileLocator::HfRFilename(rfilename) => rfilename.as_str(),
            FileLocator::FilePath(path) => path,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CheckpointFiles {
    Single(CheckpointFile),

    Sharded {
        total: NonZeroU32,
        parts: BTreeMap<NonZeroU32, CheckpointFile>,
    },
}

impl CheckpointFiles {
    pub fn first_file(&self) -> &CheckpointFile {
        match self {
            CheckpointFiles::Single(file) => file,
            CheckpointFiles::Sharded { parts, .. } => parts
                .values()
                .next()
                .expect("Sharded files should have at least one part"),
        }
    }

    pub fn total_file_size_bytes(&self) -> u64 {
        match self {
            CheckpointFiles::Single(file) => file.total_file_size_bytes,
            CheckpointFiles::Sharded { parts, .. } => {
                parts.values().map(|f| f.total_file_size_bytes).sum()
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointFile {
    pub file_locator: FileLocator,
    pub total_file_size_bytes: u64,
    pub status: FileStatus,
}

impl CheckpointFile {
    pub fn new(file_locator: FileLocator, total_file_size_bytes: u64, status: FileStatus) -> Self {
        Self {
            file_locator,
            total_file_size_bytes,
            status,
        }
    }

    pub fn as_str(&self) -> &str {
        self.file_locator.as_str()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileStatus {
    PartiallyDownloaded {
        path: ExistingFile,
        downloaded_size: u64,
    },
    FullyDownloaded {
        path: ExistingFile,
    },
    NotDownloaded,
}

impl FileStatus {
    pub fn new(path: Option<PathBuf>, total_file_size_bytes: u64) -> Self {
        if let Some(path) = path {
            let initial_size = std::fs::metadata(&path).unwrap().len();
            std::thread::sleep(std::time::Duration::from_millis(100));
            let final_size = std::fs::metadata(&path).unwrap().len();
            if initial_size == final_size && final_size == total_file_size_bytes {
                FileStatus::FullyDownloaded {
                    path: ExistingFile::try_new(path).unwrap(),
                }
            } else {
                FileStatus::PartiallyDownloaded {
                    path: ExistingFile::try_new(path).unwrap(),
                    downloaded_size: final_size as u64,
                }
            }
        } else {
            FileStatus::NotDownloaded
        }
    }

    pub fn path(&self) -> Option<&Path> {
        match self {
            FileStatus::PartiallyDownloaded { path, .. } => Some(path.as_path()),
            FileStatus::FullyDownloaded { path } => Some(path.as_path()),
            FileStatus::NotDownloaded => None,
        }
    }
}
