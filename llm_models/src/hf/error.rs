use crate::hf::id::{
    HfNameSpaceError, HfRFilenameError, HfRepoIdError, HfRepoNameError, HfRepoShaError,
};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HfError {
    /* ────── validation & parsing ────── */
    #[error(transparent)]
    NameSpace(#[from] HfNameSpaceError),

    #[error(transparent)]
    RepoName(#[from] HfRepoNameError),

    #[error(transparent)]
    RepoId(#[from] HfRepoIdError),

    #[error(transparent)]
    Sha(#[from] HfRepoShaError),

    #[error(transparent)]
    RFilename(#[from] HfRFilenameError),

    /* ────── hub / network / fs ────── */
    #[error(transparent)]
    Api(#[from] hf_hub::api::sync::ApiError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("Cache lock at {path:?} appears stale")]
    StaleLock { path: std::path::PathBuf },

    #[error("Unsupported model format '{format}' in file '{file}'")]
    UnsupportedFormat { format: String, file: String },

    /* ────── schema / protocol drift ────── */
    #[error("Unexpected field '{field}' in Hub response")]
    UnknownField { field: String },

    /* ────── catch-all ────── */
    #[error("Internal invariant violated: {0}")]
    Internal(String),
}
