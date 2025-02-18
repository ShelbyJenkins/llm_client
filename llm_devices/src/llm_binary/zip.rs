pub(super) fn get_zip_file_path() -> crate::Result<std::path::PathBuf> {
    Ok(super::get_target_dir_sub_dir()?.join("llama.zip"))
}

pub(super) fn extract_zip() -> crate::Result<()> {
    let file = std::fs::File::open(get_zip_file_path()?)?;
    let mut archive = zip::ZipArchive::new(file)?;
    crate::trace!(
        "Extracting {} to {}",
        get_zip_file_path()?.display(),
        super::get_target_dir_sub_dir()?.display()
    );

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let is_dir = file.name().ends_with('/');

        let outpath = match file.enclosed_name() {
            Some(path) => {
                // Skip the first directory component
                let components: Vec<_> = path.components().collect();
                if components.len() > 1 {
                    // Join all components except the first one
                    super::get_target_dir_sub_dir()?
                        .join(components[1..].iter().collect::<std::path::PathBuf>())
                } else {
                    continue; // Skip top-level directory itself
                }
            }
            None => continue,
        };

        if is_dir {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p)?;
                }
            }
            let mut outfile = std::fs::File::create(&outpath).map_err(|e| {
                crate::anyhow!(
                    "Failed to create file {}: {}",
                    outpath.display(),
                    e.to_string()
                )
            })?;

            std::io::copy(&mut file, &mut outfile).map_err(|e| {
                crate::anyhow!(
                    "Failed to copy file from archive to {}: {}",
                    outpath.display(),
                    e.to_string()
                )
            })?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if let Some(mode) = file.unix_mode() {
                std::fs::set_permissions(&outpath, std::fs::Permissions::from_mode(mode))?;
            }
        }
    }
    crate::trace!("Extraction successful");
    Ok(())
}
