use mistralrs::{DeviceLayerMapMetadata, DeviceMapMetadata};
use nvml_wrapper::Nvml;

pub fn get_device_map(num_hidden_layers: u64) -> DeviceMapMetadata {
    match try_cuda(num_hidden_layers) {
        Ok(device_map) => device_map,
        Err(e) => {
            let e = format!("Error getting device map: {}", e);
            eprintln!("{e}");
            crate::info!(e);
            DeviceMapMetadata::dummy()
        }
    }
}

fn try_cuda(num_hidden_layers: u64) -> crate::Result<DeviceMapMetadata> {
    let nvml = init_nvml_wrapper()?;

    let device_count = nvml.device_count()?;
    if device_count == 0 {
        crate::bail!("No CUDA devices found");
    }
    let layer_per_device = (num_hidden_layers as f64 / device_count as f64).ceil() as usize;
    let mut device_map_metadata: Vec<DeviceLayerMapMetadata> = Vec::new();
    for i in 0..device_count {
        device_map_metadata.push(DeviceLayerMapMetadata {
            ordinal: i as usize,
            layers: layer_per_device,
        });
    }
    let device_info = format!(
        "Device count: {}, Layer per device: {}",
        device_count, layer_per_device
    );
    crate::info!("{device_info}");
    println!("{device_info}");
    Ok(DeviceMapMetadata::from_num_device_layers(
        device_map_metadata,
    ))
}

fn init_nvml_wrapper() -> crate::Result<Nvml> {
    let library_names = vec![
        "libnvidia-ml.so",   // For Linux
        "libnvidia-ml.so.1", // For WSL
        "nvml.dll",          // For Windows
    ];
    for library_name in library_names {
        match Nvml::builder().lib_path(library_name.as_ref()).init() {
            Ok(nvml) => return Ok(nvml),
            Err(e) => {
                crate::info!("Error initializing nvml_wrapper::Nvml: {}", e);
                continue;
            }
        }
    }
    crate::bail!("Failed to initialize nvml_wrapper::Nvml")
}
