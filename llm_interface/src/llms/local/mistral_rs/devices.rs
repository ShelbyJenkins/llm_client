use mistralrs::{Device, DeviceLayerMapMetadata, DeviceMapMetadata};

pub fn mistral_rs_device_map(
    generic_device_map: &crate::llms::local::devices::DeviceConfig,
) -> crate::Result<(Device, DeviceMapMetadata)> {
    match generic_device_map.gpu_count() {
        0 => new_only_cpu(generic_device_map),
        1 => new_single_gpu(generic_device_map),
        _ => new_multiple_gpu(generic_device_map),
    }
}

fn new_only_cpu(
    generic_device_map: &crate::llms::local::devices::DeviceConfig,
) -> crate::Result<(Device, DeviceMapMetadata)> {
    std::env::set_var(
        "RAYON_NUM_THREADS",
        generic_device_map
            .cpu_config
            .set_default_thread_count(generic_device_map.cpu_config.threads, 1.0)
            .to_string(),
    );
    Ok((Device::Cpu, DeviceMapMetadata::dummy()))
}

fn new_single_gpu(
    generic_device_map: &crate::llms::local::devices::DeviceConfig,
) -> crate::Result<(Device, DeviceMapMetadata)> {
    let _gpu_devices = generic_device_map.allocate_layers_to_gpus(0, 0)?;
    let main_gpu = generic_device_map.main_gpu()?;
    // let layer_count = gpu_devices
    //     .iter()
    //     .map(|d| d.allocated_layers as usize)
    //     .sum();

    Ok((
        Device::cuda_if_available(main_gpu as usize)?,
        DeviceMapMetadata::dummy(),
    ))
    // Ok((
    //     Device::cuda_if_available(main_gpu as usize)?,
    //     DeviceMapMetadata::from_num_device_layers(vec![DeviceLayerMapMetadata {
    //         ordinal: main_gpu as usize,
    //         layers: layer_count,
    //     }]),
    // ))
}

fn new_multiple_gpu(
    generic_device_map: &crate::llms::local::devices::DeviceConfig,
) -> crate::Result<(Device, DeviceMapMetadata)> {
    let gpu_devices = generic_device_map.allocate_layers_to_gpus(0, 0)?;
    let main_gpu = generic_device_map.main_gpu()?;

    let mut device_map_metadata: Vec<DeviceLayerMapMetadata> = Vec::new();
    for gpu in &gpu_devices {
        device_map_metadata.push(DeviceLayerMapMetadata {
            ordinal: gpu.ordinal as usize,
            layers: gpu.allocated_layers as usize,
        });
    }
    Ok((
        Device::cuda_if_available(main_gpu as usize)?,
        DeviceMapMetadata::from_num_device_layers(device_map_metadata),
    ))
}
