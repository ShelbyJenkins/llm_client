use llm_devices::devices::cpu::CpuConfig;
use llm_devices::devices::DeviceConfig;

pub struct LlamaCppServerConfig {
    /// -t, --threads
    /// The number of threads to use during generation (default: -1) (env: LLAMA_ARG_THREADS)
    /// - N
    threads: Option<Threads>,
    /// The number of threads to use during batch and prompt processing (default: same as --threads)
    /// - N
    threads_batch: Option<ThreadsBatch>,
    /// -ngl, --gpu-layers, --n-gpu-layers
    /// The number of layers to store in VRAM (env: LLAMA_ARG_N_GPU_LAYERS)
    /// - N
    n_gpu_layers: Option<NGpuLayers>,
    //// -sm, --split-mode
    /// how to split the model across multiple GPUs, one of:
    /// - none: use one GPU only
    /// - layer (default): split layers and KV across GPUs
    /// - row: split rows across GPUs
    split_mode: Option<SplitMode>,
    /// -ts, --tensor-split
    /// The fraction of the model to offload to each GPU
    /// - comma-separated list of proportions, e.g. 3,1
    tensor_split: Option<TensorSplit>,
    /// -mg, --main-gpu
    /// The GPU to use for the model (with split-mode = none),
    /// or for intermediate results and KV (with split-mode = row)
    /// (default: 0)
    /// - N
    main_gpu: Option<MainGpu>,
    /// -nkvo, --no-kv-offload 	disable KV offload
    /// Used when no GPUs are available
    no_kv_offload: Option<NoKvOffload>,
}

impl Default for LlamaCppServerConfig {
    fn default() -> Self {
        Self {
            threads: None,
            threads_batch: None,
            n_gpu_layers: None,
            split_mode: None,
            tensor_split: None,
            main_gpu: None,
            no_kv_offload: None,
        }
    }
}

impl LlamaCppServerConfig {
    pub fn new(device_config: &DeviceConfig) -> crate::Result<Self> {
        match device_config.gpu_count() {
            0 => Self::new_only_cpu(device_config),
            1 => Self::new_single_gpu(device_config),
            _ => Self::new_multiple_gpu(device_config),
        }
    }

    fn new_only_cpu(device_config: &DeviceConfig) -> crate::Result<Self> {
        Ok(Self {
            threads: Some(Threads::new_from_cpu_config(&device_config.cpu_config)),
            threads_batch: Some(ThreadsBatch::new_from_cpu_config(&device_config.cpu_config)),
            n_gpu_layers: Some(NGpuLayers(0)),
            no_kv_offload: Some(NoKvOffload),
            ..Default::default()
        })
    }

    fn new_single_gpu(device_config: &DeviceConfig) -> crate::Result<Self> {
        let gpu_devices = device_config.allocate_layers_to_gpus(1, 1)?;
        let layer_count = gpu_devices.iter().map(|d| d.allocated_layers).sum();
        Ok(Self {
            threads_batch: Some(ThreadsBatch::new_from_cpu_config(&device_config.cpu_config)),
            split_mode: Some(SplitMode::None),
            n_gpu_layers: Some(NGpuLayers(layer_count)),
            main_gpu: Some(MainGpu(device_config.main_gpu()?)),
            ..Default::default()
        })
    }

    fn new_multiple_gpu(device_config: &DeviceConfig) -> crate::Result<Self> {
        let gpu_devices = device_config.allocate_layers_to_gpus(1, 1)?;
        let layer_count = gpu_devices.iter().map(|d| d.allocated_layers).sum();
        Ok(Self {
            threads_batch: Some(ThreadsBatch::new_from_cpu_config(&device_config.cpu_config)),
            split_mode: Some(SplitMode::Layer),
            main_gpu: Some(MainGpu(device_config.main_gpu()?)),
            n_gpu_layers: Some(NGpuLayers(layer_count)),
            ..Default::default()
        })
    }

    pub(crate) fn populate_args(&self, command: &mut std::process::Command) {
        if let Some(threads) = &self.threads {
            command.args(threads.as_arg());
        }
        if let Some(threads_batch) = &self.threads_batch {
            command.args(threads_batch.as_arg());
        }
        if let Some(n_gpu_layers) = &self.n_gpu_layers {
            command.args(n_gpu_layers.as_arg());
        }
        if let Some(split_mode) = &self.split_mode {
            command.args(split_mode.as_arg());
        }
        if let Some(tensor_split) = &self.tensor_split {
            if !tensor_split.0.is_empty() {
                command.args(tensor_split.as_arg());
            }
        }
        if let Some(main_gpu) = &self.main_gpu {
            command.args(main_gpu.as_arg());
        }
        if let Some(no_kv_offload) = &self.no_kv_offload {
            command.arg(no_kv_offload.as_arg());
        }
    }
}

pub(crate) struct Threads(pub i16);
impl Threads {
    fn new_from_cpu_config(cpu_config: &CpuConfig) -> Self {
        Self(cpu_config.thread_count_or_default())
    }
    fn as_arg(&self) -> [String; 2] {
        ["--threads".to_string(), self.0.to_string()]
    }
}

pub(crate) struct ThreadsBatch(pub i16);
impl ThreadsBatch {
    fn new_from_cpu_config(cpu_config: &CpuConfig) -> Self {
        Self(cpu_config.thread_count_batch_or_default())
    }
    fn as_arg(&self) -> [String; 2] {
        ["--threads-batch".to_string(), self.0.to_string()]
    }
}

pub(crate) struct NGpuLayers(pub u64);
impl NGpuLayers {
    fn as_arg(&self) -> [String; 2] {
        ["--n-gpu-layers".to_string(), self.0.to_string()]
    }
}

#[allow(dead_code)]
pub(crate) enum SplitMode {
    None,
    Layer,
    Row,
}

impl SplitMode {
    fn as_arg(&self) -> [String; 2] {
        match self {
            Self::None => ["--split-mode".to_string(), "none".to_string()],
            Self::Layer => ["--split-mode".to_string(), "layer".to_string()],
            Self::Row => ["--split-mode".to_string(), "row".to_string()],
        }
    }
}

pub struct TensorSplit(pub Vec<char>);

impl TensorSplit {
    fn as_arg(&self) -> [String; 2] {
        [
            "--tensor-split".to_string(),
            self.0
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ]
    }
}

pub(crate) struct MainGpu(pub u32);

impl MainGpu {
    fn as_arg(&self) -> [String; 2] {
        ["--main-gpu".to_string(), self.0.to_string()]
    }
}

pub(crate) struct NoKvOffload;

impl NoKvOffload {
    fn as_arg(&self) -> String {
        format!("--no-kv-offload")
    }
}
