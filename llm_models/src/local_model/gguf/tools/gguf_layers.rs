use super::gguf_tensors::TensorInfo;
use std::collections::HashMap;

pub struct GgufLayers(pub HashMap<String, GgufLayer>);

impl GgufLayers {
    pub fn from_tensors(tensors: &Vec<TensorInfo>) -> Self {
        let mut layers: HashMap<String, GgufLayer> = HashMap::new();

        for t in tensors {
            let parts: Vec<&str> = t.name.split('.').collect();

            let (key, remaining) = if parts[0] == "blk" && parts.len() > 1 {
                // join first and second part, e.g. blk.%d
                (format!("{}.{}", parts[0], parts[1]), &parts[2..].join("."))
            } else {
                (parts[0].to_string(), &parts[1..].join("."))
            };

            layers
                .entry(key)
                .or_insert_with(|| GgufLayer {
                    name: t.name.clone(),
                    tensors: HashMap::new(),
                })
                .tensors
                .insert(remaining.to_string(), t.clone());
        }

        Self(layers)
    }

    pub fn blocks(&self) -> Self {
        Self(
            self.0
                .iter()
                .filter(|(k, _)| k.starts_with("blk"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    pub fn get(&self, key: &str) -> Option<&GgufLayer> {
        self.0.get(key)
    }

    pub fn count(&self) -> u64 {
        self.0.len() as u64
    }

    pub fn count_blocks(&self) -> u64 {
        self.0.iter().filter(|(k, _)| k.starts_with("blk")).count() as u64
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.0.iter().map(|(_, l)| l.size()).sum()
    }

    pub fn total_size_blocks_bytes(&self) -> u64 {
        self.0
            .iter()
            .filter(|(k, _)| k.starts_with("blk"))
            .map(|(_, l)| l.size())
            .sum()
    }
}

#[derive(Clone)]
pub struct GgufLayer {
    pub name: String,
    pub tensors: HashMap<String, TensorInfo>,
}

impl GgufLayer {
    pub fn get(&self, key: &str) -> Option<&TensorInfo> {
        self.tensors.get(key)
    }
    pub fn size(&self) -> u64 {
        self.tensors.iter().map(|(_, t)| t.size()).sum()
    }
}

impl std::fmt::Debug for GgufLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("GgufLayers");
        debug_struct.field("GgufLayers", &self.0.keys());
        debug_struct.finish()
    }
}

impl std::fmt::Debug for GgufLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("GgufLayer");
        debug_struct.field("name", &self.name);
        debug_struct.finish()
    }
}
