use std::collections::HashMap;

use super::gguf_tensors::TensorInfo;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GgufLayer {
    pub name: String,
    pub tensors: HashMap<String, TensorInfo>,
}

impl GgufLayer {
    pub fn get(&self, key: &str) -> Option<&TensorInfo> {
        self.tensors.get(key)
    }
    pub fn size(&self) -> u64 {
        self.tensors.iter().map(|(_, t)| t.size() as u64).sum()
    }
}
