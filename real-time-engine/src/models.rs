use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    pub hf_repo: String,
    pub context_len: u32,
    pub quant: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub default_model_id: String,
    pub models: Vec<ModelSpec>,
}

#[derive(Clone)]
pub struct ModelsRegistry {
    pub default_id: String,
    pub by_id: HashMap<String, ModelSpec>,
}

impl ModelsRegistry {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let data = fs::read_to_string(path)?;
        let cfg: ModelsConfig = serde_json::from_str(&data)?;
        let mut by = HashMap::new();
        for m in cfg.models.iter() {
            by.insert(m.id.clone(), m.clone());
        }
        Ok(Self { default_id: cfg.default_model_id, by_id: by })
    }

    pub fn get(&self, id: &str) -> Option<&ModelSpec> {
        self.by_id.get(id)
    }

    pub fn list(&self) -> Vec<&ModelSpec> {
        self.by_id.values().collect()
    }
} 