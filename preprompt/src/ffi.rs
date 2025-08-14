// Production-ready Python bindings for Semantic Uncertainty Runtime
// Optimized for Jupyter notebooks and Python ML pipelines

use crate::{SemanticAnalyzer, SemanticConfig, HbarResponse, SemanticError, RequestId};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyTimeoutError};
use std::collections::HashMap;
use tokio::runtime::Runtime;
use tracing::{info, error, debug, instrument};

/// Python-compatible semantic uncertainty analyzer
#[pyclass]
pub struct PySemanticAnalyzer {
    analyzer: SemanticAnalyzer,
    runtime: Runtime,
}

/// Python-compatible analysis response
#[pyclass]
#[derive(Clone)]
pub struct PyHbarResponse {
    /// Request ID for tracing
    #[pyo3(get)]
    pub request_id: String,
    
    /// Quantum-inspired semantic uncertainty metric
    #[pyo3(get)]
    pub hbar_s: f64,
    
    /// Semantic precision metric
    #[pyo3(get)]
    pub delta_mu: f64,
    
    /// Semantic flexibility metric
    #[pyo3(get)]
    pub delta_sigma: f64,
    
    /// Optional failure probability
    #[pyo3(get)]
    pub p_fail: Option<f64>,
    
    /// Collapse risk indicator
    #[pyo3(get)]
    pub collapse_risk: bool,
    
    /// Processing time in milliseconds
    #[pyo3(get)]
    pub processing_time_ms: f64,
    
    /// Embedding dimensions used
    #[pyo3(get)]
    pub embedding_dims: usize,
    
    /// Analysis timestamp (ISO format)
    #[pyo3(get)]
    pub timestamp: String,
}

/// Python-compatible configuration
#[pyclass]
#[derive(Clone)]
pub struct PySemanticConfig {
    /// Collapse detection threshold
    #[pyo3(get, set)]
    pub collapse_threshold: f64,
    
    /// Maximum sequence length for processing
    #[pyo3(get, set)]
    pub max_sequence_length: usize,
    
    /// Enable fast approximations
    #[pyo3(get, set)]
    pub fast_mode: bool,
    
    /// Embedding dimensions
    #[pyo3(get, set)]
    pub embedding_dims: usize,
    
    /// Request timeout in milliseconds
    #[pyo3(get, set)]
    pub timeout_ms: u64,
    
    /// Enable SIMD optimizations
    #[pyo3(get, set)]
    pub use_simd: bool,
}

/// Batch analysis results for efficient processing
#[pyclass]
pub struct PyBatchResults {
    /// List of analysis responses
    #[pyo3(get)]
    pub results: Vec<PyHbarResponse>,
    
    /// Overall processing time
    #[pyo3(get)]
    pub total_time_ms: f64,
    
    /// Success rate (0.0 to 1.0)
    #[pyo3(get)]
    pub success_rate: f64,
    
    /// Number of collapse detections
    #[pyo3(get)]
    pub collapse_count: usize,
}

#[pymethods]
impl PySemanticAnalyzer {
    /// Create a new semantic analyzer with configuration
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<PySemanticConfig>) -> PyResult<Self> {
        let rust_config = match config {
            Some(py_config) => py_config.to_rust_config(),
            None => SemanticConfig::performance(),
        };
        
        let analyzer = SemanticAnalyzer::new(rust_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e)))?;
        
        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create async runtime: {}", e)))?;
        
        info!("🧠 Python semantic analyzer initialized");
        
        Ok(Self { analyzer, runtime })
    }
    
    /// Analyze a single prompt-output pair
    #[pyo3(signature = (prompt, output, request_id = None))]
    pub fn analyze(
        &self,
        prompt: &str,
        output: &str,
        request_id: Option<&str>,
    ) -> PyResult<PyHbarResponse> {
        let req_id = match request_id {
            Some(id) => RequestId::new(), // Could parse from string if needed
            None => RequestId::new(),
        };
        
        let result = self.runtime.block_on(
            self.analyzer.analyze(prompt, output, req_id)
        );
        
        match result {
            Ok(response) => Ok(PyHbarResponse::from_rust(response)),
            Err(e) => match e {
                SemanticError::InvalidInput { message } => {
                    Err(PyValueError::new_err(format!("Invalid input: {}", message)))
                }
                SemanticError::Timeout { timeout_ms } => {
                    Err(PyTimeoutError::new_err(format!("Analysis timed out after {}ms", timeout_ms)))
                }
                _ => Err(PyRuntimeError::new_err(format!("Analysis failed: {}", e))),
            }
        }
    }
    
    /// Analyze multiple prompt-output pairs efficiently
    #[pyo3(signature = (pairs, max_workers = None))]
    pub fn analyze_batch(
        &self,
        pairs: Vec<(String, String)>,
        max_workers: Option<usize>,
    ) -> PyResult<PyBatchResults> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        let mut success_count = 0;
        let mut collapse_count = 0;
        
        debug!("🔄 Starting batch analysis of {} pairs", pairs.len());
        
        // Process pairs with concurrency control
        let workers = max_workers.unwrap_or(num_cpus::get().min(pairs.len()));
        let chunk_size = (pairs.len() + workers - 1) / workers;
        
        for chunk in pairs.chunks(chunk_size) {
            let chunk_results: Vec<_> = chunk.iter().map(|(prompt, output)| {
                let req_id = RequestId::new();
                let result = self.runtime.block_on(
                    self.analyzer.analyze(prompt, output, req_id)
                );
                
                match result {
                    Ok(response) => {
                        if response.collapse_risk {
                            collapse_count += 1;
                        }
                        success_count += 1;
                        Ok(PyHbarResponse::from_rust(response))
                    }
                    Err(e) => Err(e),
                }
            }).collect();
            
            for result in chunk_results {
                match result {
                    Ok(response) => results.push(response),
                    Err(e) => {
                        error!("Batch analysis error: {}", e);
                        // Continue processing other pairs
                    }
                }
            }
        }
        
        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let success_rate = success_count as f64 / pairs.len() as f64;
        
        info!("✅ Batch analysis completed: {}/{} successful in {:.2}ms", 
              success_count, pairs.len(), total_time_ms);
        
        Ok(PyBatchResults {
            results,
            total_time_ms,
            success_rate,
            collapse_count,
        })
    }
    
    /// Get analyzer statistics and performance metrics
    pub fn get_stats(&self) -> PyResult<PyDict> {
        let py = Python::acquire_gil();
        let dict = PyDict::new(py.python());
        
        dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
        dict.set_item("fast_mode", true)?;
        dict.set_item("embedding_dims", 128)?;
        dict.set_item("simd_enabled", cfg!(feature = "fast-math"))?;
        dict.set_item("wasm_enabled", cfg!(feature = "wasm"))?;
        
        Ok(dict.to_object(py.python()).downcast::<PyDict>()?.clone())
    }
    
    /// Create a performance-optimized configuration
    #[staticmethod]
    pub fn performance_config() -> PySemanticConfig {
        PySemanticConfig::from_rust(SemanticConfig::performance())
    }
    
    /// Create an ultra-fast configuration for sub-10ms analysis
    #[staticmethod]
    pub fn ultra_fast_config() -> PySemanticConfig {
        PySemanticConfig::from_rust(SemanticConfig::ultra_fast())
    }
    
    /// Create a high-accuracy configuration
    #[staticmethod]
    pub fn accuracy_config() -> PySemanticConfig {
        PySemanticConfig::from_rust(SemanticConfig::accuracy())
    }
    
    /// Benchmark the analyzer performance
    #[pyo3(signature = (iterations = 10))]
    pub fn benchmark(&self, iterations: usize) -> PyResult<PyDict> {
        let py = Python::acquire_gil();
        let dict = PyDict::new(py.python());
        
        let test_cases = vec![
            ("What is AI?", "AI is artificial intelligence."),
            ("Explain quantum physics", "Quantum physics studies subatomic particles."),
            ("Tell me about machine learning", "Machine learning uses algorithms to find patterns in data."),
        ];
        
        let mut total_time = 0.0;
        let mut successful_runs = 0;
        
        for _ in 0..iterations {
            for (prompt, output) in &test_cases {
                let start = std::time::Instant::now();
                let req_id = RequestId::new();
                
                match self.runtime.block_on(self.analyzer.analyze(prompt, output, req_id)) {
                    Ok(_) => {
                        total_time += start.elapsed().as_secs_f64() * 1000.0;
                        successful_runs += 1;
                    }
                    Err(_) => {} // Skip failed runs
                }
            }
        }
        
        let avg_time = total_time / successful_runs as f64;
        let total_analyses = iterations * test_cases.len();
        let success_rate = successful_runs as f64 / total_analyses as f64;
        
        dict.set_item("average_latency_ms", avg_time)?;
        dict.set_item("total_time_ms", total_time)?;
        dict.set_item("success_rate", success_rate)?;
        dict.set_item("successful_runs", successful_runs)?;
        dict.set_item("total_analyses", total_analyses)?;
        
        // Performance assessment
        let performance_grade = if avg_time <= 10.0 {
            "EXCELLENT (Sub-10ms)"
        } else if avg_time <= 100.0 {
            "GOOD (Sub-100ms)"
        } else {
            "NEEDS_OPTIMIZATION"
        };
        
        dict.set_item("performance_grade", performance_grade)?;
        
        Ok(dict.to_object(py.python()).downcast::<PyDict>()?.clone())
    }
}

#[pymethods]
impl PyHbarResponse {
    /// Convert to Python dictionary
    pub fn to_dict(&self) -> PyResult<PyDict> {
        let py = Python::acquire_gil();
        let dict = PyDict::new(py.python());
        
        dict.set_item("request_id", &self.request_id)?;
        dict.set_item("hbar_s", self.hbar_s)?;
        dict.set_item("delta_mu", self.delta_mu)?;
        dict.set_item("delta_sigma", self.delta_sigma)?;
        if let Some(pf) = self.p_fail { dict.set_item("p_fail", pf)?; }
        dict.set_item("collapse_risk", self.collapse_risk)?;
        dict.set_item("processing_time_ms", self.processing_time_ms)?;
        dict.set_item("embedding_dims", self.embedding_dims)?;
        dict.set_item("timestamp", &self.timestamp)?;
        
        Ok(dict.to_object(py.python()).downcast::<PyDict>()?.clone())
    }
    
    /// String representation for debugging
    pub fn __repr__(&self) -> String {
        format!(
            "PyHbarResponse(hbar_s={:.4}, collapse_risk={}, time={:.2}ms)",
            self.hbar_s, self.collapse_risk, self.processing_time_ms
        )
    }
    
    /// Check if semantic collapse was detected
    pub fn is_collapsed(&self) -> bool {
        self.collapse_risk
    }
    
    /// Get the uncertainty level as a human-readable string
    pub fn uncertainty_level(&self) -> &str {
        if self.hbar_s < 0.5 {
            "VERY_LOW"
        } else if self.hbar_s < 1.0 {
            "LOW"
        } else if self.hbar_s < 2.0 {
            "MODERATE"
        } else if self.hbar_s < 3.0 {
            "HIGH"
        } else {
            "VERY_HIGH"
        }
    }
}

#[pymethods]
impl PySemanticConfig {
    /// Create a new configuration
    #[new]
    #[pyo3(signature = (
        collapse_threshold = 1.0,
        max_sequence_length = 256,
        fast_mode = true,
        embedding_dims = 128,
        timeout_ms = 5000,
        use_simd = true
    ))]
    pub fn new(
        collapse_threshold: f64,
        max_sequence_length: usize,
        fast_mode: bool,
        embedding_dims: usize,
        timeout_ms: u64,
        use_simd: bool,
    ) -> Self {
        Self {
            collapse_threshold,
            max_sequence_length,
            fast_mode,
            embedding_dims,
            timeout_ms,
            use_simd,
        }
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "PySemanticConfig(threshold={}, dims={}, fast={}, simd={})",
            self.collapse_threshold, self.embedding_dims, self.fast_mode, self.use_simd
        )
    }
}

#[pymethods]
impl PyBatchResults {
    /// Get summary statistics
    pub fn summary(&self) -> PyResult<PyDict> {
        let py = Python::acquire_gil();
        let dict = PyDict::new(py.python());
        
        dict.set_item("total_analyses", self.results.len())?;
        dict.set_item("total_time_ms", self.total_time_ms)?;
        dict.set_item("success_rate", self.success_rate)?;
        dict.set_item("collapse_count", self.collapse_count)?;
        dict.set_item("collapse_rate", self.collapse_count as f64 / self.results.len() as f64)?;
        
        if !self.results.is_empty() {
            let avg_hbar_s: f64 = self.results.iter().map(|r| r.hbar_s).sum::<f64>() / self.results.len() as f64;
            let avg_time: f64 = self.results.iter().map(|r| r.processing_time_ms).sum::<f64>() / self.results.len() as f64;
            
            dict.set_item("average_hbar_s", avg_hbar_s)?;
            dict.set_item("average_time_ms", avg_time)?;
        }
        
        Ok(dict.to_object(py.python()).downcast::<PyDict>()?.clone())
    }
    
    /// Filter results by collapse risk
    pub fn filter_collapsed(&self) -> Vec<PyHbarResponse> {
        self.results.iter().filter(|r| r.collapse_risk).cloned().collect()
    }
    
    /// Filter results by uncertainty threshold
    pub fn filter_by_uncertainty(&self, min_hbar_s: f64) -> Vec<PyHbarResponse> {
        self.results.iter().filter(|r| r.hbar_s >= min_hbar_s).cloned().collect()
    }
}

// Implementation helpers

impl PyHbarResponse {
    fn from_rust(response: HbarResponse) -> Self {
        Self {
            request_id: response.request_id.to_string(),
            hbar_s: response.hbar_s as f64,
            delta_mu: response.delta_mu as f64,
            delta_sigma: response.delta_sigma as f64,
            p_fail: response.p_fail,
            collapse_risk: response.collapse_risk,
            processing_time_ms: response.processing_time_ms,
            embedding_dims: response.embedding_dims,
            timestamp: response.timestamp.to_rfc3339(),
        }
    }
}

impl PySemanticConfig {
    fn from_rust(config: SemanticConfig) -> Self {
        Self {
            collapse_threshold: config.collapse_threshold as f64,
            max_sequence_length: config.max_sequence_length,
            fast_mode: config.fast_mode,
            embedding_dims: config.embedding_dims,
            timeout_ms: config.timeout_ms,
            use_simd: config.use_simd,
        }
    }
    
    fn to_rust_config(&self) -> SemanticConfig {
        SemanticConfig {
            model_path: "models/sentence_transformer.onnx".to_string(),
            collapse_threshold: self.collapse_threshold as f32,
            perturbation_count: if self.fast_mode { 2 } else { 10 },
            max_sequence_length: self.max_sequence_length,
            fast_mode: self.fast_mode,
            entropy_min_threshold: if self.fast_mode { 0.05 } else { 0.001 },
            js_min_threshold: if self.fast_mode { 0.005 } else { 0.001 },
            embedding_dims: self.embedding_dims,
            timeout_ms: self.timeout_ms,
            use_simd: self.use_simd,
        }
    }
}

/// Convenience function for quick analysis
#[pyfunction]
pub fn quick_analyze(prompt: &str, output: &str) -> PyResult<PyHbarResponse> {
    let config = SemanticConfig::ultra_fast();
    let analyzer = SemanticAnalyzer::new(config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create analyzer: {}", e)))?;
    
    let runtime = Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
    
    let req_id = RequestId::new();
    let result = runtime.block_on(analyzer.analyze(prompt, output, req_id));
    
    match result {
        Ok(response) => Ok(PyHbarResponse::from_rust(response)),
        Err(e) => Err(PyRuntimeError::new_err(format!("Analysis failed: {}", e))),
    }
}

/// Get version information
#[pyfunction]
pub fn version_info() -> PyResult<PyDict> {
    let py = Python::acquire_gil();
    let dict = PyDict::new(py.python());
    
    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item("description", env!("CARGO_PKG_DESCRIPTION"))?;
    dict.set_item("rust_version", env!("RUST_VERSION").unwrap_or("unknown"))?;
    dict.set_item("features", vec![
        #[cfg(feature = "fast-math")] "fast-math",
        #[cfg(feature = "wasm")] "wasm",
        #[cfg(feature = "onnx")] "onnx",
        "python",
    ])?;
    
    Ok(dict.to_object(py.python()).downcast::<PyDict>()?.clone())
}

/// Python module definition
#[pymodule]
fn semantic_uncertainty_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add classes
    m.add_class::<PySemanticAnalyzer>()?;
    m.add_class::<PyHbarResponse>()?;
    m.add_class::<PySemanticConfig>()?;
    m.add_class::<PyBatchResults>()?;
    
    // Add convenience functions
    m.add_function(wrap_pyfunction!(quick_analyze, m)?)?;
    m.add_function(wrap_pyfunction!(version_info, m)?)?;
    
    // Add module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Semantic Uncertainty Team")?;
    m.add("__description__", "High-performance semantic uncertainty analysis for AI collapse prevention")?;
    
    info!("🐍 Python module initialized successfully");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_config_creation() {
        let config = PySemanticConfig::new(
            Some("test_model.onnx".to_string()),
            Some(0.5),
            Some(5),
            Some(256),
        );

        assert_eq!(config.model_path(), "test_model.onnx");
        assert_eq!(config.collapse_threshold(), 0.5);
        assert_eq!(config.perturbation_count(), 5);
        assert_eq!(config.max_sequence_length(), 256);
    }

    #[test]
    fn test_py_response_conversion() {
        let response = HbarResponse {
            hbar_s: 1.5,
            delta_mu: 0.8,
            delta_sigma: 0.6,
            collapse_risk: false,
        };

        let py_response: PyHbarResponse = response.into();
        assert_eq!(py_response.hbar_s(), 1.5);
        assert_eq!(py_response.delta_mu(), 0.8);
        assert_eq!(py_response.delta_sigma(), 0.6);
        assert!(!py_response.collapse_risk());
    }
} 