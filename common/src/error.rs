#[derive(thiserror::Error, Debug)]
pub enum SemanticError {
	#[error("Input validation failed: {message}")]
	InvalidInput { message: String },
	#[error("Embedding computation failed: {source}")]
	EmbeddingError { source: anyhow::Error },
	#[error("Math computation failed: {operation}")]
	MathError { operation: String },
	#[error("Configuration error: {message}")]
	ConfigError { message: String },
	#[error("Timeout occurred after {timeout_ms}ms")]
	Timeout { timeout_ms: u64 },
	#[error("Internal error: {source}")]
	Internal { source: anyhow::Error },
} 