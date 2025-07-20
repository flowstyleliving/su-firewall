// 🌐 Cloudflare Worker WASM Entry Point
// Ultra-fast edge computing with ℏₛ = √(Δμ × Δσ) guided security

use worker::*;
use semantic_uncertainty_runtime::cloudflare_worker;

// Re-export the main function from cloudflare_worker module  
pub use cloudflare_worker::main;

// WASM-specific initialization
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_main() {
    // Initialize console error panic hook for better debugging
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();

    // Initialize tracing for WASM
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .without_time()
        .init();
}