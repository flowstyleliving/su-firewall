#[cfg(feature = "api")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use semantic_uncertainty_runtime::api::create_app;
    use axum::routing::get;
    use axum::Router;
    use tracing::info;

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let app = create_app().await?;

    let addr = std::env::var("API_BIND").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("📡 semantic-uncertainty-runtime API listening on http://{}", addr);

    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

#[cfg(not(feature = "api"))]
fn main() {
    eprintln!("The 'api' feature is not enabled. Run with: cargo run --features api");
} 