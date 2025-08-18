use axum::Router;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	let app = realtime::router();

	let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8080));
	tracing::info!(%addr, "starting realtime server");
	axum::serve(
		tokio::net::TcpListener::bind(addr).await.unwrap(), 
		app.into_make_service_with_connect_info::<std::net::SocketAddr>()
	)
		.await
		.unwrap();
} 