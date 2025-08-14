use axum::Router;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	let api = realtime::router();

	let static_service = tower_http::services::ServeDir::new("realtime/web-ui/dist");
	let app = Router::new()
		.nest("/", api)
		.nest_service("/ui", static_service);

	let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 8080));
	tracing::info!(%addr, "starting realtime server");
	axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app)
		.await
		.unwrap();
} 