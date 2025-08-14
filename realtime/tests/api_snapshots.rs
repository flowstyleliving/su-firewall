use axum::{Router, body::Body, http::{Request, StatusCode}};
use realtime::router;
use http_body_util::BodyExt;
use tower::util::ServiceExt; // for .oneshot

fn test_router() -> Router {
	router()
}

#[tokio::test]
async fn snapshot_analyze_topk() {
	let app = test_router();
	let payload = serde_json::json!({
		"model_id": "test-model",
		"topk_indices": [[0,1,2,3,4]],
		"topk_probs": [[0.4, 0.2, 0.15, 0.15, 0.1]],
		"rest_mass": [0.0],
		"prompt_next_topk_indices": [0,1,2,3,4],
		"prompt_next_topk_probs": [0.5, 0.2, 0.1, 0.1, 0.1],
		"prompt_next_rest_mass": 0.0,
		"vocab_size": 10,
		"method": "full_fim_dir"
	});
	let req = Request::builder()
		.method("POST")
		.uri("/api/v1/analyze_topk")
		.header("content-type", "application/json")
		.body(Body::from(serde_json::to_vec(&payload).unwrap()))
		.unwrap();

	let res = app.clone().oneshot(req).await.unwrap();
	assert_eq!(res.status(), StatusCode::OK);
	let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
	let mut json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
	// Redact non-deterministic fields
	if let Some(obj) = json.as_object_mut() {
		obj.insert("request_id".to_string(), serde_json::json!("<redacted>"));
		obj.insert("timestamp".to_string(), serde_json::json!("<redacted>"));
		obj.insert("processing_time_ms".to_string(), serde_json::json!(0.0));
	}
	insta::assert_json_snapshot!("analyze_topk_ok", json);
}

#[tokio::test]
async fn snapshot_analyze_topk_compact() {
	let app = test_router();
	let payload = serde_json::json!({
		"model_id": "test-model",
		"prompt_next_topk_indices": [0,1,2,3,4],
		"prompt_next_topk_probs": [0.5, 0.2, 0.1, 0.1, 0.1],
		"prompt_next_rest_mass": 0.0,
		"topk_indices": [0,1,2,3,4],
		"topk_probs": [0.4, 0.2, 0.15, 0.15, 0.1],
		"rest_mass": 0.0,
		"method": "full_fim_dir"
	});
	let req = Request::builder()
		.method("POST")
		.uri("/api/v1/analyze_topk_compact")
		.header("content-type", "application/json")
		.body(Body::from(serde_json::to_vec(&payload).unwrap()))
		.unwrap();

	let res = app.clone().oneshot(req).await.unwrap();
	assert_eq!(res.status(), StatusCode::OK);
	let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
	let mut json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
	// Redact non-deterministic fields
	if let Some(obj) = json.as_object_mut() {
		obj.insert("request_id".to_string(), serde_json::json!("<redacted>"));
		obj.insert("timestamp".to_string(), serde_json::json!("<redacted>"));
		obj.insert("processing_time_ms".to_string(), serde_json::json!(0.0));
	}
	insta::assert_json_snapshot!("analyze_topk_compact_ok", json);
} 