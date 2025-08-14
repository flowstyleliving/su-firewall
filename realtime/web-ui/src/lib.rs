use futures_util::StreamExt;
use gloo_net::websocket::{futures::WebSocket, Message};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use yew::prelude::*;

#[derive(Deserialize, Clone, Debug, PartialEq)]
struct WsEvent {
	type_field: Option<String>,
	hbar_s: Option<f64>,
}

#[function_component(App)]
pub fn app() -> Html {
	let events = use_state(|| Vec::<f64>::new());
	let status = use_state(|| String::from("connecting"));

	{
		let events = events.clone();
		let status = status.clone();
		use_effect_with((), move |_| {
			wasm_bindgen_futures::spawn_local(async move {
				let loc = web_sys::window().unwrap().location();
				let origin = loc.origin().unwrap_or_else(|_| String::from("http://127.0.0.1:8080"));
				let ws_base = origin.replace("http://", "ws://").replace("https://", "wss://");
				let url = format!("{}/ws", ws_base);
				status.set(format!("ws: {}", url));
				let ws = WebSocket::open(&url).expect("ws open");
				let (_write, mut read) = ws.split();
				while let Some(msg) = read.next().await {
					match msg {
						Ok(Message::Text(txt)) => {
							if let Ok(val) = serde_json::from_str::<serde_json::Value>(&txt) {
								if let Some(h) = val.get("hbar_s").and_then(|v| v.as_f64()) {
									let mut v = (*events).clone();
									v.push(h);
									if v.len() > 200 { v.drain(0..v.len()-200); }
									events.set(v);
								}
							}
						}
						_ => {}
					}
				}
			});
			|| ()
		});
	}

	html! {
		<div>
			<h3>{"Realtime ℏ_s"}</h3>
			<p>{ (*status).clone() }</p>
			<ul>
				{ for (*events).iter().rev().take(20).map(|h| html!{ <li>{ format!("{:.4}", h) }</li> }) }
			</ul>
		</div>
	}
}

#[wasm_bindgen(start)]
pub fn run() {
	yew::Renderer::<App>::new().render();
} 