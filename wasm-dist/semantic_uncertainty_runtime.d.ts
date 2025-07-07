/* tslint:disable */
/* eslint-disable */
/**
* Semantic Uncertainty Runtime WASM Module
* Ultra-fast edge computing with ℏₛ = √(Δμ × Δσ) guided security
*/

export interface HbarResponse {
  request_id: string;
  hbar_s: number;
  delta_mu: number;
  delta_sigma: number;
  collapse_risk: boolean;
  processing_time_ms: number;
  embedding_dims: number;
  security_assessment?: SecurityAssessment;
  timestamp: string;
}

export interface SecurityAssessment {
  overall_security_score: number;
  security_emoji: string;
  security_phrase: string;
  action: string;
  threat_indicators: any[];
}

/**
* WASM Semantic Analyzer
*/
export class WasmSemanticAnalyzer {
  free(): void;
  /**
  * @returns {WasmSemanticAnalyzer}
  */
  constructor();
  /**
  * @param {string} prompt
  * @param {string} output
  * @returns {Promise<any>}
  */
  analyze(prompt: string, output: string): Promise<any>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmsemanticanalyzer_free: (a: number) => void;
  readonly wasmsemanticanalyzer_new: () => number;
  readonly wasmsemanticanalyzer_analyze: (a: number, b: number, c: number, d: number, e: number) => number;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
