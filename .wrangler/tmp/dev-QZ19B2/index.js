var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// .wrangler/tmp/bundle-3MAYc8/checked-fetch.js
var urls = /* @__PURE__ */ new Set();
function checkURL(request, init) {
  const url = request instanceof URL ? request : new URL(
    (typeof request === "string" ? new Request(request, init) : request).url
  );
  if (url.port && url.port !== "443" && url.protocol === "https:") {
    if (!urls.has(url.toString())) {
      urls.add(url.toString());
      console.warn(
        `WARNING: known issue with \`fetch()\` requests to custom HTTPS ports in published Workers:
 - ${url.toString()} - the custom port will be ignored when the Worker is published using the \`wrangler deploy\` command.
`
      );
    }
  }
}
__name(checkURL, "checkURL");
globalThis.fetch = new Proxy(globalThis.fetch, {
  apply(target, thisArg, argArray) {
    const [request, init] = argArray;
    checkURL(request, init);
    return Reflect.apply(target, thisArg, argArray);
  }
});

// .wrangler/tmp/bundle-3MAYc8/strip-cf-connecting-ip-header.js
function stripCfConnectingIPHeader(input, init) {
  const request = new Request(input, init);
  request.headers.delete("CF-Connecting-IP");
  return request;
}
__name(stripCfConnectingIPHeader, "stripCfConnectingIPHeader");
globalThis.fetch = new Proxy(globalThis.fetch, {
  apply(target, thisArg, argArray) {
    return Reflect.apply(target, thisArg, [
      stripCfConnectingIPHeader.apply(null, argArray)
    ]);
  }
});

// src/semantic_engine.js
var SemanticEngine = class {
  constructor(wasmModule) {
    this.wasm = wasmModule;
  }
  async analyze(prompt, model = "gpt4") {
    const startTime = Date.now();
    try {
      const result = this.generateSemanticAnalysis(prompt, model);
      const processingTime = Date.now() - startTime;
      return {
        h_bar: result.h_bar,
        delta_mu: result.delta_mu,
        delta_sigma: result.delta_sigma,
        risk_level: result.risk_level,
        processing_time: processingTime,
        edge_location: this.getEdgeLocation()
      };
    } catch (error) {
      console.error("Semantic analysis error:", error);
      throw new Error("Failed to analyze semantic uncertainty");
    }
  }
  async batchAnalyze(prompts, model = "gpt4") {
    const startTime = Date.now();
    const results = [];
    for (const prompt of prompts) {
      const analysis = await this.analyze(prompt, model);
      results.push({
        prompt,
        ...analysis
      });
    }
    const totalTime = Date.now() - startTime;
    return {
      results,
      total_prompts: prompts.length,
      total_time: totalTime,
      average_h_bar: results.reduce((sum, r) => sum + r.h_bar, 0) / results.length,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    };
  }
  generateSemanticAnalysis(prompt, model) {
    const promptLength = prompt.length;
    const complexity = this.calculateComplexity(prompt);
    const modelFactor = this.getModelFactor(model);
    const delta_mu = this.calculatePrecision(prompt, complexity, modelFactor);
    const delta_sigma = this.calculateFlexibility(prompt, complexity, modelFactor);
    const h_bar = Math.sqrt(delta_mu * delta_sigma);
    let risk_level;
    if (h_bar < 1) {
      risk_level = "high_collapse_risk";
    } else if (h_bar < 1.2) {
      risk_level = "moderate_instability";
    } else {
      risk_level = "stable";
    }
    return {
      h_bar: parseFloat(h_bar.toFixed(4)),
      delta_mu: parseFloat(delta_mu.toFixed(4)),
      delta_sigma: parseFloat(delta_sigma.toFixed(4)),
      risk_level
    };
  }
  calculateComplexity(prompt) {
    const words = prompt.split(/\s+/).length;
    const sentences = prompt.split(/[.!?]+/).length;
    const questions = (prompt.match(/\?/g) || []).length;
    const imperatives = (prompt.match(/\b(write|create|generate|explain|describe|analyze)\b/gi) || []).length;
    return {
      word_count: words,
      sentence_count: sentences,
      question_density: questions / sentences,
      imperative_density: imperatives / sentences,
      overall_complexity: words * 0.1 + sentences * 0.2 + questions * 0.3 + imperatives * 0.4
    };
  }
  getModelFactor(model) {
    const factors = {
      "gpt4": 0.85,
      "claude3": 0.82,
      "gemini": 0.78,
      "mistral": 0.75,
      "grok3": 0.8,
      "openai_o3": 0.9
    };
    return factors[model] || 0.75;
  }
  calculatePrecision(prompt, complexity, modelFactor) {
    const baseVariation = 0.1 + complexity.overall_complexity * 0.02;
    const modelAdjustment = 1 - modelFactor;
    const lengthFactor = Math.min(prompt.length / 1e3, 1);
    return baseVariation + modelAdjustment + lengthFactor * 0.1;
  }
  calculateFlexibility(prompt, complexity, modelFactor) {
    const baseFlexibility = 0.8 + complexity.question_density * 0.3;
    const creativityBoost = complexity.imperative_density * 0.2;
    const modelStability = modelFactor;
    return baseFlexibility + creativityBoost + (1 - modelStability);
  }
  getEdgeLocation() {
    return "global-edge";
  }
};
__name(SemanticEngine, "SemanticEngine");

// src/index.js
var RateLimiter = class {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }
  async fetch(request) {
    const ip = request.headers.get("CF-Connecting-IP");
    const key = `rate_limit:${ip}`;
    const current = await this.state.storage.get(key) || 0;
    const limit = parseInt(this.env.RATE_LIMIT_PER_MINUTE) || 100;
    if (current >= limit) {
      return new Response(JSON.stringify({
        error: "Rate limit exceeded",
        limit,
        reset_in: 60
      }), {
        status: 429,
        headers: { "Content-Type": "application/json" }
      });
    }
    await this.state.storage.put(key, current + 1, { expirationTtl: 60 });
    return new Response(JSON.stringify({ allowed: true }), {
      headers: { "Content-Type": "application/json" }
    });
  }
};
__name(RateLimiter, "RateLimiter");
var src_default = {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": env.ALLOWED_ORIGINS || "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
      "Access-Control-Max-Age": "86400"
    };
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    try {
      const url = new URL(request.url);
      const path = url.pathname;
      if (path === "/health") {
        return new Response(JSON.stringify({
          status: "healthy",
          timestamp: (/* @__PURE__ */ new Date()).toISOString(),
          version: "1.0.0",
          engine: "semantic-uncertainty-cloudflare"
        }), {
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        });
      }
      const apiKey = request.headers.get("X-API-Key") || request.headers.get("Authorization")?.replace("Bearer ", "");
      if (!apiKey || apiKey !== env.API_KEY_SECRET) {
        return new Response(JSON.stringify({
          error: "Unauthorized",
          message: "Valid API key required"
        }), {
          status: 401,
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        });
      }
      if (path === "/api/v1/analyze" && request.method === "POST") {
        const body = await request.json();
        const { prompt, model = "gpt4" } = body;
        if (!prompt) {
          return new Response(JSON.stringify({
            error: "Bad Request",
            message: "Prompt is required"
          }), {
            status: 400,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders
            }
          });
        }
        const engine = new SemanticEngine(null);
        const result = await engine.analyze(prompt, model);
        return new Response(JSON.stringify({
          success: true,
          data: {
            prompt,
            model,
            semantic_uncertainty: result.h_bar,
            precision: result.delta_mu,
            flexibility: result.delta_sigma,
            risk_level: result.risk_level,
            processing_time: result.processing_time,
            timestamp: (/* @__PURE__ */ new Date()).toISOString()
          }
        }), {
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        });
      }
      if (path === "/api/v1/batch" && request.method === "POST") {
        const body = await request.json();
        const { prompts, model = "gpt4" } = body;
        if (!prompts || !Array.isArray(prompts)) {
          return new Response(JSON.stringify({
            error: "Bad Request",
            message: "Prompts array is required"
          }), {
            status: 400,
            headers: {
              "Content-Type": "application/json",
              ...corsHeaders
            }
          });
        }
        const engine = new SemanticEngine(null);
        const results = await engine.batchAnalyze(prompts, model);
        return new Response(JSON.stringify({
          success: true,
          data: results
        }), {
          headers: {
            "Content-Type": "application/json",
            ...corsHeaders
          }
        });
      }
      return new Response(JSON.stringify({
        error: "Not Found",
        message: "Endpoint not found"
      }), {
        status: 404,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    } catch (error) {
      console.error("Worker error:", error);
      return new Response(JSON.stringify({
        error: "Internal Server Error",
        message: "Something went wrong"
      }), {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          ...corsHeaders
        }
      });
    }
  }
};

// node_modules/wrangler/templates/middleware/middleware-ensure-req-body-drained.ts
var drainBody = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } finally {
    try {
      if (request.body !== null && !request.bodyUsed) {
        const reader = request.body.getReader();
        while (!(await reader.read()).done) {
        }
      }
    } catch (e) {
      console.error("Failed to drain the unused request body.", e);
    }
  }
}, "drainBody");
var middleware_ensure_req_body_drained_default = drainBody;

// node_modules/wrangler/templates/middleware/middleware-miniflare3-json-error.ts
function reduceError(e) {
  return {
    name: e?.name,
    message: e?.message ?? String(e),
    stack: e?.stack,
    cause: e?.cause === void 0 ? void 0 : reduceError(e.cause)
  };
}
__name(reduceError, "reduceError");
var jsonError = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } catch (e) {
    const error = reduceError(e);
    return Response.json(error, {
      status: 500,
      headers: { "MF-Experimental-Error-Stack": "true" }
    });
  }
}, "jsonError");
var middleware_miniflare3_json_error_default = jsonError;

// .wrangler/tmp/bundle-3MAYc8/middleware-insertion-facade.js
var __INTERNAL_WRANGLER_MIDDLEWARE__ = [
  middleware_ensure_req_body_drained_default,
  middleware_miniflare3_json_error_default
];
var middleware_insertion_facade_default = src_default;

// node_modules/wrangler/templates/middleware/common.ts
var __facade_middleware__ = [];
function __facade_register__(...args) {
  __facade_middleware__.push(...args.flat());
}
__name(__facade_register__, "__facade_register__");
function __facade_invokeChain__(request, env, ctx, dispatch, middlewareChain) {
  const [head, ...tail] = middlewareChain;
  const middlewareCtx = {
    dispatch,
    next(newRequest, newEnv) {
      return __facade_invokeChain__(newRequest, newEnv, ctx, dispatch, tail);
    }
  };
  return head(request, env, ctx, middlewareCtx);
}
__name(__facade_invokeChain__, "__facade_invokeChain__");
function __facade_invoke__(request, env, ctx, dispatch, finalMiddleware) {
  return __facade_invokeChain__(request, env, ctx, dispatch, [
    ...__facade_middleware__,
    finalMiddleware
  ]);
}
__name(__facade_invoke__, "__facade_invoke__");

// .wrangler/tmp/bundle-3MAYc8/middleware-loader.entry.ts
var __Facade_ScheduledController__ = class {
  constructor(scheduledTime, cron, noRetry) {
    this.scheduledTime = scheduledTime;
    this.cron = cron;
    this.#noRetry = noRetry;
  }
  #noRetry;
  noRetry() {
    if (!(this instanceof __Facade_ScheduledController__)) {
      throw new TypeError("Illegal invocation");
    }
    this.#noRetry();
  }
};
__name(__Facade_ScheduledController__, "__Facade_ScheduledController__");
function wrapExportedHandler(worker) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return worker;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  const fetchDispatcher = /* @__PURE__ */ __name(function(request, env, ctx) {
    if (worker.fetch === void 0) {
      throw new Error("Handler does not export a fetch() function.");
    }
    return worker.fetch(request, env, ctx);
  }, "fetchDispatcher");
  return {
    ...worker,
    fetch(request, env, ctx) {
      const dispatcher = /* @__PURE__ */ __name(function(type, init) {
        if (type === "scheduled" && worker.scheduled !== void 0) {
          const controller = new __Facade_ScheduledController__(
            Date.now(),
            init.cron ?? "",
            () => {
            }
          );
          return worker.scheduled(controller, env, ctx);
        }
      }, "dispatcher");
      return __facade_invoke__(request, env, ctx, dispatcher, fetchDispatcher);
    }
  };
}
__name(wrapExportedHandler, "wrapExportedHandler");
function wrapWorkerEntrypoint(klass) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return klass;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  return class extends klass {
    #fetchDispatcher = (request, env, ctx) => {
      this.env = env;
      this.ctx = ctx;
      if (super.fetch === void 0) {
        throw new Error("Entrypoint class does not define a fetch() function.");
      }
      return super.fetch(request);
    };
    #dispatcher = (type, init) => {
      if (type === "scheduled" && super.scheduled !== void 0) {
        const controller = new __Facade_ScheduledController__(
          Date.now(),
          init.cron ?? "",
          () => {
          }
        );
        return super.scheduled(controller);
      }
    };
    fetch(request) {
      return __facade_invoke__(
        request,
        this.env,
        this.ctx,
        this.#dispatcher,
        this.#fetchDispatcher
      );
    }
  };
}
__name(wrapWorkerEntrypoint, "wrapWorkerEntrypoint");
var WRAPPED_ENTRY;
if (typeof middleware_insertion_facade_default === "object") {
  WRAPPED_ENTRY = wrapExportedHandler(middleware_insertion_facade_default);
} else if (typeof middleware_insertion_facade_default === "function") {
  WRAPPED_ENTRY = wrapWorkerEntrypoint(middleware_insertion_facade_default);
}
var middleware_loader_entry_default = WRAPPED_ENTRY;
export {
  RateLimiter,
  __INTERNAL_WRANGLER_MIDDLEWARE__,
  middleware_loader_entry_default as default
};
//# sourceMappingURL=index.js.map
