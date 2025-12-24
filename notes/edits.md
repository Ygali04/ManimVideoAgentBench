# edits.md — proposed repo edits (OpenRouter-first + modern model selection)

This document describes **technical edits** to apply to this repo so that:

1) **Generation + evaluation can run primarily through OpenRouter** (single API key, many providers)  
2) The experiment harness can easily target **the latest model families** (OpenAI GPT‑5.x, Gemini 3, Claude 4.5, Kimi K2, DeepSeek R1, plus open-source)

It’s written as a developer implementation checklist; it is **not** the actual code change.

Reference behavior/result baselines are described on the project homepage: [TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/).

---

## 0) High-level design choice

### Goal
Make **all inference calls** (planner, coder, helper, evaluation judges) go through a **single “OpenAI-compatible” API surface**, backed by OpenRouter.

### Key reality check (updated: OpenRouter supports video inputs)
- This repo currently mixes:
  - **LiteLLM** (provider-agnostic) for most text/image calls
  - **Direct Gemini SDK** (`mllm_tools/gemini.py`) for video evaluation (uploads mp4s)
- OpenRouter’s OpenAI-compatible Chat Completions endpoint supports **video inputs** via `video_url` content parts, including **base64 “data:” URLs** for local files ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview)).  
  That means you can keep TEA’s **mp4-chunk → video model** evaluation pattern, but you **cannot keep** Gemini’s `upload_file` workflow.
- OpenRouter does **not** publish one global max video payload/timeout; constraints are provider/model-specific, so you should implement a **frame-based fallback** for robustness ([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview)).

---

## 1) Add OpenRouter env + configuration

### 1.1 Add `.env.template` (currently referenced in README but missing in repo)
Create a repo-root `.env.template` including at minimum:
- `OPENROUTER_API_KEY=""`
- `OPENROUTER_API_BASE="https://openrouter.ai/api/v1"`
- (Optional but recommended by OpenRouter) metadata headers:
  - `OPENROUTER_SITE_URL=""`
  - `OPENROUTER_APP_NAME="TheoremExplainAgent"`

Keep existing keys for optional fallbacks (Langfuse, Kokoro, etc.).

### 1.2 Centralize provider selection in config
Add a config switch, e.g.:
- `TEA_PROVIDER=openrouter` (default), or
- CLI `--provider openrouter`

Use this to:
- choose default model IDs (OpenRouter slugs)
- choose which wrappers to instantiate (LiteLLM-only vs direct SDK fallbacks)

Files likely involved:
- `src/config/config.py` (or a new `src/config/runtime.py`)
- `generate_video.py`
- `evaluate.py`

---

## 2) Make model selection “modern + structured”

### 2.1 Introduce a model registry file
Add a repo file like `src/utils/model_registry.json` (or `models.yaml`) that defines:
- **alias**: stable short name used in scripts (`gpt5_2_thinking`, `claude45_opus`, `gemini3_flash`, etc.)
- **openrouter_id**: actual OpenRouter model slug
- **capabilities**:
  - `text: true/false`
  - `vision: true/false` (image input)
  - `video: true/false` (derive from OpenRouter model metadata; some Gemini models explicitly include `video` in `input_modalities`) ([OpenRouter models API](https://openrouter.ai/api/v1/models), [OpenRouter models guide](https://openrouter.ai/docs/guides/overview/models))
- **recommended params**:
  - `temperature` (or `null`)
  - `max_tokens` / `max_output_tokens`
  - reasoning params (see §3.3)

This avoids hard-coding model lists in `allowed_models.json` and CLI `choices=...`.

### 2.2 Replace the allowlist mechanism
Current behavior:
- `generate_video.py` and `evaluate.py` use `choices=allowed_models` (from `src/utils/allowed_models.json`).

Recommended edits:
- Keep a lightweight safety check but allow rapid model iteration:
  - Option A: accept any model string, warn if not in registry
  - Option B: accept aliases only, resolve via registry

Concretely:
- Update `generate_video.py` args to remove `choices=allowed_models` and instead validate via registry.
- Update `evaluate.py` similarly (especially for `--model_text` and `--model_image`).

---

## 3) Make LiteLLMWrapper truly provider-agnostic (OpenRouter-compatible)

### 3.1 Ensure OpenRouter routing works in LiteLLM
In `mllm_tools/litellm.py` (`LiteLLMWrapper.__call__`), detect OpenRouter models and pass required params:
- `api_base="https://openrouter.ai/api/v1"`
- `api_key=os.getenv("OPENROUTER_API_KEY")`
- `extra_headers`:
  - `HTTP-Referer: $OPENROUTER_SITE_URL` (if set)
  - `X-Title: $OPENROUTER_APP_NAME` (if set)

Do not rely on implicit environment configuration only; make behavior explicit and debuggable.

### 3.2 Fix multimodal formatting logic (critical for non-“gpt” model families)
Current `LiteLLMWrapper` decides message formatting by string matching:
- if `"gemini" in model_name` → Gemini-style
- elif `"gpt" in model_name` → GPT-style
- else → raises error for multimodal

That blocks image-capable models like Claude / Kimi / DeepSeek / open-source vision models.

Recommended edit:
- Switch from substring heuristics to **capability-based** formatting:
  - If model supports vision, send OpenAI-compatible `image_url` parts regardless of vendor name.
  - If model supports video via OpenRouter, support OpenRouter’s `video_url` parts (not Gemini SDK uploads) ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)).
  - Keep a “direct HTTP to OpenRouter” escape hatch for video, because some client libraries may not faithfully pass through OpenRouter’s video part schema end-to-end (OpenRouter documents the schema; client support varies) ([OpenRouter API overview](https://openrouter.ai/docs/api-reference/overview), [OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)).

Implementation approach:
- Introduce `ModelSpec` resolved from the new registry (`vision: true/false`) and route formatting based on that.

### 3.3 Add “reasoning/thinking” parameter support
The current wrapper special-cases OpenAI “o-series” models with `reasoning_effort="medium"`.

Modern model families require a more general abstraction:
- **OpenAI GPT‑5 Thinking** style params (OpenAI reasoning controls)
- **Anthropic Claude “thinking”** style params (budget-based, if exposed through LiteLLM/OpenRouter)
- **DeepSeek / Kimi** reasoning variants (often model-id-based rather than param-based)

Recommended edit:
- Extend registry to include a `reasoning` object per model:
  - `mode: none | openai_reasoning_effort | anthropic_thinking | model_variant_only`
  - `effort: low|medium|high` or `budget_tokens`
- In `LiteLLMWrapper.__call__`, translate that into provider-specific params supported by LiteLLM.

Also recommended:
- Add an optional “dynamic capability check” mode that queries OpenRouter’s models API and inspects `supported_parameters` / `input_modalities` to validate whether `reasoning` / `include_reasoning` (or similar) is supported for a given model ID ([OpenRouter models API](https://openrouter.ai/api/v1/models)).

### 3.4 Normalize “max tokens”/“max output” across providers
Add explicit `max_tokens` (or equivalent) to prevent silent truncation of plans/code, especially for long scenes.

---

## 4) Unify video evaluation to run via OpenRouter (keep mp4-chunk pattern)

### 4.1 Keep “mp4 chunk → video model” evaluation, but replace Gemini `upload_file`
Keep the existing evaluator pattern:
- split full video into chunks
- score each chunk
- aggregate via geometric mean

Change the **transport**:
- Replace `google.generativeai upload_file` with OpenRouter Chat Completions calls that include the chunk as a `video_url` content part.
- Use **base64 “data:” URLs** to send local mp4 chunks without hosting ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)).

Schema pitfall:
- OpenRouter’s documentation/examples use a `video_url` content part but some examples show a camelCase nested field (e.g., `videoUrl`) rather than an OpenAI-style `video_url` object. To avoid client-schema mismatch, implement the video call path as **direct HTTP to OpenRouter** (bypassing strict OpenAI validators), or make your wrapper tolerant to both field spellings ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter API overview](https://openrouter.ai/docs/api-reference/overview)).

Model selection:
- Programmatically filter the OpenRouter models API by `input_modalities` containing `"video"` ([OpenRouter models API](https://openrouter.ai/api/v1/models)).
- Initial candidates from the facts you provided:
  - `google/gemini-3-pro-preview`
  - `google/gemini-3-flash-preview`
  - `google/gemini-2.5-flash`
  - `google/gemini-2.0-flash-lite-001`  
  (All list `video` in `input_modalities` in the models API) ([OpenRouter models API](https://openrouter.ai/api/v1/models)).

Provider routing caveat:
- OpenRouter notes that constraints are provider-specific; for Gemini routed via Google AI Studio, **video URLs may be restricted to YouTube links**, so using base64 data URLs and/or explicitly controlling provider routing is important ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter provider routing guide](https://openrouter.ai/docs/guides/routing/provider-selection)).

Files to edit (expected scope: wrapper-level refactor, not a full redesign):
- `evaluate.py`: allow `--model_video` to be any OpenRouter model ID that supports video (rather than hardcoded Gemini SDK IDs)
- `eval_suite/video_utils.py`: keep chunking logic; swap the *model wrapper* implementation to send `video_url` parts
- `mllm_tools/litellm.py` or a new wrapper (e.g., `mllm_tools/openrouter.py`): add first-class support for `video_url` parts when using OpenRouter

### 4.2 Add a frame-based fallback (robustness across providers/models)
Even if OpenRouter supports video input, it does not publish one global payload/time limit and warns constraints vary per provider/model. You should implement a fallback that:
- samples \(k\) frames per chunk (and keeps frame count within per-request limits)
- sends frames as multiple `image_url` parts to a vision-capable judge
- asks the same “visual consistency” questions framed as cross-frame consistency  
([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview))

### 4.3 Keep a “legacy Gemini SDK upload” mode (optional)
If you want to reproduce the original repo’s evaluation pathway, keep a selectable mode:
- `--video_eval_mode openrouter_video|frames|gemini_sdk_upload`

---

## 5) Update default model set to “modern”

Populate the model registry with (examples; use actual OpenRouter slugs):

### OpenAI
- GPT‑5.2 base + “thinking”
- GPT‑5.1 base + “thinking”
- GPT‑5 base + “thinking”

### Google
- Gemini 3 base / pro / flash

### Anthropic
- Claude 4.5 opus / sonnet / haiku

### China labs
- Kimi K2 base / thinking
- DeepSeek R1 (latest)

### Open source baselines
- Llama / Qwen / etc. (include at least one strong open-source vision-capable model if available)

### 5.1 Concrete OpenRouter IDs + modality notes (from the models API)
Use OpenRouter’s models API as the ground truth for:
- `id` (the exact model string to send)
- `input_modalities` (whether the model supports `video` / `image`)
- `supported_parameters` (whether `reasoning` / `include_reasoning` exists)

([OpenRouter models API](https://openrouter.ai/api/v1/models), [OpenRouter models guide](https://openrouter.ai/docs/guides/overview/models))

Examples from the facts you provided:

| Model (OpenRouter `id`) | Text | Image | Video | Notes |
|---|---:|---:|---:|---|
| `google/gemini-3-pro-preview` | ✅ | ✅ | ✅ | Good candidate for OpenRouter video-eval judge |
| `google/gemini-3-flash-preview` | ✅ | ✅ | ✅ | Faster/cheaper video-eval judge candidate |
| `openai/gpt-5.2-pro` | ✅ | ✅ | ❌ | No `video` modality listed; use for text/image judging |
| `anthropic/claude-opus-4.5` | ✅ | ✅ | ❌ | No `video` modality listed; use for text/image judging |
| `deepseek/deepseek-r1` | ✅ | ❌ | ❌ | Text-only in the models API |
| `moonshotai/kimi-k2` | ✅ | ❌ | ❌ | Text-only in the models API |

For reproducibility, include:
- `released_at` or “pinned version tag” where possible
- A short note if the model is known to be text-only vs vision-capable

---

## 6) RAG embeddings: keep as-is or make OpenRouter-compatible

Current: `src/rag/vector_store.py` uses LiteLLM embeddings with either:
- `azure/text-embedding-3-large`, or
- `vertex_ai/text-embedding-005`

If the goal is “one key only”, OpenRouter supports embeddings via `POST /api/v1/embeddings` ([OpenRouter embeddings: create](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings), [OpenRouter embeddings: list models](https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models)).

Recommended options:
- **OpenRouter embeddings** (single key): route embeddings through OpenRouter
- **Local embeddings** (no network): sentence-transformers for fully offline RAG indexing

Make this a first-class switch, e.g.:
- `--embedding_backend local|openrouter|vertex|azure`

---

## 7) Test plan after edits

Add a minimal smoke regimen:
1) **Plan-only**: `generate_video.py --only_plan` on 2–3 topics
2) **Single-scene render**: run 1 topic with `--scenes 1` (after fixing scene-indexing bug; see `understanding.md`)
3) **Full theorem render**: 1 topic end-to-end
4) **Eval**: run `evaluate.py --eval_type all` on that output folder

Video-eval-specific validation (do this before large sweeps):
- Run a “tiny chunk” OpenRouter video-eval request (base64 data URL) to a known video-capable Gemini model, then **increase chunk duration/bytes** until you observe consistent failures. OpenRouter warns constraints vary per provider/model, so you need empirical ceilings for your chosen routing setup ([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview), [OpenRouter provider routing guide](https://openrouter.ai/docs/guides/routing/provider-selection)).

Record:
- success/failure mode
- cost per run (LiteLLM accumulated cost)
- tokens or output length (optional)


