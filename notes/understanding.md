# Internal repo notes: TheoremExplainAgent (TEA)

These are **developer-facing, implementation-level notes** intended to help a contributor understand how this repo works end-to-end (generation/inference, RAG, evaluation), and where to extend it to new model providers.

Sources of truth are the code in this repo and the project homepage ([TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/)).

---

## Repo map (high-signal)

- **Entrypoints**
  - `generate_video.py`: generation pipeline (plan → code → render → combine).
  - `evaluate.py`: automated evaluation (text/video/image).
- **Core pipeline**
  - `src/core/video_planner.py`: scene outline + per-scene “subplans” (storyboard, technical plan, narration plan).
  - `src/core/code_generator.py`: Manim code generation + retry-format extraction + error fixing + (optional) visual self-reflection.
  - `src/core/video_renderer.py`: runs `manim` to render each scene; combines scene videos + SRT into a final mp4/srt.
- **Model calling**
  - `mllm_tools/litellm.py`: `LiteLLMWrapper` — main provider-agnostic wrapper (OpenAI/Azure/Bedrock/etc via LiteLLM).
  - `mllm_tools/gemini.py`: `GeminiWrapper` — direct Google Generative AI SDK wrapper (not LiteLLM).
  - `mllm_tools/vertex_ai.py`: `VertexAIWrapper` — Vertex Gemini wrapper (not currently wired into generation CLI).
  - `src/utils/allowed_models.json`: allowlist gate for model names accepted by CLI for most places.
- **RAG**
  - `src/rag/vector_store.py`: Chroma vector DB creation/loading + retrieval (LangChain + Chroma + LiteLLM embeddings).
  - `src/rag/rag_integration.py`: “glue” for plugin detection + per-stage query generation.
- **Evaluation suite**
  - `eval_suite/text_utils.py`: transcript parsing/fixing + text-eval.
  - `eval_suite/video_utils.py`: chunked video eval (Gemini video model).
  - `eval_suite/image_utils.py`: sampled-frame eval (image model).
  - `eval_suite/prompts_raw/*`: evaluation prompt templates (JSON-scored).
- **TTS**
  - `src/utils/kokoro_voiceover.py`: `KokoroService` speech service for `manim-voiceover`.

---

## Generation / inference pipeline (what happens when you run `generate_video.py`)

### CLI modes (top-level switchboard)
`generate_video.py` supports two usage patterns:

1) **Single topic**: `--topic` + `--context`  
2) **Batch**: `--theorems_path` points to a JSON file like `data/thb_easy/math.json`.

Important control flags:
- `--only_plan`: produce outline + per-scene implementation plans, **no render**.
- `--only_render`: render scenes **only if no code exists** (note: uses a global `args` inside `VideoGenerator.generate_video_pipeline`).
- `--only_gen_vid`: “render using existing plans” path (bypasses planning).
- `--only_combine`: only run final combine step (ffmpeg concat + SRT merge).
- `--use_rag`: enable retrieval for planning + code generation + error fixing.
- `--use_context_learning`: inject example plans/code from `data/context_learning/**`.
- `--use_visual_fix_code`: enable “visual self-reflection” (render → critique with VLM/image model → rewrite code).
- `--max_scene_concurrency`, `--max_topic_concurrency`: async concurrency controls.

### Trace IDs, session IDs, and logging metadata
There are two “identity” concepts used throughout generation:

- **`session_id`** (per run/output_dir)
  - `VideoGenerator._load_or_create_session_id()` persists to `output_dir/session_id.txt`.
  - A per-topic copy is saved to `output_dir/<topic_slug>/session_id.txt`.
  - Many model calls pass `metadata={"session_id": session_id, ...}` which is mainly used by Langfuse callback integration.

- **`scene_trace_id`** (per scene)
  - `VideoPlanner._generate_scene_implementation_single(...)` creates a UUID and writes it to `scene{i}/subplans/scene_trace_id.txt`.
  - `generate_video.py` later tries to load that file when rendering; if missing, it creates it.
  - This `scene_trace_id` is passed into `metadata={"trace_id": scene_trace_id, ...}` for model calls, and is also used by RAG spans.

Net: `session_id` groups a run; `scene_trace_id` groups all “generations” for a particular scene (plans, codegen, fixes, RAG retrievals).

### Model objects instantiated by `generate_video.py`
`generate_video.py` instantiates **three** wrappers (all `LiteLLMWrapper` by default):
- `planner_model`: used by `VideoPlanner` for outline + per-scene subplans.
- `helper_model`: used for “helper” tasks (RAG query generation and plugin detection, depending on component).
- `scene_model`: used by `CodeGenerator` for code generation and fix loops.

The model name strings come from CLI args and are restricted by `src/utils/allowed_models.json`.

### `VideoGenerator` orchestration (main coordinator)
`VideoGenerator` (in `generate_video.py`) wires:
- `self.planner = VideoPlanner(...)`
- `self.code_generator = CodeGenerator(...)`
- `self.video_renderer = VideoRenderer(...)`

It also manages:
- `session_id`: persisted to `output_dir/session_id.txt` and copied per topic to `output_dir/<topic_slug>/session_id.txt`.
- `scene_semaphore`: enforces `max_scene_concurrency`.

### Output layout (topic slugging)
Every topic is “slugged”:
- `file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())`

All outputs live under:
- `output_dir/<file_prefix>/...`

Key artifacts:
- `output_dir/<file_prefix>/<file_prefix>_scene_outline.txt`
- `output_dir/<file_prefix>/scene{i}/`
  - `<file_prefix>_scene{i}_implementation_plan.txt`
  - `subplans/` (vision storyboard / technical plan / narration plan text files + `scene_trace_id.txt`)
  - `code/` (versioned python files + logs + error logs)
  - `succ_rendered.txt` (marker written by `VideoRenderer` on successful render)
- `output_dir/<file_prefix>/media/` (Manim media dir used for rendering)
- `output_dir/<file_prefix>/<file_prefix>_combined.mp4` and `..._combined.srt` (final combined outputs)

### Planning (outline → per-scene implementation plan)
**Outline**
- `VideoPlanner.generate_scene_outline(topic, description, session_id)`
  - Calls `task_generator.get_prompt_scene_plan()`
  - Expects output containing `<SCENE_OUTLINE> ... </SCENE_OUTLINE>` (XML-ish)
  - Saves `<file_prefix>_scene_outline.txt`

**Per-scene implementation plan**
`VideoPlanner._generate_scene_implementation_single(...)` generates, in order:
1) `<SCENE_VISION_STORYBOARD_PLAN>`
2) `<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>`
3) `<SCENE_ANIMATION_NARRATION_PLAN>`
Then concatenates these into the scene’s `*_implementation_plan.txt`.

If `--use_rag` is enabled, planning also:
- Detects relevant plugins (once per topic) and passes them to the prompt.
- Generates RAG queries per stage (storyboard/technical/narration) and injects retrieved docs into the stage prompt.

### Code generation (implementation plan → Manim python)
`CodeGenerator.generate_manim_code(...)`:
- Builds prompt using `task_generator.get_prompt_code_generation(...)`.
  - Requires the model to output fenced code: ```python ...```
- Extracts code via `_extract_code_with_retries()` which will re-prompt the model up to ~10 times if the pattern isn’t matched.
- If `--use_rag`, it also:
  - Generates RAG queries for code stage (`rag_queries_code.json` cached per scene)
  - Retrieves docs via `RAGVectorStore.find_relevant_docs(k=2)` and appends them to `additional_context`.

### Render + fix loop (Manim compiler feedback)
Per scene, `VideoGenerator.process_scene()`:
1) Generates initial code, writes:
   - `..._v0_init_log.txt`
   - `..._v0.py`
2) Calls `VideoRenderer.render_scene(...)`
3) On error: increments version and calls `CodeGenerator.fix_code_errors(...)`, writes:
   - `..._v{n}_fix_log.txt`
   - `..._v{n}.py`
4) Stops on success or after `--max_retries`.

### Concurrency model (what runs in parallel)
- **Across topics**: `--max_topic_concurrency` controls an `asyncio.Semaphore` around “process one theorem/topic”.
- **Within a topic**: `--max_scene_concurrency` controls `VideoGenerator.scene_semaphore`, which gates `process_scene()` bodies.

Important implementation detail: `render_video_fix_code()` uses `asyncio.gather(*tasks)` (one task per scene) but each task does `async with self.scene_semaphore`, so concurrency is bounded.

### Combine (final mp4 + merged subtitles)
`VideoRenderer.combine_videos(topic)`:
- Locates the “latest version” folder per scene inside `output_dir/<file_prefix>/media/videos/**`.
- Concatenates videos with ffmpeg; special-cases “missing audio stream” by injecting silent audio.
- Merges `.srt` subtitles by applying **time offsets** accumulated by video durations.

> Note: the per-scene `.srt` files are assumed to be produced by `manim-voiceover` when scenes are written using `VoiceoverScene` + `with self.voiceover(...)`.

---

## Prompt contracts (planner ↔ codegen ↔ renderer)

This repo is extremely prompt-contract-driven. The important raw templates are in:
- `task_generator/prompts_raw/`

Key expected output formats:
- Scene outline: `<SCENE_OUTLINE> ... <SCENE_1> ... </SCENE_1> ... </SCENE_OUTLINE>`
- Storyboard plan: `<SCENE_VISION_STORYBOARD_PLAN> ... </SCENE_VISION_STORYBOARD_PLAN>`
- Technical plan: `<SCENE_TECHNICAL_IMPLEMENTATION_PLAN> ... </SCENE_TECHNICAL_IMPLEMENTATION_PLAN>`
- Narration plan: `<SCENE_ANIMATION_NARRATION_PLAN> ... </SCENE_ANIMATION_NARRATION_PLAN>`
- Codegen/fix outputs: fenced ```python ...``` blocks (wrapped in `<CODE>` in the codegen prompt, but extraction looks only for fenced python)
- Plugin detection: fenced ```json [...]``` block

Two recurring global constraints are pushed into prompts:
- **Safe area margins**: 0.5 units
- **Minimum spacing**: 0.3 units

The system tries to “push layout quality” primarily through prompts (not through runtime validators).

---

## Generation sharp edges (things that matter operationally)

These are not “fixes” (we’re not editing code here), but they are important to know when running experiments.

### Partial-scene rendering can mis-number scenes
In `VideoGenerator.generate_video_pipeline()`, the code computes which scenes need processing and builds `filtered_implementation_plans` by **dropping the original scene numbers**. That list is passed to `render_video_fix_code()`, which enumerates from 1..N and writes to `scene1`, `scene2`, etc.

Implication: if you try to render only “scene 3”, it may be treated as “scene 1” for folder naming, which can overwrite or corrupt the topic folder layout.

### `use_visual_fix_code` path likely fails as-is
`VideoRenderer.render_scene()` references `self.scene_model`, but `VideoRenderer` never sets that attribute. If `--use_visual_fix_code` is enabled, it may crash when trying to decide whether to pass a video vs image to the model.

### Assumed media folder naming may differ from Manim’s actual output
Multiple helper paths assume scene video files live under folders named like:
`media/videos/<file_prefix>_scene{scene}_v{version}/1080p60/*.mp4`

Manim’s real naming often includes the python filename and the scene class name; if those diverge, snapshot selection and combine logic can miss outputs.


---

## Model calling / provider wiring (how LLMs are invoked)

There are two “model wrapper” layers in this repo:

1) **`LiteLLMWrapper`** (`mllm_tools/litellm.py`): main wrapper used by **generation** and most **evaluation** steps  
2) **Direct SDK wrappers** (`mllm_tools/gemini.py`, `mllm_tools/vertex_ai.py`): only used in limited places today (notably video eval uses `GeminiWrapper`)

### `LiteLLMWrapper` interface and behavior
Call signature:
- `LiteLLMWrapper(messages: List[{"type": ..., "content": ...}], metadata: Optional[dict]) -> str`

Message types accepted by wrapper:
- `"text"`: always supported
- `"image"`, `"audio"`, `"video"`: wrapper attempts to support these, but in practice it **only supports images for GPT** and treats other media types inconsistently

Under the hood:
- Converts input messages into `litellm.completion(...)` OpenAI-style chat payloads.
- Uses `max_retries=99` at the LiteLLM layer.
- Optionally tracks cost via `litellm.completion_cost(...)` and prints accumulated cost.
- If `use_langfuse=True`, sets global callbacks:
  - `litellm.success_callback = ["langfuse"]`
  - `litellm.failure_callback = ["langfuse"]`

Special-casing:
- If model name matches OpenAI “o-series” patterns (`^o\d+` or `openai/o...`), wrapper forces:
  - `temperature = None`
  - `reasoning_effort = "medium"`

### Multimodal limitations in `LiteLLMWrapper` (important)
The wrapper’s multimodal formatting branches on substring checks:
- if `"gemini" in model_name` → sends a `{"type":"image_url","image_url": data_url}` content item
- elif `"gpt" in model_name` → sends OpenAI image format (but explicitly raises for non-image media)
- else → raises: “Only support Gemini and Gpt for Multimodal capability now”

Implications:
- Vision-capable non-GPT models (Anthropic, DeepSeek, Kimi, open-source) are **blocked** from image usage even if the upstream provider supports it.
- “video” inputs are **not** supported in the LiteLLM path.
- In generation, even when you select `gemini/...` in CLI, `generate_video.py` still constructs a `LiteLLMWrapper` (not `GeminiWrapper`), so “upload a video file to Gemini” is not available.

### OpenRouter note: video is feasible via `video_url` parts (post-edits)
If you migrate inference/evaluation to OpenRouter, you can keep the existing evaluator *pattern* (“split video → evaluate chunks → aggregate”) but change the **transport**:
- OpenRouter supports video inputs on `/api/v1/chat/completions` via `video_url` content parts and allows **base64 “data:” URLs** for local files ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)).
- Whether a given model supports video is discoverable from the OpenRouter models API via `input_modalities` containing `"video"` ([OpenRouter models API](https://openrouter.ai/api/v1/models)).

Practical potholes to account for when implementing this:
- Provider/model constraints vary; OpenRouter does not publish a single global max payload/timeout ([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview)).
- For Gemini routed via certain providers, video URLs can have restrictions (e.g., YouTube-only links), which makes base64 data URLs and/or provider routing controls important ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)).
- Schema/client mismatch risk: OpenRouter’s video examples use `video_url` parts but some examples use camelCase nested fields (e.g., `videoUrl`). Strict “OpenAI-only” client schemas may reject or rewrite these; for reliability, implement video eval as direct HTTP to OpenRouter or verify your client library’s passthrough behavior ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter API overview](https://openrouter.ai/docs/api-reference/overview)).

### Direct SDK wrappers

**`GeminiWrapper`** (`mllm_tools/gemini.py`)
- Uses `google.generativeai` SDK directly
- Requires env var: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Supports uploading local files (including video) via `genai.upload_file(...)`
- Used in this repo for:
  - `evaluate.py`’s **video evaluation model** (chunked mp4 evaluation)

**`VertexAIWrapper`** (`mllm_tools/vertex_ai.py`)
- Uses `google-cloud-aiplatform` Vertex SDK
- Requires env var: `GOOGLE_CLOUD_PROJECT` (and standard ADC credentials)
- Not wired into generation CLI today, but referenced in `CodeGenerator.visual_self_reflection()` as a possible “video-capable” wrapper type.

### Allowed models / CLI gating
There is a static allowlist file:
- `src/utils/allowed_models.json`

Both:
- `generate_video.py` and
- `evaluate.py`
use it (or variants) to constrain allowed `--model` choices. Additionally, `evaluate.py` hard-restricts `--model_video` to a small set of Gemini model IDs.

Net effect: adding a new provider/model typically requires updating `allowed_models.json` and possibly loosening CLI restrictions (or introducing a richer model registry).

---

## RAG subsystem (docs ingestion → retrieval → prompt injection)

RAG is implemented with:
- LangChain document loaders + splitters
- Chroma persisted vector stores
- LiteLLM embeddings (remote) for vectorization

Main modules:
- `src/rag/vector_store.py`: `RAGVectorStore`
- `src/rag/rag_integration.py`: `RAGIntegration`

### Data layout expected on disk
When `--use_rag` is enabled, the repo expects Manim docs in a folder like:
- `data/rag/manim_docs/`
  - `manim_core/` (markdown + python docs)
  - `plugin_docs/`
    - `plugins.json` (list of plugins and descriptions; used for plugin detection)
    - `<plugin_name>/` (markdown/python docs per plugin)

Chroma persistence lives in:
- `data/rag/chroma_db/`
  - `manim_core/`
  - `manim_plugin_<plugin_name>/` (one per discovered plugin folder)

### Vector store creation and embeddings
`RAGVectorStore` does:
- “load if exists, else build” for the core store and each plugin store.
- Loads `.md` and `.py` files and splits using `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN/PYTHON)`.
- Prefixes each chunk with `Source: <path>` to preserve provenance in prompt injection.
- Uses a custom LangChain `Embeddings` class that calls `litellm.embedding(...)`.

Important embedding details:
- For `vertex_ai/text-embedding-005`, passes `task_type="CODE_RETRIEVAL_QUERY"`.
- Temporarily disables Langfuse callbacks during embedding calls, then restores them.

OpenRouter embedding note (post-edits):
- OpenRouter provides an embeddings endpoint (`POST /api/v1/embeddings`) and a listing endpoint for embedding models, which enables a “single OpenRouter key” setup for RAG indexing if desired ([OpenRouter embeddings: create](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings), [OpenRouter embeddings: list models](https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models)).

### Query shape and retrieval behavior
Retrieval expects queries shaped like:
```json
[
  {"type": "manim-core", "query": "..."},
  {"type": "<plugin_name>", "query": "..."}
]
```

Retrieval flow:
- Splits into core queries and plugin queries.
- Runs `similarity_search_with_relevance_scores(k=<k>, score_threshold=0.5)` in the appropriate store.
- Deduplicates results by `content`.
- Formats a single big string containing core + plugin results (including scores) and returns it for prompt injection.

### Where RAG is used in generation
RAG can influence multiple stages:

**Planning stage** (`VideoPlanner`):
- Detects relevant plugins once per topic (`RAGIntegration.detect_relevant_plugins` reading `plugin_docs/plugins.json`).
- For each scene stage (storyboard → technical → narration), generates queries and injects retrieved docs directly into the prompt.

**Codegen stage** (`CodeGenerator`):
- Generates code-stage RAG queries (`rag_queries_code.json` cached under `scene{i}/rag_cache/`).
- Retrieves docs and appends to `additional_context` for code generation.
- Similarly, for error-fixing, generates `rag_queries_error_fix.json` and injects docs into the fix prompt.

### Caching behavior (per scene)
Query generation results are cached under:
- `output/<topic_slug>/scene<i>/rag_cache/*.json`

This avoids re-querying the helper model for the same scene repeatedly.

---

## Evaluation pipeline (how `evaluate.py` scores outputs)

Evaluation entrypoint:
- `evaluate.py`

It expects:
- a theorem folder containing a video (`.mp4`) and transcript (`.srt` preferred) and evaluates in modes:
  - `text`, `video`, `image`, or `all`

### Judges used (current repo default)
- **Text judge**: `LiteLLMWrapper(model_name=--model_text)`
- **Image judge**: `LiteLLMWrapper(model_name=--model_image)`
- **Video judge**: `GeminiWrapper(model_name=--model_video)` (direct Gemini SDK upload)

OpenRouter evaluation note (post-edits):
- You can keep “mp4 chunk → judge → aggregate” but replace the Gemini upload flow with OpenRouter `video_url` content parts (base64 data URLs) to a video-capable model ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos)).
- Because OpenRouter constraints vary by provider/model, implement a frame-based fallback path for chunks that fail or for judge models that don’t support video ([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview)).

### Text evaluation
Implemented in:
- `eval_suite/text_utils.py`

Flow:
- Read transcript (`.srt` → plain text).
- If transcript looks “all lowercase” (very low uppercase proportion), run a transcript-fixing pass using an LLM with `eval_suite/prompts_raw/fix_transcript.txt`.
- Evaluate with `eval_suite/prompts_raw/text_eval_new.txt`, which returns JSON with scores (1–5) for:
  - accuracy/depth
  - logical flow

### Video evaluation (chunked)
Implemented in:
- `eval_suite/video_utils.py`

Flow:
- Split the full video into **10 chunks** (moviepy).
- Optionally reduce FPS for processing (`--target_fps`).
- Evaluate each chunk with prompt `eval_suite/prompts_raw/video_eval_new.txt` (currently only “visual consistency”).
- Aggregate chunk scores using **geometric mean**.

### Image evaluation (sampled frames)
Implemented in:
- `eval_suite/image_utils.py`

Flow:
- Sample key frames by dividing the video into `num_chunks` segments and selecting a representative frame per segment (uses “most non-black pixels” heuristic).
- Evaluate each key frame with prompt `eval_suite/prompts_raw/image_eval.txt` (visual relevance + element layout).
- Aggregate frame scores using **geometric mean**.

### Overall score
`evaluate.py` computes “overall score” as a geometric mean over all extracted `score` fields in the final merged JSON (skipping chunk detail keys).

---

## Extension notes (new providers/models)

The fastest path to “evaluate modern models” is to:
- decouple model selection from hardcoded allowlists
- standardize inference through a single OpenAI-compatible surface

See:
- `notes/edits.md` for the concrete recommended code edits (OpenRouter-first, model registry, multimodal handling fixes).
- `notes/eval.md` for how to run and what to plot to show improvements relative to the baseline results on the homepage ([TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/)).
