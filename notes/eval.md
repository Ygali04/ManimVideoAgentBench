# eval.md — evaluation plan (after `edits.md` is implemented)

This document describes how to evaluate **modern model families** on TheoremExplainBench (TEB) using this repo, and what additional analyses/plots would best demonstrate improvement over the results shown on the project homepage ([TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/)) and the tables you shared.

It assumes the repo has already been updated per `edits.md`:
- OpenRouter-first inference (single key)
- Modern model registry + aliases
- Video evaluation via OpenRouter **video-capable** models using `video_url` parts (base64 “data:” URLs for local mp4 chunks), with a **frame-based fallback** and an optional legacy Gemini SDK upload mode ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos), [OpenRouter models API](https://openrouter.ai/api/v1/models))

---

## 1) What are we measuring?

The original benchmark framing is:
- **Success rate**: can the agent generate a *complete* video artifact for a theorem?
- **Quality metrics** on successfully generated videos:
  - Text-only: accuracy/depth + logical flow
  - Visual: visual relevance + element layout
  - Video: visual consistency

These align to the 5 dimensions described on the homepage ([TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/)).

### 1.1 Define “success” precisely (important)
Use a strict definition so cross-model comparisons are fair:

**Theorem success = 1** iff:
- `output/<topic_slug>/<topic_slug>_combined.mp4` exists and is playable (duration > 0)
- `output/<topic_slug>/<topic_slug>_combined.srt` exists
- every scene folder has `succ_rendered.txt` (or equivalent success marker)

Record additional “partial success” states for debugging:
- plan exists but no code
- code exists but no render
- some scenes rendered but combine missing

This enables failure-mode plots (see §6).

---

## 2) Experimental grid (models × settings)

### 2.1 Model list
Use the new model registry aliases (examples):
- OpenAI: `gpt5`, `gpt5_thinking`, `gpt5_1`, `gpt5_1_thinking`, `gpt5_2`, `gpt5_2_thinking`
- Google: `gemini3_base`, `gemini3_flash`, `gemini3_pro`
- Anthropic: `claude45_haiku`, `claude45_sonnet`, `claude45_opus`
- Kimi: `kimi_k2`, `kimi_k2_thinking`
- DeepSeek: `deepseek_r1`
- Open-source baselines: at least 1–2 strong models (and 1 vision-capable if available)

Keep the model registry in version control so runs are reproducible.

### 2.2 Core toggles (baseline vs RAG)
For each model:
- **Baseline**: `--use_rag false`
- **RAG**: `--use_rag true` with the same `--manim_docs_path` and `--chroma_db_path`

### 2.3 Pass@N / attempts definition
The tables you shared use “N attempts” in the sense of **max code-fix retries**.
In this repo that corresponds to `--max_retries`.

Run a sweep:
- \(N \in \{0,1,2,3,4,5\}\)

Interpretation:
- `N=0`: generate code once; no fix loop
- `N=5`: allow up to 5 fix iterations per scene (largest “pass@N”)

> Optional extension: add a *true* pass@K across independent runs by re-running the full pipeline K times with different seeds/temps, and counting success if any run succeeds. That measures stochasticity rather than “repair ability”.

### 2.4 Concurrency / determinism knobs (keep constant)
To avoid confounds:
- Fix `--max_scene_concurrency` and `--max_topic_concurrency` across all models.
- Fix temperatures / reasoning settings using the model registry defaults.
- Record the exact prompt versions (hashes of `task_generator/prompts_raw/*` and `eval_suite/prompts_raw/*`).

---

## 3) Recommended run procedure

### 3.1 Dataset runs (TEB full)
Use the 240-theorem suite (80 each easy/medium/hard) across subjects (math/phys/cs/chem).

Practical suggestion:
- Start with **per-difficulty subsets** (e.g., 10 theorems each) for smoke testing.
- Then run full 240 once the harness is stable.

### 3.2 Output folder conventions
Use a deterministic directory naming scheme so evaluation can bulk-run:

```
output/
  runs/
    <date>/
      gen/<model_alias>/<baseline|rag>/N=<0..5>/<subject>/<difficulty>/
      eval/<judge_aliases>/...
```

Keep generation outputs immutable; never “reuse” a directory between different settings.

### 3.3 Generation commands (conceptual)
For each triple (model, rag_flag, N):
- run `generate_video.py` pointing to a theorems JSON (e.g., `data/thb_easy/math.json`)
- set:
  - `--model <model_alias_or_id>`
  - `--helper_model <same or cheaper helper>`
  - `--max_retries N`
  - `--use_rag` per condition

Capture logs (stdout + per-scene logs) for later failure analysis.

### 3.4 Evaluation commands (conceptual)
After generation completes, run `evaluate.py` over the output folder with:
- `--bulk_evaluate`
- `--eval_type all`
- fixed **judge models** (see §4)

---

## 4) Judge model strategy (avoid bias)

To compare generators fairly, hold judges constant.

### 4.1 Single-judge baseline (fast)
Pick one strong, stable judge for:
- text eval
- image eval
- video-consistency eval (prefer OpenRouter **video-capable** Gemini; fall back to frame-based if video payloads/providers are unreliable) ([OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos))

Practical selection rule:
- Query `https://openrouter.ai/api/v1/models` and pick a model whose `input_modalities` includes `"video"` ([OpenRouter models API](https://openrouter.ai/api/v1/models)).
- Example video-capable judge IDs from your provided facts: `google/gemini-3-pro-preview`, `google/gemini-3-flash-preview`, `google/gemini-2.5-flash`, `google/gemini-2.0-flash-lite-001` ([OpenRouter models API](https://openrouter.ai/api/v1/models)).

This produces one consistent leaderboard.

### 4.2 Panel-of-judges (better science)
Run multiple judge models and report:
- mean score
- variance across judges
- rank stability (Kendall tau)
- inter-rater reliability (Krippendorff’s alpha)

This is a “cool” addition not always elaborated in papers.

---

## 5) Core plots to reproduce + extend (beyond original tables)

### 5.1 Reproduce the originals
1) **Success rate by difficulty and subject** (baseline vs RAG) — Table 1 style.
2) **Metric scores** on successful videos — Table 2 style.
3) **Success vs N attempts** — Table 3 style (baseline/RAG curves).

### 5.2 Add plots the original paper likely didn’t emphasize

#### A) Pass@N curves with confidence intervals
- Plot success rate vs \(N\) with **Wilson intervals** or bootstrap CI.
- Separate panels for Easy/Medium/Hard and for subjects.

#### B) “Retries used” distribution
For successful theorems:
- histogram of how many fix iterations were actually needed (0..N)
- broken down by model family

This shows whether improvements come from “fewer repairs” vs “better repair skill”.

#### C) Failure taxonomy pie/stack plots
Categorize failures into:
- planning format failure (missing tags)
- code extraction failure
- Manim compile/runtime error
- combine failure (ffmpeg/srt)
- evaluation missing artifacts

Stacked bars by model and difficulty.

#### D) Cost–quality Pareto frontier
For each model:
- x-axis: median $ cost per successful theorem (LiteLLM cost logs)
- y-axis: overall score (or separate dimensions)

This is extremely useful for practical model selection.

#### E) Quality vs duration / length control
Compute:
- video duration distribution
- number of scenes
- average narration length (from SRT)

Plot correlation between duration and metric scores.

#### F) RAG contribution analysis
Track:
- number of RAG queries per stage
- total retrieved tokens injected
- whether retrieval included plugin docs vs core docs

Plot success improvement vs “retrieval volume” to see if RAG is helping in the intended way.

#### G) Text-vs-visual disagreement scatter
Scatter plot:
- x: text score (accuracy/depth, flow)
- y: visual/layout score

Highlight cases where text is high but visuals are poor (or vice versa), supporting the homepage narrative that multimodal eval reveals deeper issues ([TheoremExplainAgent site](https://tiger-ai-lab.github.io/TheoremExplainAgent/)).

---

## 6) Recommended reporting tables

For each generator model, report:
- **Success@N** for N=0..5 (baseline + RAG)
- **Overall score** (geometric mean) on successful videos
- Per-dimension breakdown (text, visual relevance, layout, consistency)
- Median time per theorem (optional)
- Median cost per theorem (optional)

Also report:
- “best N” (argmax success@N) and “diminishing returns” (delta between N=4 and N=5)

---

## 7) Practical tips / pitfalls

- Fix scene-indexing and partial-render behaviors before running large sweeps (see notes in `understanding.md`).
- Ensure the evaluation harness uses the same judge prompts across all runs (version them).
- If you rely on OpenRouter **video** inputs, treat payload and provider constraints as first-class experimental variables:
  - keep mp4 chunks small enough to avoid request-size/timeouts (OpenRouter notes constraints vary by provider/model) ([OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview))
  - explicitly record the chosen video judge model ID and (if used) provider routing configuration ([OpenRouter provider routing guide](https://openrouter.ai/docs/guides/routing/provider-selection))
  - implement and log the frame-based fallback path for any chunk that fails video upload/inference
  - beware schema/client compatibility: OpenRouter’s `video_url` examples may not match strict OpenAI validators, so prefer a direct HTTP path for video eval, or test your client library carefully ([OpenRouter API overview](https://openrouter.ai/docs/api-reference/overview), [OpenRouter video guide](https://openrouter.ai/docs/guides/overview/multimodal/videos))
- Always store:
  - generation config (model IDs, temps, N, rag on/off)
  - git commit hash
  - prompt file hashes
  - judge model IDs

This is essential to make “model improvements” claims defensible.


