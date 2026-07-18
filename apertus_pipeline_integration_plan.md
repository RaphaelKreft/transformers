# Apertus Pipeline Integration Plan

This document specifies how to extend Apertus into **Apertus 1.5** — a discrete-token **early-fusion** model that
consumes **image + audio + text** and generates **text only**. The chosen design bundles the EMU3.5 vision
tokenizer and the WavTokenizer audio tokenizer as native, **encode-only** sub-models (the Emu3 / Chameleon
pattern). A reference for the current text-only Apertus integration is included at the end.

**Contents**
- [Approach](#approach)
- [Conceptual template: Emu3](#conceptual-template-emu3)
- [Model](#model)
- [Token stream & vocab layout](#token-stream--vocab-layout)
- [Batched inference: variable images per prompt](#batched-inference-variable-images-per-prompt)
- [Config-driven sizes](#config-driven-sizes)
- [Registration](#registration)
- [Porting EMU3.5 and WavTokenizer](#porting-emu35-and-wavtokenizer)
- [Implementation steps](#implementation-steps)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Reference: vLLM integration (vllm_swissai)](#reference-vllm-integration-vllm_swissai)
- [Reference: the existing Apertus model](#reference-the-existing-apertus-model)
- [Reference: standard workflow for adding a model to Transformers](#reference-standard-workflow-for-adding-a-model-to-transformers)

---

## Integration plan

### Approach
**Model-side bundled encoders (Emu3 / Chameleon pattern).** EMU3.5 + WavTokenizer are ported as native,
**encode-only** sub-models of an `apertus1p5` model. Discrete-token **early fusion**: both tokenizers are
single-codebook → one flat stream in an extended vocab; **X→Text only** ⇒ no decode path, no per-codebook
heads/summing. `input_modalities = ("image", "audio", "text")`, `output_modalities = ("text",)`.

### Conceptual template: Emu3
Emu3 (`src/transformers/models/emu3/`) is the closest existing model: discrete VQ image tokens fused into a
**Llama-derived** text backbone (`Emu3TextModel(LlamaModel, …)`) via the *shared* embedding — the same shape as
Apertus. Mirror its skeleton and delete the generation half:
- **Config nesting:** `Emu3Config.sub_configs = {"text_config": Emu3TextConfig, "vq_config": Emu3VQVAEConfig}`
  (a name→class dict) + `vocabulary_map` (`configuration_emu3.py:123-149`).
- **Composition:** `Emu3Model = text_model + vqmodel + vocabulary_mapping` (`modular_emu3.py:914-919`; the text
  model is built via `Emu3TextModel._from_config(config.text_config)`, `:917`). `forward` embeds `input_ids`, then
  `get_image_features` (vq `encode` → `convert_img2bpe` → shared `get_input_embeddings()`, `:953-975`) and
  `masked_scatter`s the result into placeholder positions (`:1046-1054`; mask from `get_placeholder_mask`,
  `:996-1018` — a **per-model** method, not a shared `modeling_utils` helper). `Emu3ForConditionalGeneration`
  (`:1069`).
- **Drop:** `decode_image_tokens` (`:978-994`), `output_modalities = ("image", "text")` (the `GenerationMixin`
  default is already `("text",)`), and the processor's `return_for_image_generation` path. Emu3 has **no audio** —
  wire the WavTokenizer sub-model the way **kyutai_speech_to_text / CSM** plug `MimiModel` via
  `AutoModel.from_config(config.codec_config)` (`modular_csm.py:433`, `modular_kyutai_speech_to_text.py:266`;
  config side: `sub_configs = {"codec_config": AutoConfig}` + `AutoConfig.for_model(...)` default in
  `__post_init__`).
- **Not reused as-is:** Emu3's bundled `Emu3VQVAE` is the 3.0 plain VQ (nearest-neighbour
  `Emu3VQVAEVectorQuantizer`, `modular_emu3.py:97-129`), not EMU3.5's IBQ (see below).
- **⚠ Layout deviation from Emu3 — do not copy its count math.** Emu3's processor expands `<image>` to
  `H*(W+1)` placeholders (the EOL column is *scattered together with* the image features,
  `processing_emu3.py:131`) and inserts an extra `eof_token` before `eoi`. The Apertus 1.5 stream (vLLM
  reference) instead joins rows with **`H−1` real `<|img_end_of_row|>` tokens** and has no `eof`. So the
  Apertus1p5 processor must emit all structure tokens itself and expand **only the `H*W` code positions** as
  placeholders; model-side `get_image_features` returns a flat `H*W` (no EOL column, unlike `convert_img2bpe`).
  See [Token stream & vocab layout](#token-stream--vocab-layout).

### Model
**One model dir, authored entirely in `modular_apertus1p5.py`**: nested config, composed model, generation
wrapper, and a processor that never runs the encoders.
- `Apertus1p5Config`: `sub_configs = {"text_config": AutoConfig (→ apertus), "vision_tokenizer_config": …,
  "audio_tokenizer_config": …}`; `image_token_id`, `audio_token_id`, boundary ids. The extended vocab lives in
  **`text_config.vocab_size`** (it sizes `embed_tokens` + `lm_head`).
- `Apertus1p5Model = AutoModel.from_config(text_config) + vision_tokenizer + audio_tokenizer + vocabulary_mapping`;
  `get_image_features` / `get_audio_features`: encode → code→vocab-id mapping (image: vocabulary map / offset;
  audio: `+audio_token_offset`, the Kyutai-style offset) → shared `embed_tokens` → `masked_scatter` into
  placeholder positions (`get_placeholder_mask`, defined on the model as in Emu3). Set
  `input_modalities = ("image", "audio", "text")` on the pretrained-model base (`PreTrainedModel` default is
  `"text"`, `modeling_utils.py:1231`).
- `Apertus1p5ForConditionalGeneration`: `lm_head`, **no** decode methods. `output_modalities` needs no override —
  the `GenerationMixin` default is already `("text",)` (`generation/utils.py:370`).
- **Processor:** emits `pixel_values` (+ `image_sizes`) / audio `input_values` and expands `<image>`/`<audio>`
  placeholders into the full structural layout with the exact per-media token *count* — it does **not** run the
  encoders (same contract as `Emu3Processor`, `processing_emu3.py:119-154`, but implemented via the generic
  `ProcessorMixin` hooks — see
  [Batched inference: variable images per prompt](#batched-inference-variable-images-per-prompt)). Components:
  - *image processor*: area-clamp + `smart_resize` to multiples of 16 (bicubic), normalize `x/127.5 − 1`, and
    return `image_sizes` so the processor can derive the token grid `H = h_px/16`, `W = w_px/16` per image;
  - *audio feature extractor*: mono, resample to 24 kHz, **peak-normalize to −3 dBFS** — and reproduce
    WavTokenizer's exact frame-count arithmetic (codes per sample count), since a placeholder-count vs.
    encoder-output mismatch makes `masked_scatter` fail;
  - *tokenizer*: the checkpoint's, which already contains all special/visual tokens.

### Token stream & vocab layout
Byte-level ground truth from `vllm_swissai` (`vllm/model_executor/models/apertus.py`), which the transformers
processor + model must reproduce **exactly**:

- **Image** (`build_apertus_image_prompt`, `:248-271`):
  `<|img_start|>{H}*{W}<|img_token_start|> row₀ <|img_end_of_row|> row₁ … row_{H−1} <|img_end|>`
  where `H`/`W` are **token-grid** dims (pixels ÷ 16), rendered as plain text digits (header token count is
  therefore data-dependent), each rowᵢ is `W` visual tokens, and rows are **join**ed by EOL → `H−1` EOLs, no
  trailing EOL, no `eof`. Per-image total = `1 (boi) + len(tok("H*W")) + 1 (img) + H·W + (H−1) + 1 (eoi)`.
- **Audio** (`encode_audios`, `:420-466`): `<|audio_start|> code₀ … code_{N−1} <|audio_end|>`, `N` = WavTokenizer
  frame count (~40/s at 24 kHz).
- **Code→vocab-id mapping is asymmetric in vLLM:** image codes go through **token strings**
  (`VISUAL_TEMPLATE = "<|visual token {id}|>"`, `:170`) looked up in the tokenizer vocab; audio codes get a
  **numeric offset** `+262344` (`DEFAULT_AUDIO_TOKEN_OFFSET`, `:346`). Model-side both become numeric: store an
  image vocabulary map (Emu3-style, from the tokenizer's added tokens) or a plain offset **after verifying the
  `<|visual token N|>` ids are contiguous and in order** in the checkpoint tokenizer, plus the audio offset, in
  `Apertus1p5Config`.
- **Vocab arithmetic** (confirm against the real checkpoint): `262344 = 2·131072 + 200` ⇒ layout is
  131,072 text + 131,072 visual + ~200 specials, then the WavTokenizer codebook (4,096 in the paper — confirm)
  ⇒ extended `vocab_size` ≈ **266.5k ≈ 2×** the text vocab (not 3×).
- In transformers `input_ids`, the `H·W` / `N` code positions hold the `image_token_id` / `audio_token_id`
  placeholder (real code ids exist only after the model-side encoders); all structure tokens (boi/eoi/img/eol/
  digits/audio_start/audio_end) are real ids emitted by the processor. Final embeddings match vLLM because
  `embed_tokens(real ids)` is scattered into those positions.

### Batched inference: variable images per prompt
**Decision: follow the standard flatten-and-order convention** — `images` nested *or* flat, flat
`pixel_values` + `image_sizes`, positional sample ownership, one row-major `masked_scatter` — hardened with
explicit count errors and **per-image encoding** for bit parity. Scope: `text=[A, B, C]` with e.g. 2 / 1 / 7
images per prompt, processor + `forward` (generation excluded); surveyed from Emu3, Qwen2-VL, LLaVA-OneVision.

#### The transformers convention (survey)
- **Input format:** `images` may be a flat list or a **nested list-of-lists (one sub-list per sample)** — the
  nested form is "the format supported by all multimodal processors" (`tests/test_processing_common.py:861`).
  Normalizers: `make_flat_list_of_images` / `make_nested_list_of_images` (`image_utils.py:202-276`).
- **Sample ownership is positional, not structural.** Images are flattened batch-wide and bound to `<image>`
  placeholders **left-to-right, sample-by-sample** via one iterator — generic path:
  `ProcessorMixin._process_images` → `replace_image_token(…, image_idx)` → `get_text_with_replacements`
  ("the i-th occurrence of `self.image_token` is replaced by `images_replacements[i]`",
  `processing_utils.py:757-767, 802-905`); Emu3 does the same with a hand-rolled loop
  (`iter(image_sizes)` + `next()` per placeholder, `processing_emu3.py:122-137`). Only LLaVA-OneVision adds an
  explicit per-sample count tensor (`batch_num_images`) — optional, not required by the mechanism.
- **Variable resolutions:** Emu3 resizes each image independently, **pads `pixel_values` to the batch max H/W**
  and returns per-image `image_sizes` (`image_processing_emu3.py:253-296, 408-415`); Qwen2-VL instead flattens
  patches to `(total_patches, dim)` + `image_grid_thw`; LLaVA-OV pads the patch dim + `image_sizes`.
- **Model side:** encode, recover each image's token count from `image_sizes`
  (`split_sizes`, `modular_emu3.py:963-966`), `torch.cat` the variable-length per-image features **in flat order**
  (`:1050`), then `get_placeholder_mask` (`input_ids == image_token_id`) + `masked_scatter`, which fills `True`
  positions in **row-major batch order** (`:996-1018, 1054`). The only guard is a *global* count check
  (`"Image features and image tokens do not match, tokens: …, features: …"`, `torch_compilable_check`) — per-sample
  alignment relies entirely on ordering.
- **Sequence padding** is plain tokenizer work (`padding=True` → `attention_mask`,
  `processing_utils.py:686`); nothing image-specific for a padded `forward` (left-padding is a *generation*
  convention only). `_check_special_mm_tokens` validates placeholder counts **per sample** after tokenization
  (`processing_utils.py:2310-2326`).
- **Pipelines:** `image-text-to-text` / `any-to-any` accept multi-image chats (media inside message `content`) and
  collate independently processed samples by concatenating `pixel_values` along dim 0 (`pipelines/base.py:95-98`)
  — the flat-image layout below survives this unchanged.

#### Design for Apertus1p5
- **Processor input:** `text: list[str]` (or chat template) + `images` flat **or nested**; normalize with
  `make_flat_list_of_images`. Use the **generic `ProcessorMixin` path** rather than Emu3's hand-rolled loop:
  implement `replace_image_token(processed_images, image_idx)` returning the full structural expansion for image
  *i* (`boi + digits("H*W") + img + H rows of W placeholders joined by eol + eoi` — per-image strings are exactly
  what the hook expects, and it gives batch-wide ordered consumption for free). Audio uses the same hook family
  for `<|audio|>`.
- **Tensors / ownership:** `pixel_values (num_images_total, 3, maxH, maxW)` padded to batch max (Emu3-style)
  + `image_sizes (num_images_total, 2)`; audio `input_values` padded + per-clip lengths. No `batch_num_images`
  tensor — ownership stays positional (flat order == placeholder order), matching Emu3/Qwen2-VL.
- **Validation / errors (be stricter than Emu3, which leaks `StopIteration` and silently drops extra images):**
  - placeholders ≠ images → explicit `ValueError` in `__call__` (message style of
    `internvl/processing_internvl.py:212`: "Number of image placeholders in the prompt does not match the number
    of images.");
  - truncation losses → `_check_special_mm_tokens` (per-sample, automatic);
  - model side → the standard `get_placeholder_mask` global check (exact Emu3 wording).
- **Model-side scattering:** `get_image_features(pixel_values, image_sizes)` returns per-image code embeddings of
  length `H_i·W_i`; concat flat; one `masked_scatter` over the batch. **Deviation from Emu3 for parity:** Emu3
  encodes the *padded* batch and crops token grids afterwards (`modular_emu3.py:750-781`) — with conv receptive
  fields (and any encoder attention) the padded zeros can perturb codes vs. encoding each image alone, and vLLM
  encodes **per image** (`encode_images` loop, `apertus.py:287-320`). So Apertus1p5 must **crop each image to its
  `image_size` and encode individually** (or bucket identical sizes); bit-exact codes are the correctness gate.
- **Context / memory:** worst-case ≈ 7.9k tokens per max-res image (88×88 grid + EOLs + header; ceiling 8168) vs.
  `max_position_embeddings = 65536` — prompt C's 7 large images ≈ 55k tokens alone. No implicit truncation:
  let the length check / OOM surface naturally, document the budget in the model doc page. Padded pixel batches
  are cheap next to that (10 max-res F32 images ≈ 235 MB); the per-image encode loop bounds encoder peak memory.
- **Tests** (mirror existing patterns):
  - *processor unit:* nested heterogeneous batch `[[], [img], [img]·7]` (0/1/many — the
    `test_processor_text_has_no_visual` pattern, `tests/test_processing_common.py:836`, via `ProcessorTesterMixin`);
    mixed resolutions ⇒ per-sample expansion counts match `_get_num_multimodal_tokens`; mismatch cases raise the
    explicit errors above (no `StopIteration`).
  - *model integration (slow):* ragged batch `[2, 1, 7]` images through `forward` ⇒ per-sample logits equal each
    sample run unbatched (the `test_small_model_integration_test_batch_matches_single` pattern,
    `tests/models/llava_onevision/test_modeling_llava_onevision.py:552`; nested-uneven precedent `:450`,
    `tests/models/internvl/test_modeling_internvl.py:397`); wrong feature/token count raises
    (`test_mismatching_num_image_tokens` pattern, `tests/models/qwen2_vl/test_modeling_qwen2_vl.py:202`);
    right-padded batch ⇒ valid-position logits unchanged vs. unpadded.

### Config-driven sizes
**Decision: declare new config fields, hardcode no sizes — every value rides in the checkpoint's `config.json`.**
Apertus 1.5's continued pretraining enlarged `vocab_size` (text + image + audio codebooks) and changed
`rope_parameters` relative to base Apertus. `from_pretrained` builds the config via
`cls(**config.json)` (`configuration_utils.py:837-889`), so every field present in the checkpoint's `config.json`
(here the `text_config` block) overrides the class default; class-level defaults only fill fields **absent** from
the json, and unknown keys are tolerated (`@strict(accept_kwargs=True)`). The final 1.5 checkpoint already ships the
**enlarged weights** + matching `config.json`, `tokenizer_config.json` (new special tokens), and
`generation_config.json` — so `embed_tokens`/`lm_head` are instantiated at the enlarged `vocab_size` and the weights
load directly. **No `resize_token_embeddings`.**

### Registration
New `model_type = "apertus1p5"` (own dir + doc under *Multimodal models* + `tests/models/apertus1p5/`); register
the vision/audio tokenizer configs + models so `AutoModel.from_config` resolves them; add the generation class to
`MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` (`modeling_auto.py:1045`, hand-maintained → `image-text-to-text`
pipeline). `MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES` (`:1130`, backs `pipeline("any-to-any")` via
`AutoModelForMultimodalLM`) **spreads the image-text-to-text map**, so one entry covers both pipelines. Then
`make fix-repo`.

Duplicate-work check (2026-07-13): no open huggingface/transformers issues or PRs for "WavTokenizer" or
"Emu3.5" — re-check and open/claim the "new model" issues before coding, per `CLAUDE.md`.

### Porting EMU3.5 and WavTokenizer
Two owned, pre-trained tokenizers are reused; **Apertus1p5 only ever calls `encode()`** (X→Text). The EMU3.5 port
is encode-only (decoder dropped); for WavTokenizer, keeping its small decoder is recommended (see its bullet).
Both are **absent from transformers** and must be ported; **preserve their attribution** in the class docstrings
and the model doc page.

- **EMU3.5 Vision Tokenizer** — by BAAI, from *Emu3.5: Native Multimodal Models are World Learners*
  ([arXiv:2510.26583](https://arxiv.org/abs/2510.26583); weights
  [`BAAI/Emu3.5-VisionTokenizer`](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer)). A VQ image tokenizer using
  **IBQ** (Index Backpropagation Quantization — *Scalable Image Tokenization with IBQ*,
  [arXiv:2412.02692](https://arxiv.org/abs/2412.02692)); codebook **131,072**, 16× downsample (both confirmed in
  the repo's `config.json`; "IBQ" is asserted in the paper — the shipped quantizer class is
  `Emu3p5VisionVQVectorQuantizer`, encode-time = projection + argmax over codebook logits). **0.455B params,
  shipped in F32 (1.82 GB `model.safetensors` + duplicate `model.ckpt`; ~0.9 GB in bf16).** Shipped as remote
  custom code (`trust_remote_code`, `model_type = "Emu3p5VisionVQ"`, arch `Emu3p5VisionVQModel`). The built-in
  `Emu3VQVAE` (Emu3 3.0, plain VQ — *Emu3: Next-Token Prediction is All You Need*,
  [arXiv:2409.18869](https://arxiv.org/abs/2409.18869)) is **not reusable** (no IBQ) → port a native encode-only
  tokenizer (IBQ encoder + quantizer).
  > Class naming (Apertus-branded vs. keeping the Emu3.5 name) is an open decision — see
  > [Design decisions](#design-decisions) · **D1**.
- **WavTokenizer** — from *WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language
  Modeling* ([arXiv:2408.16532](https://arxiv.org/abs/2408.16532), ICLR 2025); **single-codebook** (40/75 tok/s);
  MIT-licensed (GitHub + HF). Base-class choice: `PreTrainedAudioTokenizerBase` (`modeling_utils.py:5236`) has
  **abstract `encode` *and* `decode`** — subclassed by EnCodec/Dac/Xcodec, **not** by Mimi (plain
  `PreTrainedModel`). So either **port the decoder too** *(recommended — it is small, makes `WavTokenizerModel` a
  complete standalone codec, and allows registering it in `MODEL_FOR_AUDIO_TOKENIZATION_NAMES`,
  `modeling_auto.py:1968`, alongside dac/xcodec2)*, or subclass plain `PreTrainedModel` encode-only. Note
  Mimi/Dac are **multi-codebook RVQ** — keep their `(batch, num_codebooks=1, frames)` codes shape for API
  consistency. Register in `CONFIG_MAPPING` + `MODEL_MAPPING`. Keeping the **WavTokenizer** name already
  preserves attribution.
- **Weights already exist on the Hub** (`BAAI/Emu3.5-VisionTokenizer`; WavTokenizer on its own repo) but in the
  *remote-code* layout keyed to the original module. (The wrappers `vllm_swissai` uses — `vision_tokenizer`,
  `apertus-audio-tokenizer` — are git-only packages, not on PyPI.) Porting needs only a **state-dict key remap**
  onto the native class — a small `convert_*_to_hf.py` (cf. `convert_emu3_weights_to_hf.py`), trivial/near-identity
  if the native module mirrors the original, and skippable if the key names already match. Where the converted
  weights are **published** for Apertus 1.5 (bundled vs. referenced) is [D2](#design-decisions).
- **Vocab budget:** see [Token stream & vocab layout](#token-stream--vocab-layout) — text 131,072 + visual
  131,072 + ~200 specials + audio codebook ⇒ extended `vocab_size` ≈ **2×** the text vocab (~266.5k). The vLLM
  audio offset `262344 = 2·131072 + 200` pins the ordering. Confirm against the real checkpoint.

### Implementation steps
1. **Scaffold (CLI):** `transformers add-new-model-like` → clone **`emu3`** as `apertus1p5` (model dir, `Apertus1p5*`
   classes, doc + toctree, tests, auto-mapping stubs); point the text sub-config at `apertus`.
2. **Skeleton end-to-end (stub encoders):** wire `Apertus1p5Config` / `Apertus1p5Model` (text + vision/audio
   tokenizers + vocabulary_mapping) / `Apertus1p5ForConditionalGeneration` + a placeholder-count processor;
   `make fix-repo`; confirm a random-weights `AutoModelForImageTextToText.from_config(...)` forward.
3. **Real tokenizers + processor:** port `Apertus1p5VisionVQ` (IBQ) and `WavTokenizerModel` (+ `convert_*_to_hf.py`);
   **test each against the original** (same input ⇒ identical codes); verify from the checkpoint tokenizer that the
   `<|visual token N|>` ids are contiguous/ordered and read off the audio offset (→ config fields); **freeze
   real-encoder goldens once** (vLLM's tests mostly stub the encoders — see Testing); implement
   `Apertus1p5Processor` (matching the `vllm_swissai` token layout byte-for-byte) + processing tests vs. the
   goldens.
4. **Final pass:** end-to-end `pipeline("image-text-to-text")` / `any-to-any`; parity vs `vllm_swissai`;
   `make fix-repo` + `make check-repo` + `RUN_SLOW=1` tests; doc attribution + license check ([D2](#design-decisions)).

### Testing
Test bottom-up; the **original tokenizers** and **`vllm_swissai`** are the ground truth.
1. **Tokenizer parity (unit):** native `Apertus1p5VisionVQ` / `WavTokenizerModel` `.encode()` vs. the original
   (remote-code EMU3.5 / WavTokenizer pkg) on the same image/audio ⇒ assert **identical integer codes** (bit-exact;
   `allclose` on pre-quant latents only when debugging). This is the correctness gate for the port.
2. **Processor golden (unit):** `Apertus1p5Processor(images=, audio=, text=)` vs. a **frozen expected stream**.
   Caveat: transformers `input_ids` hold `image_token_id`/`audio_token_id` **placeholders** at the code positions
   (real codes only exist after the model-side encoders), so the direct comparison covers the *structural* tokens
   (boi/`H*W` digits/img/eol/eoi, audio_start/end) and the *placeholder counts* — vLLM's real-code ids at those
   positions can't be compared here. Also note `vllm_swissai`'s `test_apertus.py` mostly stubs the encoders
   (`install_fake_encoders`); its only true goldens are the 2×2 image-layout string and the audio serialization
   round-trip — reuse those, and **freeze one real-encoder golden** from `ApertusMultiModalProcessor` for the rest.
   Add a count-determinism test: processor-predicted audio placeholder count == WavTokenizer frame count across
   durations. (`_check_special_mm_tokens`, `processing_utils.py:2310`, guards placeholder/media count mismatches.)
3. **Stream parity (slow):** splice the model-side code→vocab ids (`get_image_features`/`get_audio_features`
   mapping output) into the placeholder positions of the processor's `input_ids` ⇒ the full sequence must equal the
   vLLM golden **byte-for-byte**. This closes the gap test 2 can't cover.
4. **Model parity (slow):** `AutoModelForImageTextToText.from_pretrained(...)` forward on fixed inputs ⇒ greedy-decode
   token IDs match the reference (`vllm_swissai` / original) exactly; logits within tolerance.
5. **Pipeline (e2e):** `pipeline("image-text-to-text")` / `any-to-any` on a canonical image+audio prompt returns the
   expected text; sanity-diff against the `vllm_swissai` example outputs.
6. **Batched / variable image counts:** see the test list in
   [Batched inference: variable images per prompt](#batched-inference-variable-images-per-prompt) —
   0/1/many nested batches, ragged `[2, 1, 7]` forward vs. unbatched, mismatch errors, padding invariance.

---

## Design decisions

To resolve before implementation; recommendation first, then options.

### D1 — Naming of the ported vision tokenizer (attribution vs. namespace)
**Leaning: B — keep the Emu3.5 name.** The requirement to preserve tokenizer ownership/attribution argues for it
(A is acceptable with prominent docstring + doc credit); audio is already settled by keeping the **WavTokenizer**
name. Options for the native class (§ [Porting](#porting-emu35-and-wavtokenizer)):
- **A. Apertus-branded** — e.g. `Apertus1p5VisionVQ` under `apertus1p5/`, EMU3.5 + IBQ credited in the class
  docstring and doc page. Self-contained; provenance is documented, not in the name.
- **B. Keep the Emu3.5 name** — a standalone `emu3p5` model (`Emu3p5VisionVQ` / `Emu3p5VisionVQModel`, matching the
  Hub `model_type`), plugged into `apertus1p5` via `AutoModel.from_config`. Ownership is explicit in the class name
  and the tokenizer stays independently reusable, at the cost of a second `model_type` / model dir.

### D2 — Where the tokenizer weights live (bundle vs. reference upstream)
**Recommendation: bundle into the Apertus 1.5 checkpoint** for a single flagship model; use the owned mirror only
if several variants would otherwise duplicate the identical tokenizer. Both tokenizers are **sub-modules of
`Apertus1p5Model`**, so their weights are part of the state dict *by construction* (as Emu3 bundles `Emu3VQVAE`) —
the question is only what we publish:
- **Reference upstream** — pull EMU3.5 from `BAAI/Emu3.5-VisionTokenizer` / WavTokenizer from its repo at load time.
  Couples us to third-party repos (rename / delete / re-gate / revision drift), needs remote code or an external
  package, and adds a hot-path download. ❌ fragile.
- **Bundle into the Apertus 1.5 checkpoint** *(recommended)* — convert once (offline) and save the native VQ + codec
  inside the Apertus1p5 safetensors. One self-contained `from_pretrained`, **no `trust_remote_code`** at inference,
  guaranteed weight/version match with training, fully independent of upstream Hub churn.
- **Owned mirror (middle ground)** — publish a frozen `swiss-ai/apertus-1.5-tokenizers` repo *we* control, referenced
  by every Apertus1p5 variant. Still upstream-independent, and **de-duplicates** the VQ across variants, at the cost
  of one cross-repo reference (to our own repo).

Costs / caveats:
- **Size:** the IBQ VQ is 0.455B params (ships in F32 at 1.82 GB; ~0.9 GB bf16) copied into each checkpoint;
  WavTokenizer is small (tens of MB). Negligible beside an 8B+ backbone, but multiplied across many variants →
  consider the owned-mirror option.
- **Licensing:** bundling **redistributes** the weights. Verified: WavTokenizer is **MIT** (GitHub + HF);
  `BAAI/Emu3.5-VisionTokenizer` is **tagged `apache-2.0` on the model card but ships no LICENSE file** — include
  the Apache-2.0 text + attribution when redistributing, and consider confirming with BAAI (also the ownership
  point from D1).
- **vLLM parity:** keep the bundled revision **bit-identical** to what `vllm_swissai` loads (or point vLLM at the same
  repo), so both stacks produce identical tokens.

### D3 — Cross-stack checkpoint compatibility (new `model_type` vs. today's vLLM loader)
**Recommendation: A — vLLM adapts**, so one flagship checkpoint serves both stacks. Background: today
`vllm_swissai` loads the checkpoint as a **plain, flat `ApertusConfig`** (`model_type = "apertus"`,
`apertus.py:470-471, 966`) — it reads **no** multimodal fields from `config.json`; all MM behavior comes from
hardcoded constants + `mm_processor_kwargs`, and the external `vision_tokenizer` / `apertus-audio-tokenizer`
packages fetch their own weights. The transformers design introduces `model_type = "apertus1p5"` with a **nested**
`text_config` and bundled tokenizer weights — a checkpoint published in that layout will **not load in current
vllm_swissai** without a small adaptation. Options:
- **A. vLLM adapts** *(recommended)* — vLLM gains a config shim (accept the new `model_type`, read the nested
  `text_config`) and reads constants (offsets, ds factor, pixel bounds, sample rate) from `config.json`,
  removing today's hardcoding.
- **B. Dual publishing** — keep a flat-config variant for vLLM and a nested one for transformers. ❌ two artifacts
  to keep in sync; violates the "same checkpoint serves identically under both stacks" constraint.

---

## Reference: vLLM integration (vllm_swissai)

A full vLLM implementation of this exact model lives in the sibling repo **`vllm_swissai`** (branch
`apertus_integration`). Main files: `vllm/model_executor/models/apertus.py`, with
`examples/offline_inference/apertus_multimodal.py` and `tests/models/multimodal/processing/test_apertus.py`. 
It is the **reference for the vLLM side** and a **template for the tokenization / processor logic** (not the modeling).

In it, the base model is the plain `ApertusForCausalLM`; all multimodality is a `ApertusMultiModalProcessor`
(`SupportsMultiModal` + `MULTIMODAL_REGISTRY.register_processor`) that loads the **EMU3.5** tokenizer from an external
`vision_tokenizer` package (`vq_type="ibq"`, Hub weights via `resolve_emu35_weights`) and **WavTokenizer** from
`apertus-audio-tokenizer` (audio → 24 kHz, 40 tok/s), configured via `mm_processor_kwargs`.

**What `Apertus1p5` (transformers) can reuse directly:**
- **Token layout, verbatim** — `build_apertus_image_prompt` (`:248-271`): `{BOI}{H}*{W}{IMG} row0 {EOL} row1 …
  {EOI}` with the exact strings `<|img_start|>`, `<|img_token_start|>`, `<|img_end_of_row|>`, `<|img_end|>`;
  `H`/`W` are **token-grid** dims (pixels ÷ 16) as text digits; each IBQ index rendered by
  `VISUAL_TEMPLATE = "<|visual token {id}|>"`; rows **join**ed by EOL (⇒ `H−1` EOLs). Audio via
  `serialize_audio_token_ids` (`convert_ids_to_tokens` of `code + 262344`) wrapped in `<|audio_start|>` /
  `<|audio_end|>`. Copy the exact special tokens + ordering into `Apertus1p5Processor` and the vocabulary mapping
  (details in [Token stream & vocab layout](#token-stream--vocab-layout)).
- **Preprocessing constants** — image: area-clamp in `encode_images` (`target_area = max(min(1400², w·h), 256²)`,
  `:291-294`), then `smart_resize` rounds each side to a multiple of `EMU35_DS_FACTOR = 16` (round-half-up,
  bicubic, `:184-191`), normalization `/127.5 − 1` in F32; audio: mono, resample to 24 kHz, **peak-normalize to
  −3 dBFS** (`DEFAULT_TARGET_PEAK_DBFS`, `:345, 442-445`). Reuse these so the encoders see identical inputs.
- **Token ceilings** — `ApertusProcessingInfo.get_mm_max_tokens_per_item` (image `max_px // ds² + 512` = 8168,
  audio `40/s · 300 s + 4`) is a **profiling ceiling**, *not* the placeholder expansion — the exact per-media
  count is the deterministic layout math above.
- **The tokenizer wrappers** — `ApertusImageTokenizer` / `ApertusAudioTokenizer` (encode calls, grid extraction
  `encode_out[2][2].view(H, W)`, caching, `mm_processor_kwargs` knobs) map almost 1:1 onto the encode path of the
  native `Apertus1p5VisionVQ` / `WavTokenizerModel`; port their logic, swapping the external-package call for the
  native `encode()`.
- **Golden tests** — `tests/models/multimodal/processing/test_apertus.py` is mostly **fake-encoder** plumbing
  tests (`install_fake_encoders`); the two true goldens are the 2×2 image-layout string (`:185-197`) and the audio
  serialization round-trip (`:454-470`). Real-code goldens (and the `+262344` offset) are **not** asserted there —
  freeze them once with the real packages.

Behavior worth knowing: image + audio only (video is rejected); unlimited multi-image/multi-audio with ordered
left-to-right placeholder binding; more media than placeholders → error, more placeholders than media → replaced
with `""` + log. The vLLM side reads a **plain flat `ApertusConfig`** and hardcodes all MM constants (see
[D3](#design-decisions)); the tokenizer (e.g. `tokenizer/apertus_emu3.5_instruct`) must already contain every
special/visual token. (Minor drift: the example's `apertus_vq_hub` / `apertus_emu35_codebase` kwargs are ignored
by `apertus.py` — `vq_hub` is hardcoded to `BAAI/Emu3.5-VisionTokenizer`.)

**The one hard constraint:** the transformers `Apertus1p5Processor` must emit the *same* special tokens, grid layout,
resize/DS factor, audio rate/normalization, and code→vocab-id mapping as this processor — otherwise a checkpoint
won't serve identically under both stacks.

---

## Reference: the existing Apertus model

Apertus is currently integrated as a **text-only causal LM** (Llama-derived, with q/k-norm,
`xielu` MLP activation, and `attention_layernorm`/`feedforward_layernorm` decoder norms). Its
full footprint in the repo:

### Model files — `src/transformers/models/apertus/`
| File                       | Role                                                                                                                                       |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `modular_apertus.py`       | **Central modular source** — the only file authored by hand. Subclasses Llama/Nemotron classes; `__all__` at `modular_apertus.py:262`.     |
| `configuration_apertus.py` | Generated config. `model_type = "apertus"` (`:45`), `__all__ = ["ApertusConfig"]` (`:99`).                                                 |
| `modeling_apertus.py`      | Generated modeling. `__all__` at `:505` → `ApertusModel`, `ApertusForCausalLM`, `ApertusForTokenClassification`, `ApertusPreTrainedModel`. |
| `__init__.py`              | Lazy-loader boilerplate (`_LazyModule` + `define_import_structure`), driven by each submodule's `__all__`.                                 |

> Note: `configuration_apertus.py` and `modeling_apertus.py` are **auto-generated** from
> `modular_apertus.py`. Do not edit them directly — edits are overwritten.

### Registration / wiring footprint
| Location                                             | Entry                                                                                            |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `src/transformers/models/__init__.py:26`             | `from .apertus import *`                                                                         |
| `src/transformers/models/auto/auto_mappings.py:37`   | `("apertus", "ApertusConfig")` in `CONFIG_MAPPING_NAMES` (**auto-generated file**)               |
| `src/transformers/models/auto/modeling_auto.py:50`   | `("apertus", "ApertusModel")` in `MODEL_MAPPING_NAMES`                                           |
| `src/transformers/models/auto/modeling_auto.py:669`  | `("apertus", "ApertusForCausalLM")` in `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`                       |
| `src/transformers/models/auto/modeling_auto.py:1570` | `("apertus", "ApertusForTokenClassification")` in `MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES` |
| `docs/source/en/model_doc/apertus.md`                | Model doc page                                                                                   |
| `docs/source/en/_toctree.yml:510-511`                | Toctree entry (under the **Text models** section)                                                |
| `tests/models/apertus/`                              | `__init__.py` + `test_modeling_apertus.py`                                                       |

Apertus is intentionally **absent** from `tokenization_auto.py` (its tokenizer is resolved from
the checkpoint's `tokenizer_config.json`) and has **no processor / image processor / vision**
components today, so it is not registered in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`.

## Reference: standard workflow for adding a model to Transformers
- Coordinate on the “New model” issue and check for overlapping PRs before coding.
- Bootstrap boilerplate with `transformers add-new-model-like`.
- Implement `modular_<name>.py` as the source of truth; inherit existing components and override only differences.
- Generate files with `python utils/modular_model_converter.py <name>`; never edit generated files directly.
- Add configs, processors, AutoClass/pipeline mappings, checkpoint conversion, docs, and tests.
- Validate against the original checkpoint; run model tests, `make check-repo`, and finally `make fix-repo`.
- For Apertus 1.5, author `modular_apertus1p5.py` and include native image/audio processing integrations.
- References: `CONTRIBUTING.md`, `docs/source/en/{add_new_model,modular_transformers,testing,add_vision_processing_components,weightconverter}.md`, and `AGENTS.md`.
