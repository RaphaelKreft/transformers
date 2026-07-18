# Apertus 1.5 → transformers: interface & design (as built)

**What:** `apertus1p5`: image + audio + text → text, discrete-token early fusion. Frozen encode-only
tokenizers (EMU3.5-IBQ vision, WavTokenizer audio) turn media into discrete codes that plain integer offsets
map into an enlarged shared vocabulary (266,752 = ~131k text + 131,072 visual @ 131272 + 4,096 audio @ 262344
+ specials), so the Apertus backbone sees one token stream. Byte-level ground truth:
`vllm_swissai@apertus_integration`. Status: M0-M6 done, reviewed, all gates green (see `milestone_*.md`);
runnable examples in `ap_testcase/`.

## User interface

```python
processor = AutoProcessor.from_pretrained(ckpt)     # Apertus1p5Processor (+ Apertus1p5ImageProcessor,
model = Apertus1p5ForConditionalGeneration.from_pretrained(ckpt, dtype=torch.bfloat16)  # + WavTokenizerFeatureExtractor)

# (a) instruct quick start: standard HF chat; media auto-loaded from url/path, audio auto-resampled to 24 kHz
messages = [{"role": "user", "content": [{"type": "image", "url": ...}, {"type": "text", "text": "..."}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")

# (b) base model / full control: rendered text with one <|image|>/<|audio|> placeholder per media item
inputs = processor(text="<|image|> what is said here: <|audio|>", images=[img], audio=[wave_24k],
                   return_tensors="pt")

# (c) batching: arbitrary media counts per sample; nested lists (one sub-list per sample, [] allowed)
#     or flat lists consumed left-to-right by placeholder order; padding=True (tokenizer is left-padding)
inputs = processor(text=[t1, t2, t3], images=[[a, b], [], [c]], audio=[[], [clip], []], padding=True, ...)

out = model.generate(**inputs, max_new_tokens=64)   # text-only output; beam search / num_return_sequences OK
```

- Processor output = model input, exactly: `input_ids`, `attention_mask`, `pixel_values (total_images,3,Hmax,Wmax)`
  fp32 in [-1,1] + `image_sizes (total_images,2)` (true resized sizes), `input_features (total_clips,1,Tmax)` +
  `feature_attention_mask` (right-padded). Media tensors are flattened over items, not batch rows.
- Strict validation both ways: any placeholder/media count mismatch raises a ValueError (per sample for nested
  input, totals for flat); all-empty collections (`[[],[]]`) mean "no media". Videos are ignored (unsupported
  modality).
- **Assumed inputs vs performed validation:** images must be UNSCALED (PIL/uint8-range; a float image already
  in [0,1] would be rescaled again, standard TorchvisionBackend `do_rescale` convention with no heuristic
  warning); RGB conversion, x16 resize and [-1,1] normalization are applied by the image processor, so the
  model contract holds by construction. Bare waveforms are assumed 24 kHz mono but scale-independent (every
  clip is peak-normalized to -3 dBFS). Rejected loudly: stereo or empty clips, a declared `sampling_rate`
  other than 24000, count mismatches, and (model-side) non-right-padded masks, zero-length clips and
  batch-dimension inconsistencies.
- **Media fetching (gemma-style, on both entry points):** image/audio entries may be URL or local-path strings
  instead of loaded objects. Direct calls: the inherited `ProcessorMixin.prepare_inputs_layout`
  (`processing_utils.py:704`) runs `image_processor.fetch_images` (`image_processing_base.py:473`, recursive
  over flat/nested lists) and `feature_extractor.fetch_audio(..., sampling_rate=24000)`
  (`feature_extraction_sequence_utils.py:371`, which calls `load_audio`, `audio_utils.py:194`: httpx download,
  librosa/torchcodec decode, resample to the FE's 24 kHz); our override
  (`processing_apertus1p5.py::prepare_inputs_layout`) applies `fetch_audio` per sub-list so nested audio keeps
  URL support. Chat path: `apply_chat_template(tokenize=True)` (`processing_utils.py:1963`, media collection
  around `:2101-2148`) gathers `url`/`path`/`base64` content blocks, eagerly loads audio at the FE's sampling
  rate, then calls the same `__call__`. Bare waveform ARRAYS are never resampled; they are assumed to be
  24 kHz mono (matches the vLLM reference). Audio file loading needs `librosa`.
- Standalone components also public: `Apertus1p5VisionTokenizerModel` (image → code grid, encode-only) and the reusable
  `WavTokenizerModel`/`WavTokenizerFeatureExtractor` (separate standalone model, encode + decode).

## Design

**Processor** (`processing_apertus1p5.py`, generic `ProcessorMixin` hooks, no custom `__call__`; never runs
the encoders). Each `<|image|>` expands to the byte-exact vLLM run
`<|img_start|>{H}*{W}<|img_token_start|>` + H rows of W `<|image|>` joined by exactly H−1 `<|img_end_of_row|>`
+ `<|img_end|>` (H/W = resized pixels ÷ 16, height first, plain digits; no eof, no Emu3 `H·(W+1)`); each
`<|audio|>` becomes `<|audio_start|>` + ceil(samples/600) placeholders + `<|audio_end|>` (no header). Audio is
peak-normalized to −3 dBFS before the feature extractor (reference behavior); placeholder counts derive from
the FE *output*, so truncation can never desync counts from features. BOS comes from the tokenizer's own
post-processor; the processor adds nothing.

**Image processor** (`image_processing_apertus1p5.py`, `TorchvisionBackend`): vLLM's exact `smart_resize`
(area clamp [256², 1400²] → aspect-preserving → round-half-up to ×16), `/127.5 − 1` fp32, pad-to-batch-max
(padding never reaches the encoder; the model crops by `image_sizes`). Torchvision BICUBIC ≈ PIL BICUBIC:
sizes/counts identical, pixel values differ slightly (~0–0.8% code flips; PIL preprocessing gives byte-exact
reference codes, measured as INFO by `scripts/check_apertus1p5_processor_parity.py`).

**Text backbone** (`apertus1p5_text`, supersedes plain `apertus` for 1.5 checkpoints):
`Apertus1p5TextConfig(ApertusConfig)` overrides only `vocab_size=266752` and adds `output_vocab_size`
(all other hyperparameters come from the checkpoint config). Checkpoints ship a **pruned output layer**:
the multimodal rows are removed from `lm_head.weight` and `output_vocab_size` (131072 for the released
checkpoints) records the retained prefix; input embeddings keep the full vocabulary, retained output ids
equal their token ids, so generation needs no remapping and the model can never emit multimodal ids.
Pruned heads cannot tie and cannot be resized (validated/guarded); the loss uses the head size; beam search
resolves its vocab via `get_output_embeddings().out_features` (class special case in `generation/utils.py`).
`Apertus1p5TextForCausalLM` also loads DIRECTLY from the joint composite checkpoint: the config is extracted
via `base_config_key = "text_config"` and the `model.language_model.*` keys are remapped by a `PrefixChange`
entry in `conversion_mapping.py` (qwen3_5_text precedent); the tokenizer weights surface as ignorable
unexpected keys. The local pruning was cross-checked **bit-exact** against the reference hub pruning
(`apertus-ai/Apertus-v1.5-8B-integration`, rev `refs/pr/1`).

**Model** (`modular_apertus1p5.py` → generated files; fully Emu3-independent).
- `Apertus1p5Config`: `sub_configs = {text_config → apertus1p5_text (AutoConfig), vision_tokenizer_config, audio_tokenizer_config →
  wavtokenizer (AutoConfig)}` + token ids/offsets (defaults = the verified real-tokenizer values); validates
  the offset layout, vocabulary coverage, and tie-vs-pruned-head consistency at construction.
- `Apertus1p5Model`: `AutoModel` Apertus backbone + bundled `Apertus1p5VisionTokenizerModel` + AutoModel-resolved
  WavTokenizer. `get_image_tokens`/`get_audio_tokens`: **each image/clip encoded individually** (global
  attention makes padded-batch encoding perturb codes; matches vLLM) → `codes.flatten() + offset` → shared
  `embed_tokens` → one `masked_scatter` per modality into the placeholder positions (row-major flat order ==
  placeholder order; only a global count guard, like all HF VLMs).
- `Apertus1p5ForConditionalGeneration`: `lm_head` (tied to embeddings when configured), `output_modalities =
  ("text",)`, **no decode methods anywhere** (the vision decoder is not even ported); media tensors are
  dropped after the prefill step. `_expand_inputs_for_generation` override (qwen2-vl-family pattern) repeats
  flattened media by per-sample *groups* so beam search / `num_return_sequences` keep media↔prompt ownership
  (verified token-for-token vs per-sample generation).
- **fp32 is enforced, not advised**: `_keep_in_fp32_modules_strict = ["vision_tokenizer", "audio_tokenizer"]` keeps both
  tokenizers fp32 even on bf16/fp16 loads (code assignment is an argmax; bf16 flips ~8% of visual codes);
  `encode` upcasts half-precision inputs.

**Checkpoint** (one self-contained repo; `convert_apertus1p5_weights_to_hf.py` assembles it from backbone +
converted vision tokenizer + converted WavTokenizer): weights under `model.language_model.* / model.vision_tokenizer.* /
model.audio_tokenizer.* / lm_head.weight` (source-grouped shards + index; same pattern as Emu3's merged
checkpoint), nested config, tokenizer (media special tokens injected, `padding_side="left"` pinned), unified
`processor_config.json`, and the chat template patched to also accept standard list-of-content-blocks
messages (string and `{"parts": [...]}` forms unchanged; fix to be upstreamed to apertus-omni-tokenizer).

## Verification (all executed; commands in `milestone_5.md` and `milestone_6.md`)
Bit-exact encoder parity vs both original repos (real weights) · byte-equal spliced token stream vs the vLLM
reference builder · golden id sequences against the real vocabulary · pruning cross-checked bit-exact against
the reference hub pruning · real-8B integration tests 9/9 (chat three-content-forms equivalence, multimodal
generate, fp32 guard, text-from-composite) · 320+ fast tests / 0 failures · six adversarial review rounds
with all findings fixed. Open (M7): publish checkpoints, upstream the template patch, PR packaging.
