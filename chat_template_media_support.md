# Apertus 1.5 chat template: user-message media formats and URL fetching

## What the current (patched) template accepts for user content

| Content shape                                 | Rendered                                | URLs fetched / media prepared automatically?  |
|-----------------------------------------------|-----------------------------------------|-----------------------------------------------|
| Flat string                                   | verbatim (placeholders typed literally) | NO, manual media prep                         |
| `parts` mapping (upstream form)               | one placeholder per media part          | NO, `url` keys are ignored, manual media prep |
| List of blocks (added by our converter patch) | identical to `parts` (test-verified)    | YES, one-call fetch + prep + tokenize         |

The template itself never fetches anything in any shape; fetching lives in the
transformers media collector (`processing_utils.py`), which only understands
list-of-blocks content. The unpatched upstream template accepted only the first
two shapes; list content raised `Invalid user message`.

### 1. Flat string: manual media prep

```python
messages = [{"role": "user", "content": "<|image|>Describe this image."}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # a STRING

image = PIL.Image.open("photo.jpg")                       # you load media yourself
inputs = processor(text=prompt, images=[image], return_tensors="pt")
out = model.generate(**inputs)
```

### 2. `parts` mapping: template places the placeholder, media prep still manual

```python
messages = [{"role": "user", "content": {"parts": [
    {"type": "image"},                                    # a "url" key here would be IGNORED
    {"type": "text", "text": "Describe this image."},
]}}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # same string as (1)
inputs = processor(text=prompt, images=[PIL.Image.open("photo.jpg")], return_tensors="pt")
```

Calling `apply_chat_template(tokenize=True, return_dict=True)` on (1) or (2)
does NOT work for multimodal input: no media is collected, so the placeholders
have no tensors and validation raises.

### 3. List of blocks (our patch): one call does everything

```python
messages = [{"role": "user", "content": [
    {"type": "image", "url": "https://example.com/photo.jpg"},   # downloaded
    {"type": "image", "path": "/data/scan.png"},                 # opened from disk
    {"type": "text",  "text": "Compare the images with the clip:"},
    {"type": "audio", "url": "https://example.com/clip.wav"},    # downloaded + resampled to 24 kHz
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
out = model.generate(**inputs, max_new_tokens=64)
```

Renders the same placeholder stream as (2), but the media collector reads the
blocks, loads everything, and the processor returns model-ready tensors
(`input_ids`, `pixel_values`, `image_sizes`, `input_features`, ...). This is
also the OpenAI-style message schema, so OpenAI-compatible clients, LangChain,
vLLM/SGLang serving, and the transformers `image-text-to-text` pipeline emit it
natively. On the UNPATCHED template this exact call crashes with
`Invalid user message: user`.

## The possible solutions

1. **Template patch: add the blocks branch (IMPLEMENTED).**
   One Jinja `elif` accepting list content, rendering exactly like `parts`
   (aliases `image_url`/`input_image`/`audio_url`/`input_audio` included).
   Pros: data-level fix that ships inside the checkpoint, works for every
   consumer, backward compatible (flat and `parts` unchanged), no transformers
   code change. To be upstreamed to apertus-omni-tokenizer.

2. **Core code change: teach the shared transformers media collector the
   `parts` shape.** Rejected: the collection loop is shared core infrastructure
   (inlined in `ProcessorMixin.apply_chat_template`,
   `processing_utils.py:2101-2142`, not an overridable hook), a per-model
   message schema there is effectively unupstreamable, it would fix only
   transformers, and it couples the checkpoint to transformers versions.

3. **Model-scoped subclass shim on `Apertus1p5Processor` (viable add-on, not
   implemented).** Override `apply_chat_template` with a ~10-line normalization
   that converts `{"parts": [...]}` into the equivalent block list BEFORE
   delegating to `super()`. The base collector then sees blocks and does the
   fetching; the template renders via its blocks branch (render-identical).
   Precedent: the base method itself already normalizes OpenAI `image_url`
   blocks the same way (`processing_utils.py:2084-2099`). Model-owned code,
   upstreamable. Only affects calls through the transformers processor.

Decision: solution 1, with solution 3 available later if `parts` messages
should also get auto-loading.

## Q&A

**Is a fix possible WITHOUT the template patch?** Technically yes, but strictly
worse: a subclass would have to collect media from blocks itself (duplicating
the core collection loop), convert blocks into `parts` so the unpatched
template can render them, and re-plumb tokenization. That duplicates core logic
that changes between versions, and it fixes only transformers: vLLM/SGLang
serving and OpenAI-style clients render the template directly, so blocks
messages would still crash there. The template is the only layer every
consumer shares.

**Can the `parts` path also get auto-loading?** Yes, via solution 3: after the
parts-to-blocks normalization, `parts` messages go through the same
fetch + prep + tokenize path. Rendering is unchanged (the two branches are
token-identical). Direct template rendering of `parts` outside transformers
keeps its current behavior (renders fine, manual media prep).

**Side effects of the blocks branch (tool calling, thinking, etc.)?** None on
those paths: the branch lives inside the USER-role handling only. Tool calls,
tool outputs, and thinking/inner sections (`<|inner_prefix|>`/`<|inner_suffix|>`)
are rendered in the assistant/tool branches, which are untouched; the assistant
"blocks" form is a MAPPING (`{"blocks": [...]}`), so it cannot collide with the
user-role plain-list dispatch (`is mapping` vs `is iterable and not mapping`).
Known benign edges: an empty list now renders an empty user turn instead of
raising; unknown part types raise `Invalid user part` instead of
`Invalid user message` (loud either way). Remaining scope limit (pre-existing,
not caused by the patch): blocks lists are accepted for USER messages only.
System content must stay a string or `{"text": ...}`, and assistant content a
string or `{"blocks": ...}` mapping, so replaying an OpenAI-style conversation
that uses list content for system or assistant turns still raises.