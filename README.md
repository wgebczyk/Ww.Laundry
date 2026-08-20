# WatermarkTool

A .NET console app that implements the token-level green-list/red-list text watermark from
Kirchenbauer et al., *"A Watermark for Large Language Models"* (2023), on top of
[LLamaSharp](https://github.com/SciSharp/LLamaSharp) `0.27.0` and a local GGUF model.

It can:

- **generate** text whose token choices are statistically biased by a secret key,
- **detect** that watermark in a piece of text using only the tokenizer and the key — no model
  inference required, and
- **rewatermark** — a robustness harness that measures how far the z-score drops when tokens are
  substituted to flip green/red membership under a different key.

## How it works

At every generation step the tool hashes the secret key together with the last `ngram` tokens into
a step seed. That seed deterministically splits the vocabulary into a **green list** (a `gamma`
fraction of it) and a **red list**. Green tokens get `delta` added to their logit before sampling,
so watermarked text picks green tokens noticeably more often than chance.

Detection replays the same hash over the tokenized text, counts how many tokens landed on the green
list, and runs a one-sided z-test:

```
z = (greenCount - gamma * T) / sqrt(T * gamma * (1 - gamma))
```

Human-written text has no reason to prefer green tokens, so it scores `z ≈ 0`. Watermarked text
scores high.

The hash is a splitmix64-style mixer, **not** `System.Random`, because .NET makes no guarantee that
`Random`'s algorithm is stable across releases — and detection must be bit-for-bit reproducible.

## Build

```
dotnet build WatermarkTool/WatermarkTool.csproj
```

Targets `net10.0`. Uses `LLamaSharp` + `LLamaSharp.Backend.Cpu`; swap the backend package for
`LLamaSharp.Backend.Cuda12` if you have a GPU, then pass `--gpu-layers`.

## Usage

### generate

```
dotnet run --project WatermarkTool -- generate \
  --model C:\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --prompt "Write a short essay about the sea." \
  --key 8675309 \
  [--gamma 0.5] [--delta 2.0] [--ngram 4] [--max-tokens 512] \
  [--temp 0.8] [--top-k 40] [--top-p 0.95] [--seed 1234] \
  [--ctx 4096] [--gpu-layers 0]
```

Generated text goes to **stdout**; progress and the watermark parameters go to **stderr**, so you
can redirect cleanly:

```
dotnet run --project WatermarkTool -- generate --model ... --prompt "..." --key 8675309 > out.txt
```

### detect

```
dotnet run --project WatermarkTool -- detect \
  --model C:\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --text-file out.txt \
  --key 8675309 \
  [--gamma 0.5] [--ngram 4] [--z-threshold 4.0]
```

The model is opened with `VocabOnly = true`, so this loads the tokenizer and nothing else — it is
fast and needs no GPU. You can also pass the text inline with `--text "..."`.

Output reports the scored token count, green count, z-score, p-value and a verdict.

### selftest

```
dotnet run --project WatermarkTool -- selftest
```

Validates the watermark statistics end to end against a simulated model, so you can check the
implementation without a GGUF file. It asserts that watermarked text is detected with its own key,
is *not* detected with a different key, and that unwatermarked text is not detected.

### rewatermark (robustness harness)

```
dotnet run --project WatermarkTool -- rewatermark \
  --model C:\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --text-file out.txt \
  --old-key 8675309 --new-key 1234567 \
  [--gamma 0.5] [--ngram 4] [--top-n 32] [--max-logit-drop 2.0] \
  [--ctx 4096] [--gpu-layers 0] [--out modified.txt]
```

This measures **how much of an existing watermark survives targeted token substitution**. It:

1. Scores the input against `--old-key` first, and says so if the input is not actually
   watermarked (in which case the before/after comparison measures nothing).
2. Teacher-forces the text through the model to recover the logits at each position.
3. At positions where the token is red under `--new-key`, swaps in the highest-probability
   alternative that is green under the new key, ranks inside `--top-n`, and is within
   `--max-logit-drop` of the best token at that position. Positions with no suitable candidate are
   left alone.
4. Reports the old-key z-score **before vs. after**, the same for the new key, and how many
   positions were substituted vs. left unchanged.

It reports the after-score twice: once on the modified token ids, and once on the emitted text
re-tokenized from scratch, which is what a real detector would actually compute.

#### Measured results

Verified end to end with `Qwen2.5-1.5B-Instruct-Q4_K_M`, 300 generated tokens, `gamma=0.5`,
`delta=2.0`, `ngram=4`:

| Check | z-score | Verdict |
|---|---|---|
| Watermarked text, generating key | **7.32** | detected |
| Watermarked text, different key | −1.28 | not detected |
| Unwatermarked text (`--delta 0`), same key | −1.97 | not detected |

Running `rewatermark` against that text, varying `--max-logit-drop`:

| `--max-logit-drop` | Substitutions | old-key z | Readability |
|---|---|---|---|
| (none) | 0 | 7.32 | intact |
| 0.25 | 39 / 296 | **0.93** | degraded but readable |
| 0.5 | 47 / 296 | 1.40 | degraded |
| 1.0 | 55 / 296 | 1.16 | poor |
| 2.0 | 87 / 296 | 1.63 | badly broken |

Two things worth drawing out of this:

- **The watermark is fragile.** Substituting ~13% of tokens drops z from 7.32 to below 1. The
  scheme's own context dependence works against it: changing one token also changes the step seed
  for the next `ngram` positions, so each edit corrupts a window rather than a single score. Larger
  `ngram` amplifies this. If you need robustness, that trade-off is the thing to tune.
- **z-degradation is not monotonic in the substitution budget.** More aggressive substitution
  destroys fluency long before it further reduces z, because once the green rate is at chance,
  additional swaps just add noise. The tightest gate gave both the best z reduction and the most
  readable output.


- Logits come from a **single forward pass over the original token sequence**. As soon as one token
  is substituted, the model's true conditional distribution for every later position changes, and
  this tool does not re-run the forward pass. The further into the text you go, the less the logits
  reflect what the model would really predict.
- The `--max-logit-drop` gate is a crude stand-in for "means the same thing". A high-probability
  alternative is not necessarily a synonym, so substitutions degrade fluency and can change meaning.
- Only a fraction of red positions are substitutable at all. Expect a partial reduction in z, not a
  clean erasure — full re-watermarking of arbitrary text is inherently incomplete.
- Even at the tightest gate the output has visible grammatical damage (dropped connectives, mangled
  word forms, repeated tokens). This is a robustness *measurement*, not a paraphraser.

#### A note on scope

This mode is included as an **evaluation tool for your own watermark**: you cannot claim a
watermarking scheme is robust without measuring what it takes to break it, and reporting
z-degradation is exactly that measurement. That framing is deliberate, and it is why the command's
primary output is a before/after z comparison rather than just rewritten text.

The same machinery, pointed at someone else's watermarked text, is a provenance-stripping tool.
Watermarks exist so that AI-generated content can be attributed, and defeating a third party's
watermark undermines that. Substituting a *new* key on top can also make text look as though it
came from a source it didn't, which is the spoofing side of the same coin. Please keep this pointed
at watermarks you own and want to stress-test.

## Parameters

| Parameter | Meaning | Typical |
|---|---|---|
| `key` | Secret that seeds the green/red split. **Must match between generate and detect.** | any `ulong` |
| `gamma` | Fraction of the vocabulary that is green at each step. Lower gamma makes each green hit more surprising, so the signal per token is stronger, but it constrains the model more. | 0.25 – 0.5 |
| `delta` | Logit bonus given to green tokens. Higher = stronger, more detectable watermark, but more visible quality degradation. | 1.5 – 4.0 |
| `ngram` | How many preceding tokens seed each step's split. Larger values make the watermark more context-dependent and harder to reverse-engineer, but more fragile to edits. | 2 – 4 |
| `z-threshold` | z above which detection is declared. 4.0 is standard practice and corresponds to a very low false-positive rate. | 4.0 |

### Quality vs. strength

`delta` is the main quality knob. Around `2.0` the text usually stays fluent. Push it to `6`–`8`
and you will see the model reach for odd word choices, because the bias starts overriding genuine
preference — that trade-off is inherent to the scheme, not a bug.

`gamma` and `ngram` are **detection** parameters and must match at detect time. `delta` only
affects generation and is not needed for detection.

## Important caveats

- **The key must be kept consistent.** Detecting with the wrong `key`, `gamma` or `ngram` fails
  silently: you get `z ≈ 0`, which is indistinguishable from "no watermark".
- **Detect the generated continuation, not the prompt.** The sampler's history starts empty at the
  first generated token, so the scored sequence is the generated text alone.
- **Re-tokenization drift.** Detection re-tokenizes text from a string, which may not reproduce the
  exact token ids that were generated (whitespace and merge boundaries can differ). This costs a
  little signal; it is usually not enough to hide a watermark over a few hundred tokens, but it is
  why short samples are unreliable.
- **Short text is not detectable.** The z-test needs a few hundred tokens for a confident verdict.
  Fewer than ~50 scored tokens is noise.
- The first `ngram` tokens are neither watermarked nor scored, since there is not yet enough context
  to derive a step seed.

## Project layout

```
WatermarkTool/
  Program.cs                                # CLI: generate / detect / rewatermark / selftest
  CommandLine.cs                            # tiny --flag value parser
  Watermarking/
    GreenListWatermarker.cs                 # PRF, green/red split, logit bias (no LLamaSharp dependency)
    WatermarkSamplingPipeline.cs            # LLamaSharp sampling pipeline integration
    WatermarkDetector.cs                    # z-test detector, tokenizer only
  Rewatermark/
    TokenSubstitutor.cs                     # teacher-force + substitute, for robustness testing
```

`GreenListWatermarker` is deliberately shared by the generator and the detector rather than being
reimplemented on each side — if the two ever diverged, detection would silently stop working.

### Note on LLamaSharp 0.27.0

The sampling extension point in this version is
`BaseSamplingPipeline.CreateChain(SafeLLamaContextHandle)`, which builds a native llama.cpp sampler
chain. A managed stage is injected via `chain.AddCustom(...)` implementing `ICustomSampler`. The
watermark stage is added **first**, ahead of top-k/top-p/temperature, so the bias can actually push
green tokens into the candidate set before truncation happens. `llama_sampler_sample` accepts the
chosen token into the chain itself, which is how the sampler keeps its token history in sync.
