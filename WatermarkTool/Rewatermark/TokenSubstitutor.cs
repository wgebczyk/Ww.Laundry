using LLama;
using LLama.Abstractions;
using LLama.Native;
using WatermarkTool.Watermarking;
using Vocabulary = LLama.Native.SafeLlamaModelHandle.Vocabulary;

namespace WatermarkTool.Rewatermark;

/// <summary>Outcome of a re-watermarking pass, framed as a robustness measurement.</summary>
public sealed record RewatermarkResult(
    int[] OriginalTokens,
    int[] ModifiedTokens,
    string ModifiedText,
    int EligiblePositions,
    int RedUnderNewKey,
    int Substituted,
    int LeftUnchanged);

/// <summary>
/// Teacher-forces existing text through the model and substitutes tokens at positions where a
/// near-equivalent alternative flips red -> green under a different key.
///
/// This exists to *measure* how much of a watermark's z-score survives targeted substitution — it
/// is an attack harness for evaluating watermark robustness, and it is approximate by construction
/// (see the README). Two approximations matter:
///
/// 1. Logits come from a single forward pass over the ORIGINAL token sequence. Once a token is
///    substituted, the real model conditional for every later position has changed, but we do not
///    re-run the forward pass, so later scores drift from what the model would actually predict.
/// 2. "Meaning is preserved" is approximated by a logit-gap threshold, which is a crude proxy for
///    semantic equivalence. Substitutions can and do degrade fluency.
/// </summary>
public static class TokenSubstitutor
{
    public static RewatermarkResult Run(
        LLamaWeights weights,
        IContextParams contextParams,
        IReadOnlyList<int> tokenIds,
        GreenListWatermarker newWatermarker,
        int topN = 32,
        float maxLogitDrop = 2.0f)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(newWatermarker);

        var original = tokenIds.ToArray();
        var modified = (int[])original.Clone();

        if (original.Length <= newWatermarker.NgramLen)
        {
            return new RewatermarkResult(original, modified, string.Empty, 0, 0, 0, 0);
        }

        using var context = weights.CreateContext(contextParams);

        if (original.Length > context.ContextSize)
        {
            throw new InvalidOperationException(
                $"Text is {original.Length} tokens but the context is {context.ContextSize}. Raise --ctx.");
        }

        var vocab = weights.Vocab;
        var batch = new LLamaBatch();
        var batchSize = (int)Math.Max(1, Math.Min(context.BatchSize, (uint)original.Length));

        var eligible = 0;
        var red = 0;
        var substituted = 0;

        // Logits for the last position of the previous chunk, needed to score the first token of
        // the next chunk (position t is predicted by the logits at position t-1).
        float[]? carryLogits = null;

        for (var chunkStart = 0; chunkStart < original.Length; chunkStart += batchSize)
        {
            var chunkEnd = Math.Min(chunkStart + batchSize, original.Length);

            batch.Clear();
            var logitIndex = new int[chunkEnd - chunkStart];
            for (var i = chunkStart; i < chunkEnd; i++)
            {
                logitIndex[i - chunkStart] = batch.Add(original[i], i, LLamaSeqId.Zero, logits: true);
            }

            var decodeResult = context.Decode(batch);
            if (decodeResult != DecodeResult.Ok)
            {
                throw new InvalidOperationException($"Teacher-forcing decode failed: {decodeResult}");
            }

            for (var t = chunkStart; t < chunkEnd; t++)
            {
                // Positions before the n-gram window are neither watermarked nor scored.
                if (t < newWatermarker.NgramLen)
                {
                    continue;
                }

                eligible++;

                var seed = newWatermarker.StepSeed(modified.AsSpan(0, t));
                if (newWatermarker.IsGreen(seed, modified[t]))
                {
                    continue;
                }

                red++;

                ReadOnlySpan<float> logits = t == chunkStart
                    ? (carryLogits is null ? default : carryLogits.AsSpan())
                    : context.NativeHandle.GetLogitsIth(logitIndex[t - 1 - chunkStart]);

                if (logits.IsEmpty)
                {
                    continue;
                }

                var replacement = FindGreenAlternative(
                    logits, seed, modified[t], newWatermarker, vocab, topN, maxLogitDrop);

                if (replacement >= 0)
                {
                    modified[t] = replacement;
                    substituted++;
                }
            }

            carryLogits = context.NativeHandle.GetLogitsIth(logitIndex[^1]).ToArray();
        }

        var decoder = new StreamingTokenDecoder(context);
        foreach (var token in modified)
        {
            decoder.Add(token);
        }

        var text = decoder.Read();

        return new RewatermarkResult(
            OriginalTokens: original,
            ModifiedTokens: modified,
            ModifiedText: text,
            EligiblePositions: eligible,
            RedUnderNewKey: red,
            Substituted: substituted,
            LeftUnchanged: red - substituted);
    }

    /// <summary>
    /// Highest-logit token that is green under the new key, is within <paramref name="maxLogitDrop"/>
    /// of the best token at this position, and ranks inside the top <paramref name="topN"/>.
    /// Returns -1 when nothing suitable exists, in which case the position is left alone.
    /// </summary>
    private static int FindGreenAlternative(
        ReadOnlySpan<float> logits,
        ulong seed,
        int originalToken,
        GreenListWatermarker watermarker,
        Vocabulary vocab,
        int topN,
        float maxLogitDrop)
    {
        var maxLogit = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
            {
                maxLogit = logits[i];
            }
        }

        var bestToken = -1;
        var bestLogit = float.NegativeInfinity;

        for (var i = 0; i < logits.Length; i++)
        {
            if (i == originalToken || logits[i] <= bestLogit)
            {
                continue;
            }

            if (!watermarker.IsGreen(seed, i))
            {
                continue;
            }

            // Never swap in a control / end-of-generation token: that would truncate or corrupt
            // the text rather than paraphrase it.
            var token = (LLamaToken)i;
            if (token.IsControl(vocab) || token.IsEndOfGeneration(vocab))
            {
                continue;
            }

            bestToken = i;
            bestLogit = logits[i];
        }

        if (bestToken < 0 || bestLogit < maxLogit - maxLogitDrop)
        {
            return -1;
        }

        // Enforce the top-N rank cap.
        var betterCount = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > bestLogit && ++betterCount >= topN)
            {
                return -1;
            }
        }

        return bestToken;
    }
}
