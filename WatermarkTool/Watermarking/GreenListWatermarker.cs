namespace WatermarkTool.Watermarking;

/// <summary>
/// Core green-list / red-list watermark logic (Kirchenbauer et al., 2023).
///
/// Every generation step derives a pseudo-random seed from the secret key plus the last
/// <see cref="NgramLen"/> tokens. That seed deterministically partitions the vocabulary into a
/// "green" list (a <see cref="Gamma"/> fraction) and a "red" list. Green tokens get
/// <see cref="Delta"/> added to their logit, so watermarked text over-samples green tokens.
/// A detector that knows the key can recompute the same partition and z-test the green rate.
///
/// This type deliberately has no LLamaSharp dependency: the generator and the detector share
/// this exact implementation, which is what makes detection reproducible.
/// </summary>
public sealed class GreenListWatermarker
{
    public ulong SecretKey { get; }
    public int NgramLen { get; }
    public float Gamma { get; }
    public float Delta { get; }

    public GreenListWatermarker(ulong secretKey, int ngramLen = 4, float gamma = 0.5f, float delta = 2.0f)
    {
        if (ngramLen < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(ngramLen), "ngramLen must be >= 1.");
        }

        if (gamma is <= 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(gamma), "gamma must be in (0,1).");
        }

        if (delta < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(delta), "delta must be >= 0.");
        }

        SecretKey = secretKey;
        NgramLen = ngramLen;
        Gamma = gamma;
        Delta = delta;
    }

    /// <summary>
    /// splitmix64-style mixer. Deterministic and stable across .NET versions, unlike
    /// <see cref="Random"/>, whose algorithm is explicitly not guaranteed stable.
    /// </summary>
    public static ulong MixHash(ulong a, ulong b)
    {
        unchecked
        {
            var z = a ^ (b + 0x9E3779B97F4A7C15UL + (a << 6) + (a >> 2));
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    /// <summary>
    /// Seed for the current step: fold <see cref="MixHash"/> over the secret key and the last
    /// <see cref="NgramLen"/> entries of <paramref name="prevTokens"/>.
    /// Callers must only score positions where at least <see cref="NgramLen"/> tokens of history
    /// exist, so that generation and detection see identical context windows.
    /// </summary>
    public ulong StepSeed(ReadOnlySpan<int> prevTokens)
    {
        if (prevTokens.Length < NgramLen)
        {
            throw new ArgumentException(
                $"Need at least {NgramLen} tokens of history to derive a step seed (got {prevTokens.Length}).",
                nameof(prevTokens));
        }

        var window = prevTokens[^NgramLen..];

        var seed = MixHash(SecretKey, (ulong)NgramLen);
        foreach (var token in window)
        {
            seed = MixHash(seed, (ulong)(uint)token);
        }

        return seed;
    }

    /// <summary>
    /// True if <paramref name="tokenId"/> is on the green list for this step. Pure function of
    /// (stepSeed, tokenId) with no side effects, so the detector reproduces it exactly.
    /// </summary>
    public bool IsGreen(ulong stepSeed, int tokenId)
    {
        var h = MixHash(stepSeed, (ulong)(uint)tokenId);

        // Top 53 bits -> uniform double in [0,1).
        var u = (h >> 11) * (1.0 / 9007199254740992.0);
        return u < Gamma;
    }

    /// <summary>True if this position has enough history to be watermarked / scored.</summary>
    public bool CanScore(int historyLength) => historyLength >= NgramLen;

    /// <summary>
    /// Adds <see cref="Delta"/> to the logit of every green token, in place.
    /// Assumes <paramref name="logits"/> is indexed by token id (a dense vocab-sized array).
    /// No-op when there is not yet enough history, keeping parity with the detector.
    /// </summary>
    public void ApplyBias(Span<float> logits, ReadOnlySpan<int> prevTokens)
    {
        if (!CanScore(prevTokens.Length))
        {
            return;
        }

        var seed = StepSeed(prevTokens);
        for (var tokenId = 0; tokenId < logits.Length; tokenId++)
        {
            if (IsGreen(seed, tokenId))
            {
                logits[tokenId] += Delta;
            }
        }
    }

    public override string ToString() =>
        $"key={SecretKey} ngram={NgramLen} gamma={Gamma} delta={Delta}";
}
