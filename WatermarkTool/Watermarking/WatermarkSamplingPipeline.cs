using LLama.Native;
using LLama.Sampling;

namespace WatermarkTool.Watermarking;

/// <summary>
/// An <see cref="ISamplingPipeline"/> (via <see cref="BaseSamplingPipeline"/>) that applies the
/// green-list watermark bias before the usual temperature / top-k / top-p stages.
///
/// In LLamaSharp 0.27 the extension point is <see cref="BaseSamplingPipeline.CreateChain"/>, which
/// builds a native llama.cpp sampler chain. A managed stage is injected with
/// <see cref="SafeLLamaSamplerChainHandle.AddCustom{TSampler}"/>; that stage is
/// <see cref="WatermarkSampler"/> below, which both biases logits and tracks the accepted-token
/// history used to seed each step.
/// </summary>
public sealed class WatermarkSamplingPipeline : BaseSamplingPipeline
{
    private readonly WatermarkSampler _sampler;

    /// <summary>Sampling temperature. Values &lt;= 0 select greedy sampling.</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Keep only the top K tokens. Values &lt;= 0 disable top-k.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Nucleus sampling threshold. Values &gt;= 1 disable top-p.</summary>
    public float TopP { get; init; } = 0.95f;

    /// <summary>Minimum number of candidates truncation stages must keep.</summary>
    public int MinKeep { get; init; } = 1;

    /// <summary>RNG seed for the final distribution sampler. Null means a random seed.</summary>
    public uint? Seed { get; init; }

    public GreenListWatermarker Watermarker => _sampler.Watermarker;

    public WatermarkSamplingPipeline(ulong secretKey, int ngramLen = 4, float gamma = 0.5f, float delta = 2.0f)
        : this(new GreenListWatermarker(secretKey, ngramLen, gamma, delta))
    {
    }

    public WatermarkSamplingPipeline(GreenListWatermarker watermarker)
    {
        _sampler = new WatermarkSampler(watermarker);
    }

    /// <summary>Token ids sampled so far in the current generation.</summary>
    public IReadOnlyList<int> History => _sampler.History;

    protected override SafeLLamaSamplerChainHandle CreateChain(SafeLLamaContextHandle context)
    {
        var chain = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());

        // Watermark bias must run on raw logits, before any truncation, so that green tokens are
        // actually able to enter the top-k / top-p candidate set.
        chain.AddCustom(_sampler);

        if (Temperature <= 0f)
        {
            chain.AddGreedySampler();
            return chain;
        }

        if (TopK > 0)
        {
            chain.AddTopK(TopK);
        }

        if (TopP < 1f)
        {
            chain.AddTopP(TopP, (nint)Math.Max(1, MinKeep));
        }

        chain.AddTemperature(Temperature);
        chain.AddDistributionSampler(Seed ?? (uint)Random.Shared.Next());

        return chain;
    }

    /// <summary>
    /// The managed sampler stage. <c>llama_sampler_sample</c> applies the chain and then accepts the
    /// chosen token, so <see cref="Accept"/> fires exactly once per generated token and is the
    /// authoritative place to grow the history.
    /// </summary>
    private sealed class WatermarkSampler : ICustomSampler
    {
        private readonly List<int> _history;

        public GreenListWatermarker Watermarker { get; }

        public string Name => "watermark-greenlist";

        public IReadOnlyList<int> History => _history;

        public WatermarkSampler(GreenListWatermarker watermarker, IEnumerable<int>? history = null)
        {
            Watermarker = watermarker;
            _history = history is null ? new List<int>() : new List<int>(history);
        }

        public void Apply(ref LLamaTokenDataArrayNative tokenData)
        {
            if (!Watermarker.CanScore(_history.Count))
            {
                return;
            }

            var seed = Watermarker.StepSeed(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_history));

            var candidates = tokenData.Data[..(int)tokenData.Size];
            for (var i = 0; i < candidates.Length; i++)
            {
                if (Watermarker.IsGreen(seed, (int)candidates[i].ID))
                {
                    candidates[i].Logit += Watermarker.Delta;
                }
            }

            // Logits changed, so any previously established ordering is no longer valid.
            tokenData.Sorted = false;
        }

        public void Accept(LLamaToken token) => _history.Add((int)token);

        public void Reset() => _history.Clear();

        public ICustomSampler Clone() => new WatermarkSampler(Watermarker, _history);

        public void Dispose()
        {
        }
    }
}
