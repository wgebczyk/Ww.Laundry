using WatermarkTool.Watermarking;

namespace WatermarkTool.Execution;

internal sealed record SelfTestRequest(int TokenCount, int Ngram, float Gamma, float Delta);

/// <summary>
/// End-to-end check of the watermark statistics that does not need a GGUF model: it simulates a
/// language model with a fixed random logit distribution, samples with and without the bias, and
/// verifies the three acceptance criteria (same key detects, wrong key does not, unwatermarked
/// text does not).
/// </summary>
internal static class SelfTestRunner
{
    private const int VocabSize = 32000;
    private const ulong KeyA = 12345678901234567UL;
    private const ulong KeyB = 98765432109876543UL;

    public static int Run(SelfTestRequest request)
    {
        var wmA = new GreenListWatermarker(KeyA, request.Ngram, request.Gamma, request.Delta);
        var wmB = new GreenListWatermarker(KeyB, request.Ngram, request.Gamma, request.Delta);

        var watermarked = SimulateGeneration(VocabSize, request.TokenCount, wmA, seed: 42);
        var plain = SimulateGeneration(VocabSize, request.TokenCount, watermarker: null, seed: 42);

        var sameKey = WatermarkDetector.Detect(watermarked, wmA);
        var wrongKey = WatermarkDetector.Detect(watermarked, wmB);
        var noWatermark = WatermarkDetector.Detect(plain, wmA);

        Console.WriteLine("=== watermarked text, correct key (expect z > 4) ===");
        ResultPrinter.Print(sameKey, wmA);
        Console.WriteLine();
        Console.WriteLine("=== watermarked text, wrong key (expect |z| < 3) ===");
        ResultPrinter.Print(wrongKey, wmB);
        Console.WriteLine();
        Console.WriteLine("=== unwatermarked text, correct key (expect |z| < 3) ===");
        ResultPrinter.Print(noWatermark, wmA);
        Console.WriteLine();

        var checks = new (string Name, bool Ok)[]
        {
            ("watermarked text detected with the generating key", sameKey.ZScore > 4.0),
            ("watermarked text NOT detected with a different key", Math.Abs(wrongKey.ZScore) < 3.0),
            ("unwatermarked text NOT detected", Math.Abs(noWatermark.ZScore) < 3.0),
            ("StepSeed/IsGreen are deterministic", DeterminismCheck(wmA)),
        };

        var allOk = true;
        foreach (var (name, ok) in checks)
        {
            Console.WriteLine($"[{(ok ? "PASS" : "FAIL")}] {name}");
            allOk &= ok;
        }

        return allOk ? 0 : 1;
    }

    private static bool DeterminismCheck(GreenListWatermarker w)
    {
        int[] history = [11, 22, 33, 44, 55, 66];
        var seed1 = w.StepSeed(history);
        var seed2 = w.StepSeed(history);
        if (seed1 != seed2)
        {
            return false;
        }

        // A different context must produce a different partition.
        var other = w.StepSeed([11, 22, 33, 44, 55, 67]);
        if (seed1 == other)
        {
            return false;
        }

        // Roughly gamma of the vocabulary should be green.
        var green = 0;
        for (var i = 0; i < 20000; i++)
        {
            if (w.IsGreen(seed1, i))
            {
                green++;
            }
        }

        var fraction = green / 20000.0;
        return Math.Abs(fraction - w.Gamma) < 0.02;
    }

    /// <summary>
    /// Samples a token sequence from a synthetic, fixed "model" distribution. With a watermarker the
    /// green bias is applied first, exactly as <see cref="WatermarkSamplingPipeline"/> does at runtime.
    /// </summary>
    private static int[] SimulateGeneration(int vocabSize, int tokenCount, GreenListWatermarker? watermarker, int seed)
    {
        var rng = new Random(seed);

        // Fixed per-token "model preference", Zipf-like so the distribution is peaked like a real LM.
        var baseLogits = new float[vocabSize];
        for (var i = 0; i < vocabSize; i++)
        {
            baseLogits[i] = (float)(-Math.Log(i + 1) + rng.NextDouble());
        }

        var history = new List<int>(tokenCount);
        var logits = new float[vocabSize];
        var probs = new double[vocabSize];

        for (var step = 0; step < tokenCount; step++)
        {
            baseLogits.CopyTo(logits, 0);

            // Rotate the preference each step so the sequence is not a constant token.
            var offset = rng.Next(vocabSize);
            for (var i = 0; i < vocabSize; i++)
            {
                logits[i] += baseLogits[(i + offset) % vocabSize] * 0.5f;
            }

            watermarker?.ApplyBias(logits, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history));

            var max = float.NegativeInfinity;
            foreach (var l in logits)
            {
                max = Math.Max(max, l);
            }

            var sum = 0.0;
            for (var i = 0; i < vocabSize; i++)
            {
                probs[i] = Math.Exp(logits[i] - max);
                sum += probs[i];
            }

            var target = rng.NextDouble() * sum;
            var acc = 0.0;
            var chosen = vocabSize - 1;
            for (var i = 0; i < vocabSize; i++)
            {
                acc += probs[i];
                if (acc >= target)
                {
                    chosen = i;
                    break;
                }
            }

            history.Add(chosen);
        }

        return history.ToArray();
    }
}
