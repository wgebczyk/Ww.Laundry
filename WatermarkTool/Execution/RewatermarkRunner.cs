using System.Globalization;
using System.Text;
using LLama;
using LLama.Common;
using WatermarkTool.Rewatermark;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Execution;

internal sealed record RewatermarkRequest(
    string ModelPath,
    GreenListWatermarker OldWatermarker,
    GreenListWatermarker NewWatermarker,
    string Text,
    double ZThreshold,
    int ContextSize,
    int GpuLayers,
    int TopN,
    float MaxLogitDrop,
    string? OutPath);

/// <summary>
/// Robustness harness: measures how much of an existing watermark survives targeted token
/// substitution. Reports the z-score under the OLD key before and after the rewrite, so the
/// degradation is the actual output of interest, plus the z under the NEW key.
/// </summary>
internal static class RewatermarkRunner
{
    public static int Run(RewatermarkRequest request)
    {
        var inv = CultureInfo.InvariantCulture;
        var oldWm = request.OldWatermarker;
        var newWm = request.NewWatermarker;

        var modelParams = new ModelParams(request.ModelPath)
        {
            ContextSize = (uint)request.ContextSize,
            GpuLayerCount = request.GpuLayers,
        };

        Console.Error.WriteLine($"loading model: {request.ModelPath}");
        using var weights = LLamaWeights.LoadFromFile(modelParams);

        var tokens = weights.Tokenize(request.Text, add_bos: false, special: false, Encoding.UTF8)
                            .Select(t => (int)t)
                            .ToArray();

        // Step 1: is this text actually watermarked under the old key? Measuring degradation from a
        // baseline that was never watermarked tells you nothing.
        var before = WatermarkDetector.Detect(tokens, oldWm, request.ZThreshold);
        Console.Error.WriteLine();
        Console.Error.WriteLine("=== BEFORE: input scored against the old key ===");
        ResultPrinter.PrintTo(Console.Error, before, oldWm);

        if (!before.Detected)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine(
                "note: the input does not carry a detectable watermark under --old-key, so the " +
                "before/after comparison below is not meaningful as a robustness measurement.");
        }

        Console.Error.WriteLine();
        Console.Error.WriteLine("teacher-forcing and substituting...");

        var result = TokenSubstitutor.Run(
            weights,
            modelParams,
            tokens,
            newWm,
            topN: request.TopN,
            maxLogitDrop: request.MaxLogitDrop);

        // Score the modified token sequence directly...
        var afterOld = WatermarkDetector.Detect(result.ModifiedTokens, oldWm, request.ZThreshold);
        var afterNew = WatermarkDetector.Detect(result.ModifiedTokens, newWm, request.ZThreshold);

        // ...and also as a real detector would see it, by re-tokenizing the emitted text.
        var retokenized = weights.Tokenize(result.ModifiedText, add_bos: false, special: false, Encoding.UTF8)
                                 .Select(t => (int)t)
                                 .ToArray();
        var afterOldRetokenized = WatermarkDetector.Detect(retokenized, oldWm, request.ZThreshold);

        Console.Error.WriteLine();
        Console.Error.WriteLine("=== AFTER: modified tokens scored against the old key ===");
        ResultPrinter.PrintTo(Console.Error, afterOld, oldWm);
        Console.Error.WriteLine();
        Console.Error.WriteLine("=== AFTER: re-tokenized output scored against the old key (what a real detector sees) ===");
        ResultPrinter.PrintTo(Console.Error, afterOldRetokenized, oldWm);
        Console.Error.WriteLine();
        Console.Error.WriteLine("=== AFTER: modified tokens scored against the new key ===");
        ResultPrinter.PrintTo(Console.Error, afterNew, newWm);

        Console.Error.WriteLine();
        Console.Error.WriteLine("=== substitution summary ===");
        Console.Error.WriteLine($"scoreable positions        : {result.EligiblePositions}");
        Console.Error.WriteLine($"red under new key          : {result.RedUnderNewKey}");
        Console.Error.WriteLine($"substituted                : {result.Substituted}");
        Console.Error.WriteLine($"left unchanged (no candidate): {result.LeftUnchanged}");
        Console.Error.WriteLine();
        Console.Error.WriteLine($"old-key z: {before.ZScore.ToString("F3", inv)} -> {afterOld.ZScore.ToString("F3", inv)} " +
                                $"(re-tokenized {afterOldRetokenized.ZScore.ToString("F3", inv)})");
        Console.Error.WriteLine(
            "Re-watermarking is approximate: logits come from a single forward pass over the original " +
            "tokens, so it neither guarantees the old signature is erased nor that fluency is preserved.");

        if (request.OutPath is not null)
        {
            File.WriteAllText(request.OutPath, result.ModifiedText);
            Console.Error.WriteLine($"wrote modified text to {request.OutPath}");
        }
        else
        {
            Console.Write(result.ModifiedText);
        }

        return 0;
    }
}
