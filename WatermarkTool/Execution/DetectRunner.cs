using System.Text;
using LLama;
using LLama.Common;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Execution;

internal sealed record DetectRequest(
    string ModelPath,
    GreenListWatermarker Watermarker,
    string Text,
    double ZThreshold);

/// <summary>Scores text for a green-list watermark.</summary>
internal static class DetectRunner
{
    public static int Run(DetectRequest request)
    {
        // Detection needs the tokenizer only, so skip loading the weights entirely.
        var modelParams = new ModelParams(request.ModelPath)
        {
            VocabOnly = true,
            GpuLayerCount = 0,
        };

        using var weights = LLamaWeights.LoadFromFile(modelParams);
        var tokens = weights.Tokenize(request.Text, add_bos: false, special: false, Encoding.UTF8)
                            .Select(t => (int)t)
                            .ToArray();

        var result = WatermarkDetector.Detect(tokens, request.Watermarker, request.ZThreshold);
        ResultPrinter.Print(result, request.Watermarker);

        return 0;
    }
}
