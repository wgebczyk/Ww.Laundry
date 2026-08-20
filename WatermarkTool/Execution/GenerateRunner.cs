using System.Globalization;
using LLama;
using LLama.Common;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Execution;

internal sealed record GenerateRequest(
    string ModelPath,
    string Prompt,
    GreenListWatermarker Watermarker,
    int ContextSize,
    int GpuLayers,
    int MaxTokens,
    float Temperature,
    int TopK,
    float TopP,
    uint? Seed);

/// <summary>Generates watermarked text with a local GGUF model.</summary>
internal static class GenerateRunner
{
    public static async Task<int> RunAsync(GenerateRequest request, CancellationToken cancellationToken)
    {
        var modelParams = new ModelParams(request.ModelPath)
        {
            ContextSize = (uint)request.ContextSize,
            GpuLayerCount = request.GpuLayers,
        };

        Console.Error.WriteLine($"loading model: {request.ModelPath}");
        using var weights = await LLamaWeights.LoadFromFileAsync(modelParams, cancellationToken);
        var executor = new StatelessExecutor(weights, modelParams);

        using var pipeline = new WatermarkSamplingPipeline(request.Watermarker)
        {
            Temperature = request.Temperature,
            TopK = request.TopK,
            TopP = request.TopP,
            Seed = request.Seed,
        };

        var inferenceParams = new InferenceParams
        {
            MaxTokens = request.MaxTokens,
            SamplingPipeline = pipeline,
        };

        Console.Error.WriteLine($"generating with watermark: {request.Watermarker}");
        Console.Error.WriteLine();

        await foreach (var chunk in executor.InferAsync(request.Prompt, inferenceParams, cancellationToken))
        {
            Console.Write(chunk);
        }

        Console.Out.Flush();
        PrintParameters(request.Watermarker, pipeline.History.Count);

        return 0;
    }

    private static void PrintParameters(GreenListWatermarker w, int tokensGenerated)
    {
        var inv = CultureInfo.InvariantCulture;
        Console.Error.WriteLine();
        Console.Error.WriteLine();
        Console.Error.WriteLine("--- watermark parameters (save these; detection needs all of them) ---");
        Console.Error.WriteLine($"  --key   {w.SecretKey}");
        Console.Error.WriteLine($"  --ngram {w.NgramLen}");
        Console.Error.WriteLine($"  --gamma {w.Gamma.ToString(inv)}");
        Console.Error.WriteLine($"  --delta {w.Delta.ToString(inv)} (generation only; not needed to detect)");
        Console.Error.WriteLine($"  tokens generated: {tokensGenerated}");
    }
}
