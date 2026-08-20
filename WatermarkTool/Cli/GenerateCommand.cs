using System.CommandLine;
using WatermarkTool.Execution;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Cli;

/// <summary>Wires the <c>generate</c> options to <see cref="GenerateRunner"/>.</summary>
internal sealed class GenerateCommand : Command
{
    public GenerateCommand()
        : base("generate", "Generate watermarked text with a local GGUF model.")
    {
        Options.Add(CliOptions.Model);
        Options.Add(CliOptions.Prompt);
        Options.Add(CliOptions.Key);
        Options.Add(CliOptions.Ngram);
        Options.Add(CliOptions.Gamma);
        Options.Add(CliOptions.Delta);
        Options.Add(CliOptions.MaxTokens);
        Options.Add(CliOptions.Temp);
        Options.Add(CliOptions.TopK);
        Options.Add(CliOptions.TopP);
        Options.Add(CliOptions.Seed);
        Options.Add(CliOptions.Ctx);
        Options.Add(CliOptions.GpuLayers);

        SetAction((parseResult, cancellationToken) => GenerateRunner.RunAsync(Bind(parseResult), cancellationToken));
    }

    private static GenerateRequest Bind(ParseResult p) => new(
        ModelPath: p.GetValue(CliOptions.Model)!,
        Prompt: p.GetValue(CliOptions.Prompt)!,
        Watermarker: new GreenListWatermarker(
            p.GetValue(CliOptions.Key),
            p.GetValue(CliOptions.Ngram),
            p.GetValue(CliOptions.Gamma),
            p.GetValue(CliOptions.Delta)),
        ContextSize: p.GetValue(CliOptions.Ctx),
        GpuLayers: p.GetValue(CliOptions.GpuLayers),
        MaxTokens: p.GetValue(CliOptions.MaxTokens),
        Temperature: p.GetValue(CliOptions.Temp),
        TopK: p.GetValue(CliOptions.TopK),
        TopP: p.GetValue(CliOptions.TopP),
        Seed: p.GetValue(CliOptions.Seed));
}
