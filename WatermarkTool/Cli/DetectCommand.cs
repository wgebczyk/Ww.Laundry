using System.CommandLine;
using WatermarkTool.Execution;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Cli;

/// <summary>Wires the <c>detect</c> options to <see cref="DetectRunner"/>.</summary>
internal sealed class DetectCommand : Command
{
    public DetectCommand()
        : base("detect", "Score text for a green-list watermark. Only the tokenizer is loaded.")
    {
        Options.Add(CliOptions.Model);
        Options.Add(CliOptions.Text);
        Options.Add(CliOptions.TextFile);
        Options.Add(CliOptions.Key);
        Options.Add(CliOptions.Ngram);
        Options.Add(CliOptions.Gamma);
        Options.Add(CliOptions.ZThreshold);

        Validators.Add(TextInput.Validate);

        SetAction(parseResult => DetectRunner.Run(Bind(parseResult)));
    }

    private static DetectRequest Bind(ParseResult p) => new(
        ModelPath: p.GetValue(CliOptions.Model)!,
        // Delta biases generation only, so it plays no part in detection.
        Watermarker: new GreenListWatermarker(
            p.GetValue(CliOptions.Key),
            p.GetValue(CliOptions.Ngram),
            p.GetValue(CliOptions.Gamma),
            delta: 0f),
        Text: TextInput.Read(p),
        ZThreshold: p.GetValue(CliOptions.ZThreshold));
}
