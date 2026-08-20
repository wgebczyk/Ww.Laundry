using System.CommandLine;
using WatermarkTool.Execution;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Cli;

/// <summary>Wires the <c>rewatermark</c> options to <see cref="RewatermarkRunner"/>.</summary>
internal sealed class RewatermarkCommand : Command
{
    public RewatermarkCommand()
        : base(
            "rewatermark",
            "Robustness harness: reports the old-key z-score before and after targeted token " +
            "substitution. Approximate by design - see the README.")
    {
        Options.Add(CliOptions.Model);
        Options.Add(CliOptions.Text);
        Options.Add(CliOptions.TextFile);
        Options.Add(CliOptions.OldKey);
        Options.Add(CliOptions.NewKey);
        Options.Add(CliOptions.Ngram);
        Options.Add(CliOptions.Gamma);
        Options.Add(CliOptions.Delta);
        Options.Add(CliOptions.ZThreshold);
        Options.Add(CliOptions.TopN);
        Options.Add(CliOptions.MaxLogitDrop);
        Options.Add(CliOptions.Ctx);
        Options.Add(CliOptions.GpuLayers);
        Options.Add(CliOptions.Out);

        Validators.Add(TextInput.Validate);

        SetAction(parseResult => RewatermarkRunner.Run(Bind(parseResult)));
    }

    private static RewatermarkRequest Bind(ParseResult p)
    {
        var ngram = p.GetValue(CliOptions.Ngram);
        var gamma = p.GetValue(CliOptions.Gamma);
        var delta = p.GetValue(CliOptions.Delta);

        return new RewatermarkRequest(
            ModelPath: p.GetValue(CliOptions.Model)!,
            OldWatermarker: new GreenListWatermarker(p.GetValue(CliOptions.OldKey), ngram, gamma, delta),
            NewWatermarker: new GreenListWatermarker(p.GetValue(CliOptions.NewKey), ngram, gamma, delta),
            Text: TextInput.Read(p),
            ZThreshold: p.GetValue(CliOptions.ZThreshold),
            ContextSize: p.GetValue(CliOptions.Ctx),
            GpuLayers: p.GetValue(CliOptions.GpuLayers),
            TopN: p.GetValue(CliOptions.TopN),
            MaxLogitDrop: p.GetValue(CliOptions.MaxLogitDrop),
            OutPath: p.GetValue(CliOptions.Out));
    }
}
