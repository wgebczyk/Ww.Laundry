using System.CommandLine;
using WatermarkTool.Execution;

namespace WatermarkTool.Cli;

/// <summary>Wires the <c>selftest</c> options to <see cref="SelfTestRunner"/>.</summary>
internal sealed class SelfTestCommand : Command
{
    public SelfTestCommand()
        : base("selftest", "Verify the watermark statistics without needing a model file.")
    {
        Options.Add(CliOptions.Tokens);
        Options.Add(CliOptions.Ngram);
        Options.Add(CliOptions.Gamma);
        Options.Add(CliOptions.Delta);

        SetAction(parseResult => SelfTestRunner.Run(Bind(parseResult)));
    }

    private static SelfTestRequest Bind(ParseResult p) => new(
        TokenCount: p.GetValue(CliOptions.Tokens),
        Ngram: p.GetValue(CliOptions.Ngram),
        Gamma: p.GetValue(CliOptions.Gamma),
        Delta: p.GetValue(CliOptions.Delta));
}
