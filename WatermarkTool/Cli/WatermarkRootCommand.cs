using System.CommandLine;

namespace WatermarkTool.Cli;

/// <summary>Composes the whole command tree.</summary>
internal sealed class WatermarkRootCommand : RootCommand
{
    public WatermarkRootCommand()
        : base(
            "WatermarkTool - token-level green-list text watermarking (Kirchenbauer et al., 2023). " +
            "--key, --gamma and --ngram must match between generate and detect, or detection fails " +
            "silently (z will sit around 0).")
    {
        Subcommands.Add(new GenerateCommand());
        Subcommands.Add(new DetectCommand());
        Subcommands.Add(new RewatermarkCommand());
        Subcommands.Add(new SelfTestCommand());
    }
}
