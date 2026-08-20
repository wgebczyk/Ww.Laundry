using System.CommandLine;
using System.CommandLine.Parsing;

namespace WatermarkTool.Cli;

/// <summary>Helpers shared by the commands that take text either inline or from a file.</summary>
internal static class TextInput
{
    /// <summary>Rejects passing both <c>--text</c> and <c>--text-file</c>, or neither.</summary>
    public static void Validate(CommandResult result)
    {
        var hasText = result.GetResult(CliOptions.Text) is not null;
        var hasFile = result.GetResult(CliOptions.TextFile) is not null;

        if (hasText == hasFile)
        {
            result.AddError("Specify exactly one of --text or --text-file.");
        }
    }

    public static string Read(ParseResult parseResult)
    {
        var textFile = parseResult.GetValue(CliOptions.TextFile);
        var value = textFile is not null
            ? File.ReadAllText(textFile)
            : parseResult.GetValue(CliOptions.Text);

        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException("Input text is empty.");

        return value;
    }
}
