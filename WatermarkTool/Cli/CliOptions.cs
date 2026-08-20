using System.CommandLine;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Cli;

/// <summary>
/// Shared option definitions. The instances are singletons on purpose: a command reads back its
/// parsed value with the very same object it was registered with.
/// </summary>
internal static class CliOptions
{
    public static readonly Option<string> Model = new("--model")
    {
        Description = "Path to the GGUF model file.",
        Required = true,
    };

    public static readonly Option<string> Prompt = new("--prompt")
    {
        Description = "Prompt to generate from.",
        Required = true,
    };

    public static readonly Option<ulong> Key = new("--key")
    {
        Description = "Secret key selecting the green-list partition. Must match between generate and detect.",
        Required = true,
    };

    public static readonly Option<ulong> OldKey = new("--old-key")
    {
        Description = "Key the input text is expected to be watermarked with.",
        Required = true,
    };

    public static readonly Option<ulong> NewKey = new("--new-key")
    {
        Description = "Key to re-watermark the text with.",
        Required = true,
    };

    public static readonly Option<int> Ngram = new("--ngram")
    {
        Description = "Number of preceding tokens seeding the green list. Must match between generate and detect.",
        DefaultValueFactory = _ => 4,
    };

    public static readonly Option<float> Gamma = new("--gamma")
    {
        Description = "Fraction of the vocabulary placed on the green list. Must match between generate and detect.",
        DefaultValueFactory = _ => 0.5f,
    };

    public static readonly Option<float> Delta = new("--delta")
    {
        Description = "Logit bias added to green tokens. Generation only; not needed to detect.",
        DefaultValueFactory = _ => 2.0f,
    };

    public static readonly Option<double> ZThreshold = new("--z-threshold")
    {
        Description = "z-score above which the text is declared watermarked.",
        DefaultValueFactory = _ => WatermarkDetector.DefaultZThreshold,
    };

    public static readonly Option<int> Ctx = new("--ctx")
    {
        Description = "Model context size.",
        DefaultValueFactory = _ => 4096,
    };

    public static readonly Option<int> GpuLayers = new("--gpu-layers")
    {
        Description = "Number of layers to offload to the GPU.",
        DefaultValueFactory = _ => 0,
    };

    public static readonly Option<int> MaxTokens = new("--max-tokens")
    {
        Description = "Maximum number of tokens to generate.",
        DefaultValueFactory = _ => 512,
    };

    public static readonly Option<float> Temp = new("--temp")
    {
        Description = "Sampling temperature.",
        DefaultValueFactory = _ => 0.8f,
    };

    public static readonly Option<int> TopK = new("--top-k")
    {
        Description = "Top-k sampling cutoff.",
        DefaultValueFactory = _ => 40,
    };

    public static readonly Option<float> TopP = new("--top-p")
    {
        Description = "Top-p (nucleus) sampling cutoff.",
        DefaultValueFactory = _ => 0.95f,
    };

    public static readonly Option<uint?> Seed = new("--seed")
    {
        Description = "Sampler seed. Random when omitted.",
    };

    public static readonly Option<string?> Text = new("--text")
    {
        Description = "Text to analyse. Mutually exclusive with --text-file.",
    };

    public static readonly Option<string?> TextFile = new("--text-file")
    {
        Description = "File containing the text to analyse. Mutually exclusive with --text.",
    };

    public static readonly Option<int> TopN = new("--top-n")
    {
        Description = "Number of candidate replacement tokens considered per position.",
        DefaultValueFactory = _ => 32,
    };

    public static readonly Option<float> MaxLogitDrop = new("--max-logit-drop")
    {
        Description = "Maximum logit loss accepted when substituting a token.",
        DefaultValueFactory = _ => 2.0f,
    };

    public static readonly Option<string?> Out = new("--out")
    {
        Description = "File to write the modified text to. Written to stdout when omitted.",
    };

    public static readonly Option<int> Tokens = new("--tokens")
    {
        Description = "Number of tokens to simulate.",
        DefaultValueFactory = _ => 400,
    };
}
