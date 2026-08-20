using System.Globalization;
using WatermarkTool.Watermarking;

namespace WatermarkTool.Execution;

/// <summary>Renders a <see cref="WatermarkDetectionResult"/> as a fixed-width report.</summary>
internal static class ResultPrinter
{
    public static void Print(WatermarkDetectionResult r, GreenListWatermarker w) =>
        PrintTo(Console.Out, r, w);

    public static void PrintTo(TextWriter output, WatermarkDetectionResult r, GreenListWatermarker w)
    {
        var inv = CultureInfo.InvariantCulture;
        output.WriteLine($"watermark params : {w}");
        output.WriteLine($"tokens (total)   : {r.TotalTokens}");
        output.WriteLine($"tokens (scored)  : {r.ScoredTokens}");
        output.WriteLine($"green tokens     : {r.GreenTokens} ({r.GreenFraction.ToString("P2", inv)})");
        output.WriteLine($"expected green   : {r.Expected.ToString("F2", inv)} (sd {r.StdDev.ToString("F2", inv)})");
        output.WriteLine($"z-score          : {r.ZScore.ToString("F3", inv)}");
        output.WriteLine($"p-value          : {r.PValue.ToString("G4", inv)}");
        output.WriteLine($"threshold        : z > {r.ZThreshold.ToString("F2", inv)}");
        output.WriteLine($"verdict          : {r.Verdict}");
    }
}
