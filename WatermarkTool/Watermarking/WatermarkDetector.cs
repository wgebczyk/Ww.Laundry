namespace WatermarkTool.Watermarking;

/// <summary>Result of a watermark z-test over a token sequence.</summary>
public sealed record WatermarkDetectionResult(
    int TotalTokens,
    int ScoredTokens,
    int GreenTokens,
    double GreenFraction,
    double Expected,
    double StdDev,
    double ZScore,
    double PValue,
    double ZThreshold,
    bool Detected)
{
    public string Verdict => Detected
        ? "WATERMARK DETECTED"
        : "no watermark detected";
}

/// <summary>
/// Detects the green-list watermark in a token sequence. Needs only the tokenizer and the key —
/// never the model's weights or a forward pass — because <see cref="GreenListWatermarker.StepSeed"/>
/// and <see cref="GreenListWatermarker.IsGreen"/> are pure functions of the token history.
/// </summary>
public static class WatermarkDetector
{
    public const double DefaultZThreshold = 4.0;

    public static WatermarkDetectionResult Detect(
        IReadOnlyList<int> tokenIds,
        GreenListWatermarker watermarker,
        double zThreshold = DefaultZThreshold)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(watermarker);

        var tokens = tokenIds as int[] ?? tokenIds.ToArray();

        var green = 0;
        var scored = 0;

        for (var t = watermarker.NgramLen; t < tokens.Length; t++)
        {
            var seed = watermarker.StepSeed(tokens.AsSpan(0, t));
            if (watermarker.IsGreen(seed, tokens[t]))
            {
                green++;
            }

            scored++;
        }

        var gamma = (double)watermarker.Gamma;
        var expected = gamma * scored;
        var stdDev = Math.Sqrt(scored * gamma * (1.0 - gamma));

        var z = stdDev > 0 ? (green - expected) / stdDev : 0.0;
        var p = OneSidedPValue(z);

        return new WatermarkDetectionResult(
            TotalTokens: tokens.Length,
            ScoredTokens: scored,
            GreenTokens: green,
            GreenFraction: scored > 0 ? (double)green / scored : 0.0,
            Expected: expected,
            StdDev: stdDev,
            ZScore: z,
            PValue: p,
            ZThreshold: zThreshold,
            Detected: scored > 0 && z > zThreshold);
    }

    /// <summary>Upper-tail probability of the standard normal distribution.</summary>
    private static double OneSidedPValue(double z) => 0.5 * Erfc(z / Math.Sqrt(2.0));

    /// <summary>
    /// Numerical Recipes' Chebyshev approximation of erfc; fractional error &lt; 1.2e-7,
    /// which is far more precision than a reported p-value needs.
    /// </summary>
    private static double Erfc(double x)
    {
        var z = Math.Abs(x);
        var t = 2.0 / (2.0 + z);
        var ty = 4.0 * t - 2.0;

        double[] cof =
        [
            -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
            -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
            -1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
            6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
            9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
            3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15
        ];

        var d = 0.0;
        var dd = 0.0;
        for (var j = cof.Length - 1; j > 0; j--)
        {
            var tmp = d;
            d = ty * d - dd + cof[j];
            dd = tmp;
        }

        var ans = t * Math.Exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
        return x >= 0.0 ? ans : 2.0 - ans;
    }
}
