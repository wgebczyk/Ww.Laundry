using WatermarkTool.Cli;

try
{
    return await new WatermarkRootCommand().Parse(args).InvokeAsync();
}
catch (Exception ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}
