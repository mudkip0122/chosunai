param(
    [string]$DataDir = "data"
)

$ErrorActionPreference = "Stop"

$archives = @(
    @{ Zip = "train_audio.zip"; Dest = "train_audio" },
    @{ Zip = "test_audio.zip"; Dest = "test_audio" }
)

foreach ($item in $archives) {
    $zipPath = Join-Path $DataDir $item.Zip
    $destPath = Join-Path $DataDir $item.Dest

    if (Test-Path $zipPath) {
        New-Item -ItemType Directory -Force -Path $destPath | Out-Null
        Expand-Archive -Path $zipPath -DestinationPath $destPath -Force
        Write-Host "Extracted $zipPath to $destPath"
    } else {
        Write-Host "Skipped missing $zipPath"
    }
}
