param(
    [string]$OutputDirectory = "artifacts",
    [switch]$SkipArchive
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dataRoot = Join-Path $projectRoot "data"
$temporaryRoot = Join-Path $dataRoot "temp\build"
$workRoot = Join-Path $dataRoot "build\pyinstaller"
$distRoot = Join-Path $projectRoot $OutputDirectory

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

New-Item -ItemType Directory -Force -Path $temporaryRoot, $workRoot, $distRoot | Out-Null
$env:VSR_DATA_DIR = $dataRoot
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:TMPDIR = $temporaryRoot
$env:PIP_CACHE_DIR = Join-Path $dataRoot "cache\pip"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $dataRoot "cache\pyinstaller"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
$env:TORCH_HOME = Join-Path $dataRoot "cache\torch"
$env:TORCH_EXTENSIONS_DIR = Join-Path $dataRoot "cache\torch_extensions"
$env:PADDLE_HOME = Join-Path $dataRoot "cache\paddle"
$env:HF_HOME = Join-Path $dataRoot "cache\huggingface"

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --workpath $workRoot `
    --distpath $distRoot `
    (Join-Path $PSScriptRoot "vsr-full.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$releaseFolder = Get-ChildItem -LiteralPath $distRoot -Directory |
    Where-Object Name -Like "VideoSubtitleRemover-v*-win64-full" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $releaseFolder) {
    throw "Release folder was not created."
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "PORTABLE_README.txt") `
    -Destination (Join-Path $releaseFolder.FullName "PORTABLE_README.txt") `
    -Force

if (-not $SkipArchive) {
    $archive = "$($releaseFolder.FullName).zip"
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    & $python (Join-Path $PSScriptRoot "create_portable_archive.py") `
        $releaseFolder.FullName $archive
    if ($LASTEXITCODE -ne 0) {
        throw "Archive creation failed with exit code $LASTEXITCODE"
    }
}

Write-Output $releaseFolder.FullName
