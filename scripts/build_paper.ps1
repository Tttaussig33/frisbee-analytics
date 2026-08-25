$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$paperDir = Join-Path $repoRoot "paper"
$buildDir = Join-Path $paperDir "build"
$document = "ufa_team_possession_patterns"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
Get-ChildItem -LiteralPath $buildDir -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

# Remove files from older direct builds so TeX cannot read stale root-level aux data.
@("aux", "bbl", "blg", "log", "out", "pdf", "synctex.gz") | ForEach-Object {
    $stalePath = Join-Path $paperDir ("{0}.{1}" -f $document, $_)
    if (Test-Path -LiteralPath $stalePath) {
        Remove-Item -LiteralPath $stalePath -Force
    }
}

function Invoke-BuildTool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Name @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $paperDir
try {
    Invoke-BuildTool "pdflatex" @(
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory=build",
        "$document.tex"
    )
    Invoke-BuildTool "bibtex" @("build/$document")
    Invoke-BuildTool "pdflatex" @(
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory=build",
        "$document.tex"
    )
    Invoke-BuildTool "pdflatex" @(
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory=build",
        "$document.tex"
    )
}
finally {
    Pop-Location
}

Write-Host "Built $buildDir\$document.pdf"
