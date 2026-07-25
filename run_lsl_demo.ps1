param(
    [Parameter(Mandatory = $true)]
    [string]$Edf,

    [string]$RunDir = "",
    [string]$Python = "",
    [double]$Minutes = 10,
    [double]$DiscoverSeconds = 5,
    [double]$StartupWaitSeconds = 3,
    [switch]$KeepStreaming
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $ProjectRoot "work"

if (-not (Test-Path -LiteralPath $Edf -PathType Leaf)) {
    throw "EDF file not found: $Edf"
}

if (-not $Python) {
    $BrainFusionPython = "C:\Users\shen\anaconda3\envs\brainfusion\python.exe"
    $Python = if (Test-Path -LiteralPath $BrainFusionPython) { $BrainFusionPython } else { "python" }
}

if (-not $RunDir) {
    $RunDir = Join-Path $ProjectRoot ("outputs\runs\lsl_demo_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$StreamScript = Join-Path $WorkDir "edf_to_lsl_stream.py"
$DiscoverScript = Join-Path $WorkDir "lsl_resolve_check.py"
$MetadataOut = Join-Path $RunDir "lsl_stream_metadata.json"
$StateOut = Join-Path $RunDir "lsl_stream_state.json"
$ErrorOut = Join-Path $RunDir "lsl_stream_error.json"
$DiscoverOut = Join-Path $RunDir "lsl_discover.json"
$StreamLog = Join-Path $RunDir "lsl_stream.log"
$StreamErrorLog = Join-Path $RunDir "lsl_stream.stderr.log"

$StreamArgs = @(
    "-u",
    $StreamScript,
    "--edf", $Edf,
    "--minutes", $Minutes,
    "--metadata-json-out", $MetadataOut,
    "--state-json-out", $StateOut,
    "--error-json-out", $ErrorOut
)

Write-Host "Starting LSL stream..."
$StreamProcess = Start-Process -FilePath $Python -ArgumentList $StreamArgs -RedirectStandardOutput $StreamLog -RedirectStandardError $StreamErrorLog -PassThru

try {
    Start-Sleep -Seconds $StartupWaitSeconds
    if ($StreamProcess.HasExited) {
        throw "LSL stream exited early with code $($StreamProcess.ExitCode). Review $StreamErrorLog and $ErrorOut."
    }

    Write-Host "Discovering EEG streams..."
    & $Python $DiscoverScript --seconds $DiscoverSeconds | Tee-Object -FilePath $DiscoverOut
    if ($LASTEXITCODE -ne 0) {
        throw "LSL discovery failed with code $LASTEXITCODE."
    }

    Write-Host "LSL demo artifacts written to: $RunDir"
    Write-Host "  metadata: $MetadataOut"
    Write-Host "  state:    $StateOut"
    Write-Host "  discover: $DiscoverOut"
    Write-Host "  log:      $StreamLog"

    if ($KeepStreaming) {
        Write-Host "The stream remains active (PID $($StreamProcess.Id)). Stop it when the consumer is finished."
        $StreamProcess = $null
    }
}
finally {
    if ($null -ne $StreamProcess -and -not $StreamProcess.HasExited) {
        Stop-Process -Id $StreamProcess.Id -ErrorAction SilentlyContinue
        $StreamProcess.WaitForExit()
        if (Test-Path -LiteralPath $StateOut) {
            $State = Get-Content -Raw -LiteralPath $StateOut | ConvertFrom-Json
            if ($State.status -eq "running") {
                $State.status = "stopped_by_demo"
                $State | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date).ToUniversalTime().ToString("o") -Force
                $State | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $StateOut -Encoding utf8
            }
        }
        Write-Host "Stopped LSL stream process."
    }
}
