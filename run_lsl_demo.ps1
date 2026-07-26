param(
    [Parameter(Mandatory = $true)]
    [string]$Edf,

    [string]$RunDir = "",
    [string]$Python = "",
    [double]$Minutes = 10,
    [double]$DiscoverSeconds = 5,
    [double]$StartupWaitSeconds = 3,
    [double]$StartupTimeoutSeconds = 60,
    [switch]$KeepStreaming
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Join-Path $ProjectRoot "work"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory = $true)]$Value)

    if ($Value -is [System.IFormattable]) {
        $Text = $Value.ToString($null, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    else {
        $Text = [string]$Value
    }
    if ($Text.Contains('"')) {
        throw "Native process arguments cannot contain a double quote: $Text"
    }
    return '"' + $Text + '"'
}

if (-not (Test-Path -LiteralPath $Edf -PathType Leaf)) {
    throw "EDF file not found: $Edf"
}

if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python was not found. Activate an environment with mne and pylsl, or pass -Python <path>."
    }
    $Python = $PythonCommand.Source
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
$StreamArgumentLine = ($StreamArgs | ForEach-Object { ConvertTo-NativeArgument $_ }) -join " "

Write-Host "Starting LSL stream..."
$StreamProcess = Start-Process -FilePath $Python -ArgumentList $StreamArgumentLine -RedirectStandardOutput $StreamLog -RedirectStandardError $StreamErrorLog -PassThru

try {
    Start-Sleep -Seconds $StartupWaitSeconds

    $StartupDeadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    while (-not (Test-Path -LiteralPath $MetadataOut -PathType Leaf)) {
        if ($StreamProcess.HasExited) {
            $StreamProcess.WaitForExit()
            throw "LSL stream exited early with code $($StreamProcess.ExitCode). Review $StreamErrorLog and $ErrorOut."
        }
        if ((Get-Date) -ge $StartupDeadline) {
            throw "Timed out after $StartupTimeoutSeconds seconds waiting for LSL stream metadata. Review $StreamLog and $StreamErrorLog."
        }
        Start-Sleep -Milliseconds 500
    }

    Write-Host "Discovering EEG streams..."
    & $Python $DiscoverScript --seconds $DiscoverSeconds --out $DiscoverOut
    if ($LASTEXITCODE -ne 0) {
        throw "LSL discovery failed with code $LASTEXITCODE."
    }
    $Metadata = [System.IO.File]::ReadAllText($MetadataOut, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $DiscoverResult = [System.IO.File]::ReadAllText($DiscoverOut, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $MatchingStreams = @($DiscoverResult.streams | Where-Object {
        $_.source_id -eq $Metadata.source_id -and
        $_.name -eq $Metadata.name -and
        [int]$_.channel_count -eq [int]$Metadata.channel_count -and
        [Math]::Abs([double]$_.sample_rate - [double]$Metadata.nominal_srate) -lt 0.001
    })
    if ($MatchingStreams.Count -lt 1) {
        throw "The expected EEG LSL stream was not discovered (source_id=$($Metadata.source_id)). Review $StreamLog and $StreamErrorLog."
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
            $State = [System.IO.File]::ReadAllText($StateOut, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
            if ($State.status -eq "running") {
                $State.status = "stopped_by_demo"
                $State | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date).ToUniversalTime().ToString("o") -Force
                $RenderedState = $State | ConvertTo-Json -Depth 10
                [System.IO.File]::WriteAllText($StateOut, $RenderedState + [Environment]::NewLine, $Utf8NoBom)
            }
        }
        Write-Host "Stopped LSL stream process."
    }
}
