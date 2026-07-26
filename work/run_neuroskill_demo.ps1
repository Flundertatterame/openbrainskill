param(
    [string]$Python = "python",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 18444,
    [double]$Timeout = 3,
    [string]$Token = "",
    [string]$RunDir = ""
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Client = Join-Path $PSScriptRoot "neuroskill_client.py"
$Converter = Join-Path $PSScriptRoot "convert_neuroskill_sleep_to_labels.py"
if (-not $RunDir) { $RunDir = Join-Path $ProjectRoot "outputs\runs\day12" }
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Common = @("--host", $HostName, "--port", $Port, "--timeout", $Timeout)
if ($Token) { $Common += @("--token", $Token) }

$Log = Join-Path $RunDir "run_neuroskill_demo.log"
"NeuroSkill demo started: $(Get-Date -Format o)" | Set-Content -Path $Log -Encoding UTF8

& $Python $Client status @Common --out (Join-Path $RunDir "neuroskill_status.json")
"status exit_code=$LASTEXITCODE" | Add-Content -Path $Log -Encoding UTF8

& $Python $Client lsl-discover @Common --out (Join-Path $RunDir "neuroskill_lsl_discover.json")
"lsl-discover exit_code=$LASTEXITCODE" | Add-Content -Path $Log -Encoding UTF8

$SleepJson = Join-Path $RunDir "neuroskill_sleep.json"
& $Python $Client sleep @Common --out $SleepJson
$SleepExit = $LASTEXITCODE
"sleep exit_code=$SleepExit" | Add-Content -Path $Log -Encoding UTF8

if ($SleepExit -eq 0) {
    & $Python $Converter --input $SleepJson --out (Join-Path $RunDir "pred_labels_from_neuroskill.csv") --error-out (Join-Path $RunDir "convert_error.json")
    "convert exit_code=$LASTEXITCODE" | Add-Content -Path $Log -Encoding UTF8
} else {
    "conversion skipped because NeuroSkill sleep failed" | Add-Content -Path $Log -Encoding UTF8
}

Write-Host "NeuroSkill demo outputs: $RunDir"
Write-Host "Log: $Log"
