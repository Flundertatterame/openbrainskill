param(
    [string]$Python = "python",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 18444,
    [double]$Timeout = 3,
    [string]$Token = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Client = Join-Path $PSScriptRoot "neuroskill_client.py"

if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "outputs\runs\day08"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Common = @("--host", $HostName, "--port", $Port, "--timeout", $Timeout)
if ($Token) {
    $Common += @("--token", $Token)
}

Write-Host "[1/3] NeuroSkill status"
& $Python $Client status @Common --out (Join-Path $OutDir "neuroskill_status.json")
$StatusExitCode = $LASTEXITCODE

Write-Host "[2/3] NeuroSkill LSL discover"
& $Python $Client lsl-discover @Common --out (Join-Path $OutDir "neuroskill_lsl_discover.json")
$DiscoverExitCode = $LASTEXITCODE

Write-Host "[3/3] NeuroSkill sleep"
& $Python $Client sleep @Common --out (Join-Path $OutDir "neuroskill_sleep.json")
$SleepExitCode = $LASTEXITCODE

Write-Host "Saved outputs to: $OutDir"
Write-Host "Exit codes: status=$StatusExitCode, lsl-discover=$DiscoverExitCode, sleep=$SleepExitCode"

if (($StatusExitCode -ne 0) -or ($DiscoverExitCode -ne 0) -or ($SleepExitCode -ne 0)) {
    exit 1
}
