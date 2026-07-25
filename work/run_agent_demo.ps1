param(
    # Input: Sleep-EDF PSG containing the continuous EEG/PSG signal.
    [string]$Psg,

    # Input: matching Hypnogram EDF providing 30-second ground-truth labels.
    [string]$Hypnogram,

    # Input: Cycle 3 defaults to the fully offline MNE baseline.
    [ValidateSet("mne_baseline", "neuroskill_lsl")]
    [string]$Backend = "mne_baseline",

    # Output: directory for all JSON, CSV, and Markdown artifacts in this run.
    [string]$RunDir,

    # Python executable used by the Agent and every tool subprocess.
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$sampleRoot = Join-Path $projectRoot "data\raw\sleep-edf\sleep-cassette"

# With no path arguments, use the repository's SC4002E0 sample.
if (-not $Psg) {
    $Psg = Join-Path $sampleRoot "SC4002E0-PSG.edf"
}
if (-not $Hypnogram) {
    $Hypnogram = Join-Path $sampleRoot "SC4002EC-Hypnogram.edf"
}
if (-not $RunDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunDir = Join-Path $projectRoot "outputs\runs\run_cycle3_$timestamp"
}

$agent = Join-Path $PSScriptRoot "sleep_staging_agent.py"
$request = "Run sleep staging for SC4002E0 and generate a report"

Write-Host "Backend:   $Backend"
Write-Host "PSG:       $Psg"
Write-Host "Hypnogram: $Hypnogram"
Write-Host "RunDir:    $RunDir"

# Output chain: edf_check.json -> true_labels.csv -> features.csv -> pred_labels.csv
# -> aligned_predictions.csv -> metrics.json -> final_report.md.
& $Python $agent `
    --request $request `
    --psg $Psg `
    --hypnogram $Hypnogram `
    --backend $Backend `
    --run-dir $RunDir `
    --python $Python `
    --execute

if ($LASTEXITCODE -ne 0) {
    throw "SleepStagingAgent exited with code $LASTEXITCODE"
}

Write-Host "Cycle 3 run completed. Artifacts: $RunDir"
