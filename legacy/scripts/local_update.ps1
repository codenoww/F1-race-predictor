# Generates the latest prediction on this machine (which can reach F1's live-timing
# data) and pushes it to GitHub; the website loads it from there at runtime.
# Run by the "F1PodiumPredictorUpdate" scheduled task. Log: update.log in the repo root.
$ErrorActionPreference = "Continue"   # PS 5.1 turns git's stderr progress into errors under Stop
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$log = Join-Path $repo "update.log"
$python = "C:\Python314\python.exe"

function Log($msg) { Add-Content -Path $log -Value ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg) }
function Run($label, [scriptblock]$cmd) {
    $out = & $cmd 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    foreach ($line in $out) { if ($line.Trim()) { Log "[$label] $line" } }
    return $code
}

Log "=== run started ==="
if ((Run "git pull" { git pull --rebase --autostash origin main }) -ne 0) { Log "git pull failed - aborting"; exit 1 }

$env:GIT_TERMINAL_PROMPT = "0"   # never hang waiting for a credential prompt in a hidden window
if ((Run "push check" { git push --dry-run origin main }) -ne 0) {
    Log "WARNING: cannot push to GitHub from this context - a new prediction would NOT be published"
} else { Log "push access OK" }

Push-Location (Join-Path $repo "webapp")
$code = Run "generate" { & $python -u generate_site.py }
Pop-Location
if ($code -ne 0) { Log "generate_site.py failed (exit $code) - aborting"; exit 1 }

git add public/ | Out-Null
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) { Log "no change to publish"; Log "=== run finished ==="; exit 0 }

$stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm 'UTC'")
if ((Run "git commit" { git commit -m "Update prediction (local run, $stamp)" }) -ne 0) { Log "commit failed - aborting"; exit 1 }
if ((Run "git push" { git push origin main }) -ne 0) { Log "push failed - will retry next run"; exit 1 }
Log "published new prediction"
Log "=== run finished ==="
