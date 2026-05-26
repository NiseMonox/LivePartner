# cleanup.ps1 -- kill LivePartner-spawned Python servers (TTS server, UI)
# without touching unrelated Python projects or the cleanup script itself.

$ErrorActionPreference = "SilentlyContinue"
$repo = "D:\LivePartner"
$self = $PID

Write-Host ""
Write-Host "==== LivePartner cleanup ====" -ForegroundColor Cyan
Write-Host "Scanning Python processes under $repo (excluding self PID $self) ..."
Write-Host ""

$found = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $self -and
    ($_.Name -eq "python.exe" -or $_.Name -eq "pythonw.exe") -and
    (
        ($_.ExecutablePath -and $_.ExecutablePath -like "$repo*") -or
        ($_.CommandLine -and $_.CommandLine -like "*$repo*")
    )
})

if ($found.Count -eq 0) {
    Write-Host "No LivePartner Python processes running." -ForegroundColor Green
} else {
    foreach ($p in $found) {
        $cmd = if ($p.CommandLine) { $p.CommandLine } else { "(no cmdline)" }
        if ($cmd.Length -gt 100) { $cmd = $cmd.Substring(0, 100) + "..." }
        $ramHint = if ($p.WorkingSetSize -gt 500MB) { " (RAM ~$([math]::Round($p.WorkingSetSize/1MB)) MB)" } else { "" }
        Write-Host ("  kill PID {0} {1}{2}" -f $p.ProcessId, $p.Name, $ramHint) -ForegroundColor Yellow
        Write-Host ("       cmd: {0}" -f $cmd) -ForegroundColor DarkGray
        Stop-Process -Id $p.ProcessId -Force
    }
    Write-Host ""
    Write-Host ("Killed {0} LivePartner Python process(es)." -f $found.Count) -ForegroundColor Green
}

# Mumble server only killed if user passes -mumble: cleanup.bat -mumble
if ($args -contains "-mumble") {
    Write-Host ""
    Write-Host "Also killing Mumble server (-mumble flag)..." -ForegroundColor Cyan
    $mumble = @(Get-Process mumble-server -ErrorAction SilentlyContinue)
    foreach ($p in $mumble) {
        Write-Host ("  kill PID {0} mumble-server.exe" -f $p.Id) -ForegroundColor Yellow
        Stop-Process -Id $p.Id -Force
    }
    if ($mumble.Count -eq 0) {
        Write-Host "  No mumble-server running." -ForegroundColor Green
    }
}

# Quick GPU summary if nvidia-smi exists on PATH.
$nv = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nv) {
    Write-Host ""
    Write-Host "==== Current VRAM ====" -ForegroundColor Cyan
    & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}