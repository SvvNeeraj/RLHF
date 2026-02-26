$root = (Resolve-Path "$PSScriptRoot\..").Path
$env:PYTHONPATH = $root
Set-Location $root

# Prefer project venv python when available.
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$pythonCmd = if (Test-Path $venvPython) { $venvPython } else { "python" }

# Ensure old API process on port 8000 is stopped so latest code is always served.
$existing = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    foreach ($conn in $existing) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

& $pythonCmd -m uvicorn app.main:app --host 127.0.0.1 --port 8000
