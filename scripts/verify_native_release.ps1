[CmdletBinding()]
param(
    [string]$TorcsHome = 'C:\torcs\torcs',
    [int]$Steps = 500
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $pythonExe = "python"
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $pythonExe = ".\.venv\Scripts\python.exe"
    }

    Write-Host "Running TORCS installation doctor before test..."
    $beforeJson = & $pythonExe scripts\torcs_doctor.py --torcs-home $TorcsHome --json
    $before = ($beforeJson | ConvertFrom-Json).manifest

    Write-Host "Running native smoke test ($Steps steps)..."
    & $pythonExe scripts\native_smoke.py --torcs-home $TorcsHome --steps $Steps

    Write-Host "Running TORCS installation doctor after test..."
    $afterJson = & $pythonExe scripts\torcs_doctor.py --torcs-home $TorcsHome --json
    $after = ($afterJson | ConvertFrom-Json).manifest

    $beforeFingerprint = $before | ConvertTo-Json -Depth 10 -Compress
    $afterFingerprint = $after | ConvertTo-Json -Depth 10 -Compress
    if ($beforeFingerprint -ne $afterFingerprint) {
        throw 'Installed TORCS manifest changed during the native smoke run!'
    }
    Write-Output "Native TORCS release gate passed; installed simulator at $TorcsHome is completely unchanged."
}
finally {
    Pop-Location
}
