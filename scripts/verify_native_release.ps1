[CmdletBinding()]
param(
    [string]$TorcsHome = 'C:\torcs\torcs',
    [int]$Steps = 1000
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $before = (python scripts\torcs_doctor.py --torcs-home $TorcsHome --json | ConvertFrom-Json).manifest
    python scripts\native_smoke.py --torcs-home $TorcsHome --steps $Steps --json
    $after = (python scripts\torcs_doctor.py --torcs-home $TorcsHome --json | ConvertFrom-Json).manifest

    $beforeFingerprint = $before | ConvertTo-Json -Depth 10 -Compress
    $afterFingerprint = $after | ConvertTo-Json -Depth 10 -Compress
    if ($beforeFingerprint -ne $afterFingerprint) {
        throw 'Installed TORCS manifest changed during the native smoke run'
    }
    Write-Output 'Native TORCS release gate passed; installed fingerprints are unchanged.'
}
finally {
    Pop-Location
}
