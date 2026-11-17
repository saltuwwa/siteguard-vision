# Read-only hardware/event collection. No stress test or system settings changes.
param([int]$Days = 3)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$output = Join-Path $projectRoot ('runs/reboot-diagnostics-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $output | Out-Null
$status = [ordered]@{ CollectedAt = (Get-Date).ToString('o'); StressTest = $false }
try {
    Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption,Version,LastBootUpTime |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'system.json') -Encoding UTF8
    $status.System = 'collected'
} catch { $status.System = $_.Exception.Message }
try {
    Get-WinEvent -FilterHashtable @{
        LogName='System'; StartTime=(Get-Date).AddDays(-$Days)
        Id=41,6008,1001,4101,17,18,19,20,46,47,1074
    } -MaxEvents 120 |
        Select-Object TimeCreated,Id,ProviderName,Message,@{Name='Xml';Expression={$_.ToXml()}} |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $output 'events.json') -Encoding UTF8
    $status.Events = 'collected'
} catch { $status.Events = $_.Exception.Message }
try {
    & nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv |
        Set-Content -LiteralPath (Join-Path $output 'gpu.csv') -Encoding UTF8
    $status.NvidiaExitCode = $LASTEXITCODE
} catch { $status.Nvidia = $_.Exception.Message }
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'collection-status.json') -Encoding UTF8
Write-Output "Read-only diagnostic report: $output"
Write-Output 'Files stay in gitignored runs/. No GPU training, power-limit or driver changes were performed.'
