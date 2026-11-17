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
function Export-EventSubset {
    param([string]$Name, [hashtable]$Filter, [int]$Limit)
    try {
        $items = @(Get-WinEvent -FilterHashtable $Filter -MaxEvents $Limit)
        $records = @($items | Select-Object TimeCreated,Id,ProviderName,Message,@{Name='Xml';Expression={$_.ToXml()}})
        ConvertTo-Json -InputObject $records -Depth 4 |
            Set-Content -LiteralPath (Join-Path $output ($Name + '.json')) -Encoding UTF8
        $status[$Name] = "collected $($items.Count) events; maximum $Limit"
    } catch {
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {
            '[]' | Set-Content -LiteralPath (Join-Path $output ($Name + '.json')) -Encoding UTF8
            $status[$Name] = 'no matching events'
        } else { $status[$Name] = $_.Exception.Message }
    }
}
# A corrected-error storm must not evict reboot/bugcheck records from the sample.
Export-EventSubset -Name 'critical-events' -Limit 100 -Filter @{
    LogName='System'; StartTime=(Get-Date).AddDays(-$Days)
    Id=41,6008,1001,4101,46,1074,6005,6006
}
Export-EventSubset -Name 'whea-important' -Limit 100 -Filter @{
    LogName='System'; StartTime=(Get-Date).AddDays(-$Days)
    ProviderName='Microsoft-Windows-WHEA-Logger'; Id=18,19,20,46,47
}
Export-EventSubset -Name 'whea-corrected-sample' -Limit 30 -Filter @{
    LogName='System'; StartTime=(Get-Date).AddDays(-$Days)
    ProviderName='Microsoft-Windows-WHEA-Logger'; Id=17
}
try {
    & pnputil.exe /enum-devices /class Display /connected /properties /relations |
        Set-Content -LiteralPath (Join-Path $output 'display-topology.txt') -Encoding UTF8
    $status.PnpExitCode = $LASTEXITCODE
} catch { $status.Pnp = $_.Exception.Message }
try {
    & nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv |
        Set-Content -LiteralPath (Join-Path $output 'gpu.csv') -Encoding UTF8
    $status.NvidiaExitCode = $LASTEXITCODE
} catch { $status.Nvidia = $_.Exception.Message }
$status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'collection-status.json') -Encoding UTF8
Write-Output "Read-only diagnostic report: $output"
Write-Output 'Files stay in gitignored runs/. No GPU training, power-limit or driver changes were performed.'
