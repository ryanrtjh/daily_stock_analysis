<#
Registers a daily Windows Task Scheduler job that rebuilds the dashboard after the US close.
US cash close is 16:00 ET = 05:00 Singapore (04:00 during US daylight time), so 06:00 local,
Tue-Sat, always lands after the close. Run once from PowerShell:

    powershell -ExecutionPolicy Bypass -File dashboard\schedule_windows.ps1            # install
    powershell -ExecutionPolicy Bypass -File dashboard\schedule_windows.ps1 -Remove    # uninstall

Pass -At "07:30" to change the time, -Notify to also push the digest to your .env channels.
"Run as soon as possible after a missed start" is on, so a sleeping PC catches up when it wakes.
#>
param([string]$At = "06:00", [switch]$Notify, [switch]$Remove)
$name = "US Watchlist Dashboard"
if ($Remove) { Unregister-ScheduledTask -TaskName $name -Confirm:$false; "Removed '$name'."; return }

$repo = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Venv python not found at $py" }
$args = "dashboard\build.py" + $(if ($Notify) { " --notify" } else { "" })
$action   = New-ScheduledTaskAction -Execute $py -Argument $args -WorkingDirectory $repo
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday,Wednesday,Thursday,Friday,Saturday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1) -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
"Registered '$name': Tue-Sat at $At local. Output: $repo\dashboard\site\index.html"
