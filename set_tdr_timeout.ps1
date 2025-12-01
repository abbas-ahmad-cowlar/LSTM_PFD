# Increase TDR Timeout for GPU operations
# Run this as Administrator!

$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"

# TdrDelay = 60 seconds (default is 2)
New-ItemProperty -Path $regPath -Name "TdrDelay" -Value 60 -PropertyType DWORD -Force

# TdrDdiDelay = 60 seconds 
New-ItemProperty -Path $regPath -Name "TdrDdiDelay" -Value 60 -PropertyType DWORD -Force

Write-Host "TDR timeout increased to 60 seconds"
Write-Host "RESTART YOUR COMPUTER for changes to take effect!"
