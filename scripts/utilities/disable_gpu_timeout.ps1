# ============================================================================
# Disable Windows TDR (Timeout Detection and Recovery) for GPU
# ============================================================================
# Purpose: Prevent Windows from killing long-running CUDA operations
# 
# WARNING: This modifies the Windows Registry. Use with caution.
#          Disabling TDR means if your GPU truly hangs, Windows won't recover.
#
# Usage:   Run as Administrator:
#          .\scripts\utilities\disable_gpu_timeout.ps1
#
# To Revert: Run with -Revert flag:
#          .\scripts\utilities\disable_gpu_timeout.ps1 -Revert
# ============================================================================

param(
    [switch]$Revert = $false
)

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "  1. Right-click PowerShell" -ForegroundColor White
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "  3. Navigate to: $PSScriptRoot" -ForegroundColor White
    Write-Host "  4. Run: .\disable_gpu_timeout.ps1" -ForegroundColor White
    exit 1
}

# Registry path for GPU driver settings
$registryPath = "HKLM:\System\CurrentControlSet\Control\GraphicsDrivers"

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Windows TDR (Timeout Detection and Recovery) Configuration" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

if ($Revert) {
    Write-Host "REVERTING to default Windows TDR settings..." -ForegroundColor Yellow
    Write-Host ""
    
    # Remove custom TDR values (Windows will use defaults)
    $valuesToRemove = @("TdrDelay", "TdrDdiDelay", "TdrLevel")
    
    foreach ($valueName in $valuesToRemove) {
        try {
            $exists = Get-ItemProperty -Path $registryPath -Name $valueName -ErrorAction SilentlyContinue
            if ($exists) {
                Remove-ItemProperty -Path $registryPath -Name $valueName -Force
                Write-Host "✓ Removed: $valueName" -ForegroundColor Green
            } else {
                Write-Host "  (not set): $valueName" -ForegroundColor Gray
            }
        } catch {
            Write-Host "✗ Failed to remove: $valueName" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "TDR settings reverted to Windows defaults" -ForegroundColor Green
    Write-Host "Default behavior: 2-second GPU timeout with recovery" -ForegroundColor White
    
} else {
    Write-Host "DISABLING Windows TDR timeout for deep learning..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "This will allow CUDA operations to run without timeout." -ForegroundColor White
    Write-Host ""
    
    # Create registry values
    try {
        # TdrDelay: Seconds before timeout (60 seconds)
        Set-ItemProperty -Path $registryPath -Name "TdrDelay" -Value 60 -Type DWord -Force
        Write-Host "✓ Set TdrDelay = 60 seconds" -ForegroundColor Green
        
        # TdrDdiDelay: DDI timeout (60 seconds)
        Set-ItemProperty -Path $registryPath -Name "TdrDdiDelay" -Value 60 -Type DWord -Force
        Write-Host "✓ Set TdrDdiDelay = 60 seconds" -ForegroundColor Green
        
        # TdrLevel: 0 = Disable TDR recovery
        Set-ItemProperty -Path $registryPath -Name "TdrLevel" -Value 0 -Type DWord -Force
        Write-Host "✓ Set TdrLevel = 0 (TDR disabled)" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "TDR timeout successfully disabled!" -ForegroundColor Green
        
    } catch {
        Write-Host ""
        Write-Host "ERROR: Failed to modify registry" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "IMPORTANT: You must RESTART your computer for changes to take effect" -ForegroundColor Yellow
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to restart now
$restart = Read-Host "Restart computer now? (y/n)"
if ($restart -eq 'y' -or $restart -eq 'Y') {
    Write-Host ""
    Write-Host "Restarting in 10 seconds... (Press Ctrl+C to cancel)" -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    Restart-Computer -Force
} else {
    Write-Host ""
    Write-Host "Please restart your computer manually when ready." -ForegroundColor Yellow
    Write-Host ""
}
