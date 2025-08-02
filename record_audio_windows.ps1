# PowerShell script to record audio on Windows
# Run this in Windows PowerShell, not WSL

param(
    [int]$Duration = 10,
    [string]$OutputFile = "recording.wav"
)

Write-Host "Recording for $Duration seconds..."
Write-Host "Speak into your microphone now!"

# Create a simple recording using Windows Sound Recorder
$wshell = New-Object -ComObject wscript.shell
$wshell.Run("SoundRecorder /FILE $OutputFile /DURATION 0000:00:$Duration")

Write-Host "Recording saved to: $OutputFile"
Write-Host "Copy to WSL with: cp $OutputFile \\wsl.localhost\Ubuntu-22.04\home\mark\python-ble-audio\samples\"