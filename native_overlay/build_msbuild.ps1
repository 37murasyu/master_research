param(
  [ValidateSet('Debug','Release')]
  [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Join-Path $scriptDir 'overlay.vcxproj'

if (!(Test-Path $proj)) {
  Write-Error "overlay.vcxproj not found."
}

# Try to use VSWhere to find MSBuild
$vswhere = "$Env:ProgramFiles(x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) { $vswhere = "$Env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe" }

$msbuild = $null
if (Test-Path $vswhere) {
  $msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe | Select-Object -First 1
}

if (-not $msbuild) {
  # Fallback to PATH
  $msbuild = (Get-Command msbuild.exe -ErrorAction SilentlyContinue).Source
}

# Try common Visual Studio locations if still not found
if (-not $msbuild) {
  $commonPaths = @(
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\MSBuild\\Current\\Bin\\MSBuild.exe',
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe',
    'C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\MSBuild\\Current\\Bin\\MSBuild.exe',
    'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\MSBuild\\Current\\Bin\\MSBuild.exe',
    'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\MSBuild\\Current\\Bin\\MSBuild.exe',
    'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\MSBuild\\Current\\Bin\\MSBuild.exe'
  )
  foreach ($p in $commonPaths) {
    if (Test-Path $p) { $msbuild = $p; break }
  }
}

if (-not $msbuild) {
  Write-Error "MSBuild not found. Please install Visual Studio Build Tools (C++) or add MSBuild to PATH."
}

# Build
& "$msbuild" $proj /p:Configuration=$Configuration /p:Platform=x64

Write-Host "Build done. Output in x64/$Configuration/overlay.dll" -ForegroundColor Green
