param(
  [string]$BuildType = "Release",
  [string]$Generator = "Ninja"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $scriptDir "build"

if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

Push-Location $buildDir

# Configure
cmake -G $Generator -DCMAKE_BUILD_TYPE=$BuildType ..

# Build
cmake --build . --config $BuildType

Pop-Location

Write-Host "Build completed. DLL should be in: $buildDir" -ForegroundColor Green
