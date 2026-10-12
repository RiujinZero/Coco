# Build the native pipeline with Visual Studio 2022 Build Tools (MSVC, /O2 /arch:AVX2 /GL).
#   powershell -ExecutionPolicy Bypass -File scripts/build_msvc.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
$build = Join-Path $root "build-msvc"
$cmd = "call `"$vcvars`" >nul && cmake -S `"$root\cpp`" -B `"$build`" -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release && cmake --build `"$build`" && `"$build\test_kernels.exe`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
