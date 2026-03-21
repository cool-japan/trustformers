@echo off
REM TrustformeRS-C Windows DLL Builder
REM This script builds optimized DLLs for Windows platforms

echo Building TrustformeRS-C Windows DLL...

REM Configuration
set CRATE_NAME=trustformers_c
set TARGET_DIR=target\windows
set TARGETS=x86_64-pc-windows-msvc aarch64-pc-windows-msvc

REM Clean previous builds
echo Cleaning previous builds...
cargo clean

REM Create target directory
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

REM Build for x86_64 Windows
echo Building for x86_64-pc-windows-msvc...
cargo build --release --target x86_64-pc-windows-msvc --features "windows-dll"

REM Check if build succeeded
if not exist "target\x86_64-pc-windows-msvc\release\%CRATE_NAME%.dll" (
    echo Error: Build failed for x86_64-pc-windows-msvc
    exit /b 1
)

REM Build for ARM64 Windows (if supported)
echo Building for aarch64-pc-windows-msvc...
cargo build --release --target aarch64-pc-windows-msvc --features "windows-dll" 2>nul || (
    echo Warning: ARM64 build not supported or failed, continuing with x86_64 only
)

REM Copy built files
echo Copying built files...
copy "target\x86_64-pc-windows-msvc\release\%CRATE_NAME%.dll" "%TARGET_DIR%\%CRATE_NAME%-x64.dll"
copy "target\x86_64-pc-windows-msvc\release\%CRATE_NAME%.lib" "%TARGET_DIR%\%CRATE_NAME%-x64.lib"
copy "target\x86_64-pc-windows-msvc\release\%CRATE_NAME%.dll.lib" "%TARGET_DIR%\%CRATE_NAME%-x64.dll.lib" 2>nul

REM Copy ARM64 files if they exist
if exist "target\aarch64-pc-windows-msvc\release\%CRATE_NAME%.dll" (
    copy "target\aarch64-pc-windows-msvc\release\%CRATE_NAME%.dll" "%TARGET_DIR%\%CRATE_NAME%-arm64.dll"
    copy "target\aarch64-pc-windows-msvc\release\%CRATE_NAME%.lib" "%TARGET_DIR%\%CRATE_NAME%-arm64.lib"
    copy "target\aarch64-pc-windows-msvc\release\%CRATE_NAME%.dll.lib" "%TARGET_DIR%\%CRATE_NAME%-arm64.dll.lib" 2>nul
)

REM Copy header file
echo Copying header file...
copy "target\include\trustformers-c\trustformers.h" "%TARGET_DIR%\"

REM Generate .def file for explicit exports
echo Generating .def file...
echo EXPORTS > "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_version >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_create_pipeline >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_destroy_pipeline >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_load_model >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_create_tokenizer >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_encode >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_decode >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_generate >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_get_memory_usage >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_get_last_error >> "%TARGET_DIR%\%CRATE_NAME%.def"
echo trustformers_free_string >> "%TARGET_DIR%\%CRATE_NAME%.def"

REM Create NuGet package structure
echo Creating NuGet package structure...
set NUGET_DIR=%TARGET_DIR%\nuget
if not exist "%NUGET_DIR%" mkdir "%NUGET_DIR%"
if not exist "%NUGET_DIR%\lib" mkdir "%NUGET_DIR%\lib"
if not exist "%NUGET_DIR%\lib\net60" mkdir "%NUGET_DIR%\lib\net60"
if not exist "%NUGET_DIR%\runtimes\win-x64\native" mkdir "%NUGET_DIR%\runtimes\win-x64\native"
if not exist "%NUGET_DIR%\runtimes\win-arm64\native" mkdir "%NUGET_DIR%\runtimes\win-arm64\native"

REM Copy files to NuGet structure
copy "%TARGET_DIR%\%CRATE_NAME%-x64.dll" "%NUGET_DIR%\runtimes\win-x64\native\"
copy "%TARGET_DIR%\%CRATE_NAME%-x64.lib" "%NUGET_DIR%\runtimes\win-x64\native\"
copy "%TARGET_DIR%\trustformers.h" "%NUGET_DIR%\runtimes\win-x64\native\"

if exist "%TARGET_DIR%\%CRATE_NAME%-arm64.dll" (
    copy "%TARGET_DIR%\%CRATE_NAME%-arm64.dll" "%NUGET_DIR%\runtimes\win-arm64\native\"
    copy "%TARGET_DIR%\%CRATE_NAME%-arm64.lib" "%NUGET_DIR%\runtimes\win-arm64\native\"
    copy "%TARGET_DIR%\trustformers.h" "%NUGET_DIR%\runtimes\win-arm64\native\"
)

REM Create NuGet package specification
echo Creating NuGet package specification...
(
echo ^<?xml version="1.0" encoding="utf-8"?^>
echo ^<package^>
echo   ^<metadata^>
echo     ^<id^>TrustformeRS.Native^</id^>
echo     ^<version^>0.1.0^</version^>
echo     ^<title^>TrustformeRS Native Library^</title^>
echo     ^<authors^>Cool Japan^</authors^>
echo     ^<description^>Native TrustformeRS library for .NET applications^</description^>
echo     ^<projectUrl^>https://github.com/cool-japan/trustformers^</projectUrl^>
echo     ^<license type="expression"^>MIT^</license^>
echo     ^<requireLicenseAcceptance^>false^</requireLicenseAcceptance^>
echo     ^<tags^>transformers nlp rust native^</tags^>
echo   ^</metadata^>
echo   ^<files^>
echo     ^<file src="runtimes\**" target="runtimes" /^>
echo   ^</files^>
echo ^</package^>
) > "%NUGET_DIR%\TrustformeRS.Native.nuspec"

echo Windows DLL build completed!
echo Dynamic library (x64): %TARGET_DIR%\%CRATE_NAME%-x64.dll
echo Static library (x64): %TARGET_DIR%\%CRATE_NAME%-x64.lib
if exist "%TARGET_DIR%\%CRATE_NAME%-arm64.dll" (
    echo Dynamic library (ARM64): %TARGET_DIR%\%CRATE_NAME%-arm64.dll
    echo Static library (ARM64): %TARGET_DIR%\%CRATE_NAME%-arm64.lib
)
echo Header file: %TARGET_DIR%\trustformers.h
echo NuGet package: %NUGET_DIR%\

REM Optional: Create installer
if "%1"=="--installer" (
    echo Creating Windows installer...
    REM This would typically use WiX or similar tools
    echo Installer creation requires additional setup
)

echo Build process completed successfully!