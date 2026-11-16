@echo off
setlocal
cd /d "%~dp0external\acados"
if errorlevel 1 exit /b 1
if not exist CMakeLists.txt (
    echo ERROR: Put this script in the Leap-C repository root.
    exit /b 1
)
set "ACADOS_FIX_ROOT=%CD:\=/%"
echo Configuring acados with VS 2022, x64 and OpenMP...
cmake -S . -B build_msvc_v2 -G "Visual Studio 17 2022" -A x64 -DCMAKE_C_FLAGS="/std:c11" -DBUILD_SHARED_LIBS=ON -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC -DACADOS_WITH_OPENMP=ON -DACADOS_NUM_THREADS=1 "-DACADOS_INSTALL_DIR=%ACADOS_FIX_ROOT%" "-DCMAKE_INSTALL_PREFIX=%ACADOS_FIX_ROOT%"
if errorlevel 1 goto configure_failed
echo Compiling acados. Output is saved to acados_build_v2_log.txt; please wait...
cmake --build build_msvc_v2 --config Release --parallel 4 > acados_build_v2_log.txt 2>&1
if errorlevel 1 goto build_failed
echo Installing the compiled DLLs and headers...
cmake --install build_msvc_v2 --config Release > acados_install_v2_log.txt 2>&1
if errorlevel 1 goto install_failed
if not exist bin\acados.dll goto missing_dll
if not exist include\acados_c\ocp_nlp_interface.h goto missing_header
echo PASS: acados build and installation completed.
echo Next: return to the repository root and run python smoke_test_leapc_windows.py
exit /b 0

:configure_failed
echo ERROR: CMake configuration failed. Send the terminal output above.
exit /b 1

:build_failed
echo ERROR: acados compilation failed. Send this log:
echo %CD%\acados_build_v2_log.txt
findstr /N /I /C:"error" /C:"fatal" acados_build_v2_log.txt
exit /b 1

:install_failed
echo ERROR: acados installation failed. Send this log:
echo %CD%\acados_install_v2_log.txt
type acados_install_v2_log.txt
exit /b 1

:missing_dll
echo ERROR: Expected bin\acados.dll is missing after installation.
dir /s /b acados.dll
exit /b 1

:missing_header
echo ERROR: Expected include\acados_c\ocp_nlp_interface.h is missing after installation.
exit /b 1
