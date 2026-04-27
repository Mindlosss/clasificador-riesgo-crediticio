@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY_CMD="

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  for %%V in (3.12 3.11 3.10) do (
    if not defined PY_CMD (
      py -%%V -c "import sys; raise SystemExit(0)" >nul 2>nul
      if !ERRORLEVEL! EQU 0 set "PY_CMD=py -%%V"
    )
  )
)

if not defined PY_CMD (
  python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
  if %ERRORLEVEL% EQU 0 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo.
  echo No se encontro una version compatible de Python.
  echo Este proyecto debe ejecutarse con Python 3.10, 3.11 o 3.12.
  echo Python 3.14 todavia causa problemas con pywebview/pythonnet en Windows.
  echo.
  echo Instala Python 3.11 desde https://www.python.org/downloads/release/python-3119/
  echo Marca la opcion "Add python.exe to PATH" durante la instalacion.
  echo Despues borra la carpeta .venv y vuelve a ejecutar iniciar_app.bat.
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual con %PY_CMD%...
  %PY_CMD% -m venv .venv
)

.venv\Scripts\python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo La carpeta .venv fue creada con una version incompatible de Python.
  echo Borra la carpeta .venv y vuelve a ejecutar iniciar_app.bat con Python 3.10, 3.11 o 3.12 instalado.
  echo.
  pause
  exit /b 1
)

.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo No se pudieron instalar las dependencias.
  echo Si venias de Python 3.14, borra .venv e instala Python 3.11.
  echo.
  pause
  exit /b 1
)

.venv\Scripts\python app.py
