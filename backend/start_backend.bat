@echo off
echo ==============================================
echo   LSEC Backend - FastAPI en puerto 8000
echo ==============================================
echo.

cd /d "%~dp0"

REM Activar entorno virtual (.venv tiene prioridad sobre venv)
if exist ".venv\Scripts\activate.bat" (
    echo Activando entorno virtual .venv ...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activando entorno virtual venv ...
    call venv\Scripts\activate.bat
) else (
    echo AVISO: No se encontro entorno virtual.
    echo Crea uno con: python -m venv .venv
    echo.
)

echo Iniciando servidor en http://0.0.0.0:8000
echo Para la app movil: http://0.0.0.1:8000/
echo Ejecuta "ipconfig" para ver tu IP WiFi.
echo.
echo Presiona Ctrl+C para detener.
echo ==============================================
echo.

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
