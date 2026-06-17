"""Script alternativo para iniciar el backend LSEC si uvicorn no esta en PATH."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 50)
print("  LSEC Backend - FastAPI en puerto 8000")
print("=" * 50)
print()
print("Iniciando servidor en http://0.0.0.0:8000")
print("Health check: http://localhost:8000/health")
print()
print("Para la app movil (dispositivo fisico):")
print("  1. Ejecuta 'ipconfig' y busca tu IP WiFi")
print("  2. En la app: Configuracion > URL del servidor")
print("  3. Ingresa: http://<TU-IP>:8000/")
print()
print("Presiona Ctrl+C para detener el servidor.")
print("=" * 50)
print()

subprocess.run(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ],
    cwd=PROJECT_ROOT,
)
