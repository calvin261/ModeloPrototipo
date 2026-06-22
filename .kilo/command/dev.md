---
description: Arranca backend FastAPI + rebuild + instala/lanza APK en la tablet (modo dev)
---

Arranca el entorno de desarrollo LSEC completo (backend + Android). Trabaja en `C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo`. Ejecuta en orden y NO cierres el backend al final.

## Pasos

1. **Detectar IP local** (WiFi primero):
   ```
   ipconfig | Select-String "IPv4"
   ```
   Preferir la IPv4 tipo `192.168.x.x` (WiFi del usuario); usar `172.17.x.x` solo si no hay WiFi. Guardarla como `$IP`.

2. **Matar backend previo en :8000** (si existe):
   ```
   Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
   Get-Process -Name uvicorn,python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*uvicorn*' } | Stop-Process -Force -ErrorAction SilentlyContinue
   ```

3. **Arrancar backend en background** desde `backend\`:
   ```
   cd backend; .\start_backend.bat
   ```
   Si no hay `.venv` ni `venv`, avisar y sugerir `python -m venv .venv && .venv\Scripts\activate && pip install -r api/requirements.txt`. No fallar el comando, solo advertir.
   Esperar hasta 10s a que `http://localhost:8000/health` responda 200 (reintentar con `Invoke-WebRequest -UseBasicParsing`).

4. **Actualizar `serverUrl` por defecto** en el código Android a `http://$IP:8000/` en 3 sitios (usar `edit`):
   - `frontend/app/src/main/java/com/lsec/translator/data/preferences/UserPreferences.kt` — data class `serverUrl` y fallback en `preferencesFlow.map { ... serverUrl = preferences[...SERVER_URL] ?: "..." }`.
   - `frontend/app/src/main/java/com/lsec\translator/ui/screens/settings/SettingsViewModel.kt` — default UI `serverUrl`.
   Buscar con `grep` antes de editar para confirmar líneas exactas. Si ya coincide con `$IP`, no tocar.

5. **Recompilar APK**:
   ```
   cd frontend; .\gradlew.bat assembleDebug --no-daemon
   ```
   Timeout 10 min. Si falla, mostrar el error y NO continuar con install.

6. **Reinstalar y lanzar** en la tablet conectada:
   ```
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" uninstall com.lsec.translator
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r "C:\Users\jeffe\Documents\Proyectos\ModeloPrototipo\frontend\app\build\outputs\apk\debug\app-debug.apk"
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" shell monkey -p com.lsec.translator -c android.intent.category.LAUNCHER 1
   Start-Sleep 3
   & "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" shell "dumpsys window | grep mCurrentFocus"
   ```
   Si no hay device (`adb devices` vacío), avisar y omitir install/launch, pero dejar backend corriendo.

7. **Resumen final** (NO terminar hasta ver "BUILD SUCCESSFUL" + backend health 200):
   - IP detectada y URL backend
   - Health check status
   - APK path y tamaño
   - ADB device + focused activity
   - Caveat: si la tablet cambia de WiFi, la IP cambia → repetir `/dev`.

## Importante
- NO cierres procesos al terminar; el backend debe seguir corriendo.
- El APK se reinstala siempre (uninstall + install) para que el nuevo default de DataStore tome efecto.
- Si `$IP` no se detecta, usar `127.0.0.1` y advertir que la tablet no podrá llegar al backend.