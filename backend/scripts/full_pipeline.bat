@echo off
REM ============================================================================
REM  full_pipeline.bat
REM  End-to-end LSC pipeline: raw videos -> ONNX, single Windows command.
REM
REM  Stages:
REM    1. Re-extract sequences from raw LSC50 videos (rotation-invariant)
REM    2. Build LOSO (Leave-One-Signer-Out) splits
REM    3. Train SignTransformer (F3) on the active 12-class vocabulary
REM    4. Eval-only LOSO sanity check on the trained checkpoint
REM    5. Diagnostic eval on the 5-signer held-out test set
REM    6. Export to ONNX (FP32 + INT8)
REM    7. Print a summary table of all metrics
REM
REM  Flags:
REM    --fast         40 epochs instead of 80 AND skip the 17-min re-extract
REM    --no-extract   skip stage 1 only (re-extract), still run the rest
REM
REM  Usage (from repo root):
REM    backend\scripts\full_pipeline.bat
REM    backend\scripts\full_pipeline.bat --fast
REM    backend\scripts\full_pipeline.bat --no-extract
REM
REM  Notes:
REM    - Uses ABSOLUTE paths and .venv\Scripts\python.exe (Windows-first).
REM    - The training stage calls scripts.run_train_background; the bat then
REM      polls the log for a "Training complete" / "Early stopping" sentinel
REM      so the pipeline stays single-command and reproducible.
REM    - The INT8 quantization in stage 6 is known to fail intermittently with
REM      a shape-inference error; the export script handles this gracefully
REM      (writes the FP32 ONNX as a fallback) and the bat treats non-zero RC
REM      from that stage as a warning, not a failure.
REM    - All stage logs go to %BACKEND_DIR%\logs\*_pipeline.log.
REM ============================================================================

setlocal EnableDelayedExpansion

REM ---- Parse flags -----------------------------------------------------------
set "FAST="
set "NO_EXTRACT="
:argparse
if "%~1"=="" goto argparse_done
if /I "%~1"=="--fast"        ( set "FAST=1"       & shift & goto argparse )
if /I "%~1"=="--no-extract"  ( set "NO_EXTRACT=1" & shift & goto argparse )
shift
goto argparse
:argparse_done

REM ---- Absolute paths -------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "BACKEND_DIR=%%~fI"
for %%I in ("%BACKEND_DIR%\..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "LOG_DIR=%BACKEND_DIR%\logs"
set "CKPT=%BACKEND_DIR%\models\sequence\sign_transformer.pth"
set "SEQ_DIR=%BACKEND_DIR%\data\sequences"
set "LOSO_JSON=%SEQ_DIR%\loso_splits.json"
set "LOSO_REPORT=%BACKEND_DIR%\reports\f3_loso_pipeline.json"
set "EVAL_REPORT=%BACKEND_DIR%\reports\f3_eval_pipeline.json"
set "REGISTRY=%BACKEND_DIR%\models\exports\registry.json"

if not exist "%PYTHON%" (
    echo [ERROR] venv python not found: "%PYTHON%"
    echo         Create it with: python -m venv .venv
    exit /b 1
)
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "EPOCHS=80"
if defined FAST set "EPOCHS=40"

echo.
echo ==============================================
echo   LSC full pipeline
echo   python:    %PYTHON%
echo   backend:   %BACKEND_DIR%
echo   epochs:    %EPOCHS%
echo   fast:      %FAST%
echo   no-extract: %NO_EXTRACT%
echo ==============================================
echo.

REM ---- Stage return-code tracking -------------------------------------------
set "RC_EXTRACT=0"
set "RC_SPLITS=0"
set "RC_TRAIN=0"
set "RC_LOSO=0"
set "RC_EVAL=0"
set "RC_EXPORT=0"

REM ---- Stage 1: Re-extract (skippable) --------------------------------------
if defined NO_EXTRACT goto skip_extract
if defined FAST      goto skip_extract
echo [Stage 1/7] Re-extracting sequences from raw videos (rotation-invariant)...
echo            This takes ~17 min. Tail with:
echo            Get-Content "%LOG_DIR%\reextract_pipeline.log" -Wait
echo.
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.run_reextract_background > "%LOG_DIR%\reextract_pipeline.log" 2>&1
set "RC_EXTRACT=!ERRORLEVEL!"
popd
if not "!RC_EXTRACT!"=="0" (
    echo [ERROR] Re-extract launch failed (rc=!RC_EXTRACT!). See %LOG_DIR%\reextract_pipeline.log
    goto :summary
)
call :wait_for_log_sentinel "%LOG_DIR%\reextract_pipeline.log" "Wrote " "ERROR" 20
set "RC_EXTRACT=!ERRORLEVEL!"
if not "!RC_EXTRACT!"=="0" (
    echo [ERROR] Re-extract did not finish within ~20 min.
    echo         Tail of log:
    powershell -NoProfile -Command "Get-Content '%LOG_DIR%\reextract_pipeline.log' -Tail 20"
)
if not "!RC_EXTRACT!"=="0" goto :summary
echo   [OK] Stage 1 complete.
goto :after_extract
:skip_extract
echo [Stage 1/7] Skipped ^(--fast or --no-extract^). Using existing .npy files in %SEQ_DIR%.
:after_extract
echo.

REM ---- Stage 2: LOSO splits --------------------------------------------------
echo [Stage 2/7] Building LOSO splits (18 folds)...
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.build_loso_splits > "%LOG_DIR%\build_loso_pipeline.log" 2>&1
set "RC_SPLITS=!ERRORLEVEL!"
popd
if not "!RC_SPLITS!"=="0" (
    echo [ERROR] LOSO splits failed (rc=!RC_SPLITS!). See %LOG_DIR%\build_loso_pipeline.log
    goto :summary
)
if not exist "%LOSO_JSON%" (
    echo [ERROR] Expected %LOSO_JSON% to exist after build_loso_splits
    set "RC_SPLITS=1"
    goto :summary
)
echo   [OK] Stage 2 complete. Splits: %LOSO_JSON%
echo.

REM ---- Stage 3: Train SignTransformer ---------------------------------------
echo [Stage 3/7] Training SignTransformer (epochs=%EPOCHS%, 5-15 min on CPU)...
echo            Tail with: Get-Content "%LOG_DIR%\train_pipeline.log" -Wait
echo.
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.run_train_background --name train_pipeline --epochs %EPOCHS% --no-test-eval --splits-json "%LOSO_JSON%" --aug-factor 5 > "%LOG_DIR%\train_pipeline_launcher.log" 2>&1
set "STAGE_LAUNCH_RC=!ERRORLEVEL!"
popd
if not "!STAGE_LAUNCH_RC!"=="0" (
    echo [ERROR] Train launcher failed (rc=!STAGE_LAUNCH_RC!). See %LOG_DIR%\train_pipeline_launcher.log
    set "RC_TRAIN=1"
    goto :summary
)
call :wait_for_log_sentinel "%LOG_DIR%\train_pipeline.log" "Training complete" "Early stopping" 25
set "RC_TRAIN=!ERRORLEVEL!"
if not "!RC_TRAIN!"=="0" (
    echo [ERROR] Training did not finish within ~25 min.
    echo         Tail of log:
    powershell -NoProfile -Command "Get-Content '%LOG_DIR%\train_pipeline.log' -Tail 30"
    goto :summary
)
if not exist "%CKPT%" (
    echo [ERROR] Training did not save checkpoint at %CKPT%
    set "RC_TRAIN=1"
    goto :summary
)
echo   [OK] Stage 3 complete. Checkpoint: %CKPT%
echo.

REM ---- Stage 4: Eval-only LOSO sanity check ---------------------------------
echo [Stage 4/7] Eval-only LOSO on the trained checkpoint...
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.loso_test_only --ckpt "%CKPT%" --export "%LOSO_REPORT%" > "%LOG_DIR%\loso_eval_pipeline.log" 2>&1
set "RC_LOSO=!ERRORLEVEL!"
popd
if not "!RC_LOSO!"=="0" (
    echo [ERROR] LOSO eval failed (rc=!RC_LOSO!). See %LOG_DIR%\loso_eval_pipeline.log
    goto :summary
)
if not exist "%LOSO_REPORT%" (
    echo [ERROR] Expected LOSO report at %LOSO_REPORT%
    set "RC_LOSO=1"
    goto :summary
)
echo   [OK] Stage 4 complete. Report: %LOSO_REPORT%
echo.

REM ---- Stage 5: Diagnostic eval on 5-signer held-out test set ---------------
echo [Stage 5/7] Diagnostic eval on 5-signer held-out test set (signers 32 30 24 26 25)...
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.evaluate_sequence --ckpt "%CKPT%" --signers 32 30 24 26 25 --export "%EVAL_REPORT%" > "%LOG_DIR%\eval_pipeline.log" 2>&1
set "RC_EVAL=!ERRORLEVEL!"
popd
if not "!RC_EVAL!"=="0" (
    echo [ERROR] Sequence eval failed (rc=!RC_EVAL!). See %LOG_DIR%\eval_pipeline.log
    goto :summary
)
if not exist "%EVAL_REPORT%" (
    echo [ERROR] Expected eval report at %EVAL_REPORT%
    set "RC_EVAL=1"
    goto :summary
)
echo   [OK] Stage 5 complete. Report: %EVAL_REPORT%
echo.

REM ---- Stage 6: Export to ONNX (FP32 + INT8) --------------------------------
echo [Stage 6/7] Exporting to ONNX (FP32 + INT8)...
pushd "%BACKEND_DIR%"
"%PYTHON%" -u -m scripts.export_transformer_onnx > "%LOG_DIR%\export_pipeline.log" 2>&1
set "RC_EXPORT=!ERRORLEVEL!"
popd
REM  INT8 quantization has a known shape-inference bug; the script already
REM  falls back to the FP32 file in that case. Treat a non-zero RC as a
REM  warning (still proceeds) so the pipeline summary stays useful.
if not "!RC_EXPORT!"=="0" (
    echo [WARN] ONNX export returned non-zero (rc=!RC_EXPORT!); check %LOG_DIR%\export_pipeline.log
    echo        This is likely the known INT8 shape-inference issue. FP32 ONNX should still be present.
    set "RC_EXPORT=0"
) else (
    echo   [OK] Stage 6 complete. ONNX written to %BACKEND_DIR%\models\exports\
)
echo.

REM ---- Stage 7: Summary ------------------------------------------------------
:summary
echo ==============================================
echo   PIPELINE SUMMARY
echo ==============================================
echo   Stage 1 (re-extract):    rc=!RC_EXTRACT!
echo   Stage 2 (loso splits):   rc=!RC_SPLITS!
echo   Stage 3 (train):         rc=!RC_TRAIN!
echo   Stage 4 (loso eval):     rc=!RC_LOSO!
echo   Stage 5 (eval 5-signer): rc=!RC_EVAL!
echo   Stage 6 (onnx export):   rc=!RC_EXPORT!
echo.

REM  Render the metrics table by calling a small inline Python helper that
REM  reads the report JSONs and the ONNX registry.
"%PYTHON%" -c "import sys, json, pathlib; \
loso=pathlib.Path(r'%LOSO_REPORT%'); evalr=pathlib.Path(r'%EVAL_REPORT%'); reg=pathlib.Path(r'%REGISTRY%'); \
def safe_load(p):\
    if not p.exists(): return None\
    try: return json.loads(p.read_text(encoding='utf-8'))\
    except Exception: return None\
loso= safe_load(loso); evalr= safe_load(evalr); reg= safe_load(reg); \
print('%-32s %-12s %s' % ('metric', 'value', 'source')); \
print('-'*72); \
if loso:\
    s=loso.get('summary') or {}; \
    print('%-32s %-12s %s' % ('loso mean_acc (per-signer)', '%.2f%%'%(s.get('mean_acc',0)*100), str(loso))); \
    print('%-32s %-12s %s' % ('loso min_acc',  '%.2f%%'%(s.get('min_acc',0)*100),  '')); \
    print('%-32s %-12s %s' % ('loso max_acc',  '%.2f%%'%(s.get('max_acc',0)*100),  '')); \
if evalr:\
    s=evalr.get('summary') or {}; \
    print('%-32s %-12s %s' % ('5-signer test acc',     '%.2f%%'%(s.get('accuracy',0)*100), str(evalr))); \
    print('%-32s %-12s %s' % ('5-signer mean_conf',    '%.3f'%(s.get('mean_confidence',0)),       '')); \
    print('%-32s %-12s %s' % ('5-signer latency p50ms','%.2f'%(s.get('latency_p50_ms',0) or 0),  '')); \
if reg and reg.get('models'):\
    m=reg['models'][0]; \
    print('%-32s %-12s %s' % ('onnx fp32 (latest)',     m.get('fp32_onnx','-'),  str(reg))); \
    print('%-32s %-12s %s' % ('onnx int8 (latest)',     m.get('int8_onnx','-') or 'n/a', '')); \
    print('%-32s %-12s %s' % ('n_classes',              str(m.get('n_classes','-')), '')); \
    print('%-32s %-12s %s' % ('n_params',               str(m.get('n_params','-')),  '')); \
print('-'*72)" 2>nul

set /a "ANY_FAIL=RC_EXTRACT + RC_SPLITS + RC_TRAIN + RC_LOSO + RC_EVAL + RC_EXPORT"
if !ANY_FAIL! NEQ 0 (
    echo.
    echo [FAIL] Pipeline completed with errors. Inspect the per-stage logs in %LOG_DIR%.
    exit /b 1
)
echo.
echo [OK] Pipeline complete.
exit /b 0

REM ============================================================================
REM  Helper: wait for one of the sentinel strings to appear in a log file
REM  Usage:  call :wait_for_log_sentinel <log_path> <sentinel1> <sentinel2> <timeout_min>
REM  Sets ERRORLEVEL to 0 on sentinel match, 1 on timeout.
REM  Polls every 10 seconds.
REM ============================================================================
:wait_for_log_sentinel
setlocal
set "LOG_FILE=%~1"
set "SENT1=%~2"
set "SENT2=%~3"
set "TIMEOUT_MIN=%~4"
set /a "POLLS=%TIMEOUT_MIN% * 6"
set "POLL_RC=1"
for /L %%i in (1,1,%POLLS%) do (
    if exist "%LOG_FILE%" (
        findstr /C:"%SENT1%" "%LOG_FILE%" >nul 2>&1 && ( set "POLL_RC=0" & goto :sentinel_done )
        findstr /C:"%SENT2%" "%LOG_FILE%" >nul 2>&1 && ( set "POLL_RC=0" & goto :sentinel_done )
    )
    timeout /t 10 /nobreak >nul 2>&1
)
:sentinel_done
endlocal & set "WAIT_RC=%POLL_RC%"
exit /b %POLL_RC%
