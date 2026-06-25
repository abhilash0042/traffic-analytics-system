@echo off
cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
echo [1/3] Restoring pruned train files...
venv\Scripts\python.exe -u scripts\restore_pruned_helmet.py
echo.
echo [2/3] Syncing image/label pairs...
venv\Scripts\python.exe -u scripts\sync_helmet_train.py
echo.
echo [3/3] Balancing helmet train split (target 1.1:1)...
venv\Scripts\python.exe -u scripts\balance_helmet_dataset.py --status-file balance_done.txt
echo.
if exist balance_done.txt (
  echo === BALANCE COMPLETE ===
  type balance_done.txt
) else (
  echo Balance did not finish — check balance_out.txt
)
