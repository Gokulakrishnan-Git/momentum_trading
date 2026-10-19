@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -c "import numpy, pandas, matplotlib, seaborn, requests, yfinance, pandas_market_calendars, streamlit" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -m streamlit run app.py
goto done

:failed
echo.
echo The backtest app could not start. Check the Python installation and internet connection.
pause
:done
endlocal
