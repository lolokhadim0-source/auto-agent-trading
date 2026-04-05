@echo off
echo Starting Trading Agent - %date% %time%
cd /d "C:\Users\khadi\auto agent self improving bot"
C:\Users\khadi\.local\bin\uv.exe run python run_agent.py -n 50
echo Agent finished - %date% %time%
pause
