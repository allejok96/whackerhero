@echo off
python -m venv venv && call venv\Scripts\activate.bat && pip install .[gui] && copy venv\Scripts\whackergui.exe WhackerHero.exe
pause