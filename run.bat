@echo off
REM EuniServ Invoice MVP — Windows launcher
cd /d %~dp0
if not exist .venv (
  echo Creating venv...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
if not exist .env (
  echo WARNING: .env not found. Copy .env.example to .env and add GEMINI_API_KEY.
  copy .env.example .env
)
if not exist uploads mkdir uploads
if not exist chroma_db mkdir chroma_db
echo Starting at http://127.0.0.1:8000
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
