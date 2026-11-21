@echo off
setlocal
cd /d "%~dp0"
python -c "import flask, torch, numpy, matplotlib, PIL" || exit /b 1
call mvn -q -f backend/pom.xml package || exit /b 1
if not defined RAIN_AI_PORT set RAIN_AI_PORT=5050
set PORT=%RAIN_AI_PORT%
set AI_SERVICE_URL=http://127.0.0.1:%PORT%
start "Rain inference" python ai-service/flask_app.py
if not defined SERVER_PORT set SERVER_PORT=8080
echo Open http://localhost:%SERVER_PORT%/api/
java -jar backend/target/precipitation-prediction-1.0.0.jar
