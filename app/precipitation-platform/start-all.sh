#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${PYTHON:-python3}
"$python_bin" -c 'import torch, flask, numpy, matplotlib, PIL'
mvn -q -f backend/pom.xml package
export PORT=${RAIN_AI_PORT:-5050}
export AI_SERVICE_URL="http://127.0.0.1:$PORT"
"$python_bin" ai-service/flask_app.py &
ai=$!
java -jar backend/target/precipitation-prediction-1.0.0.jar &
backend=$!
trap 'kill "$ai" "$backend" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
echo "启动后访问 http://localhost:${SERVER_PORT:-8080}/api/"
while kill -0 "$ai" 2>/dev/null && kill -0 "$backend" 2>/dev/null; do sleep 1; done
exit 1
