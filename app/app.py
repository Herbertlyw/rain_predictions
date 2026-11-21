"""Single-process browser application for the bundled forecasting model."""
from datetime import datetime
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from forecast import load_model, write_results

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
RESULTS = Path(os.environ.get("RAIN_RESULTS_DIR", Path(__file__).with_name("predictions"))).resolve()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/test")
def health():
    return "服务器正常运行!"


@app.post("/predict")
def predict():
    task_id = str(uuid.uuid4())
    try:
        write_results(request.files.getlist("npy_files"), 5, 5, True, RESULTS, task_id)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(message="预测成功", session_id=task_id, zip_url=f"/download/{task_id}.zip",
                   comparison_url=f"/predictions/{task_id}/comparison.png")


@app.get("/result/<uuid:session_id>")
def result(session_id):
    if not (RESULTS / str(session_id)).is_dir():
        return jsonify(error="任务不存在"), 404
    return render_template("result.html", session_id=session_id,
                           timestamp=datetime.fromtimestamp((RESULTS / str(session_id)).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))


@app.get("/predictions/<uuid:session_id>/<filename>")
def image(session_id, filename):
    return send_from_directory(RESULTS / str(session_id), filename)


@app.get("/download/<uuid:session_id>.zip")
def download(session_id):
    if (RESULTS / str(session_id) / "results.zip").is_file():
        return send_from_directory(RESULTS / str(session_id), "results.zip", as_attachment=True,
                                   download_name=f"{session_id}.zip")
    return send_from_directory(RESULTS, f"{session_id}.zip", as_attachment=True)


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
