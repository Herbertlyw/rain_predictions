"""Multipart API used by the Spring Boot service."""
import os
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP_DIR))
from forecast import load_model, write_results

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
RESULTS = Path(os.environ.get("RAIN_RESULTS_DIR", Path(__file__).with_name("results"))).resolve()


@app.get("/")
@app.get("/health")
def health():
    load_model()
    return jsonify(status="healthy", model_loaded=True)


@app.post("/predict")
def predict():
    try:
        task_id = str(uuid.UUID(request.form["task_id"])) if "task_id" in request.form else str(uuid.uuid4())
        inputs = int(request.form.get("input_length", 5))
        steps = int(request.form.get("predict_length", 5))
        comparison = request.form.get("generate_comparison", "true").lower() == "true"
        if (RESULTS / task_id).exists():
            return jsonify(status="error", message="任务编号已存在"), 409
        write_results(request.files.getlist("file"), inputs, steps, comparison, RESULTS, task_id)
    except ValueError as error:
        return jsonify(status="error", message=str(error)), 400
    prefix = f"/results/{task_id}"
    return jsonify(status="success", message="预测完成", task_id=task_id,
                   input_images=[f"{prefix}/input_{i:03d}.png" for i in range(inputs)],
                   prediction_images=[f"{prefix}/prediction_{i:03d}.png" for i in range(steps)],
                   comparison_image=f"{prefix}/comparison.png" if comparison else None,
                   download_url=f"/download/{task_id}")


@app.get("/results/<uuid:task_id>/<filename>")
def image(task_id, filename):
    return send_from_directory(RESULTS / str(task_id), filename)


@app.get("/download/<uuid:task_id>")
def download(task_id):
    return send_from_directory(RESULTS / str(task_id), "results.zip", as_attachment=True,
                               download_name=f"{task_id}_results.zip")


if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
