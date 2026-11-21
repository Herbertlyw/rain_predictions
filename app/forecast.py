"""Inference with the two-layer ConvLSTM checkpoint shipped in best_model.pth."""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import io
import os
import zipfile

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from matplotlib import colormaps
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.conv = nn.Conv2d(channels + hidden, 4 * hidden, 3, padding=1)

    def forward(self, frame, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat((frame, h), dim=1)).chunk(4, dim=1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c


class ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_list = nn.ModuleList([ConvLSTMCell(1), ConvLSTMCell(32)])
        self.conv_last = nn.Conv2d(32, 1, 1, bias=False)

    def forward(self, frames, steps):
        state = [(frames.new_zeros(1, 32, *frames.shape[-2:]),
                  frames.new_zeros(1, 32, *frames.shape[-2:])) for _ in self.cell_list]
        predictions = []
        for t in range(len(frames) + steps - 1):
            x = frames[t][None, None] if t < len(frames) else predictions[-1]
            for layer, cell in enumerate(self.cell_list):
                state[layer] = cell(x, state[layer])
                x = state[layer][0]
            if t >= len(frames) - 1:
                predictions.append(self.conv_last(x).clamp(0, 1))
        return torch.cat(predictions)[:, 0]


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.environ.get("RAIN_MODEL_PATH", Path(__file__).with_name("best_model.pth")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastModel()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    return model.to(device).eval()


def read_frames(files, expected):
    if not 1 <= expected <= 20 or len(files) != expected:
        raise ValueError("输入帧数应在 1 到 20 之间，且与上传文件数相同")
    frames = []
    for file in files:
        if not (file.filename or "").lower().endswith(".npy"):
            raise ValueError("请选择 NPY 文件")
        try:
            frame = np.load(io.BytesIO(file.read()), allow_pickle=False)
        except (EOFError, OSError, ValueError) as error:
            raise ValueError("无法读取 NPY 文件") from error
        if not isinstance(frame, np.ndarray) or frame.dtype.kind not in "fiu":
            raise ValueError("请输入实数数组")
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 2 or min(frame.shape) < 1 or max(frame.shape) > 512:
            raise ValueError("每帧必须是高宽均在 1 到 512 之间的二维网格")
        if frames and frame.shape != frames[0].shape:
            raise ValueError("所有帧的尺寸必须相同")
        frame = frame.astype(np.float32)
        if not np.isfinite(frame).all():
            raise ValueError("网格中含有无效数值")
        frames.append(frame)
    return np.stack(frames)


_render_lock = Lock()


def write_results(files, expected, steps, comparison, directory, task_id):
    if not 1 <= steps <= 20:
        raise ValueError("预测帧数应在 1 到 20 之间")
    frames = read_frames(files, expected)
    low, high = float(frames.min()), float(frames.max())
    scale = high - low
    normalized = ((frames.astype(np.float64) - low) / (scale or 1)).astype(np.float32)
    model = load_model()
    with torch.inference_mode():
        tensor = torch.from_numpy(normalized).to(next(model.parameters()).device)
        future = model(tensor, steps).cpu().numpy()
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        for name, values in (("input", normalized), ("prediction", future)):
            for index, frame in enumerate(values):
                stem = f"{name}_{index:03d}"
                original = frames[index] if name == "input" else (frame.astype(np.float64) * scale + low).astype(np.float32)
                np.save(staging / f"{stem}.npy", original)
                pixels = colormaps["viridis"](frame, bytes=True)
                Image.fromarray(pixels).save(staging / f"{stem}.png")
        if comparison:
            with _render_lock:
                columns = max(expected, steps)
                fig = Figure(figsize=(columns * 1.8, 4), layout="constrained")
                axes = fig.subplots(2, columns, squeeze=False)
                for row, (label, series) in enumerate((("Input", normalized), ("Forecast", future))):
                    for col, ax in enumerate(axes[row]):
                        ax.set_axis_off()
                        if col < len(series):
                            ax.imshow(series[col], cmap="viridis", vmin=0, vmax=1)
                            ax.set_title(f"{label} {col + 1}")
                fig.savefig(staging / "comparison.png")
        with zipfile.ZipFile(staging / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(staging.iterdir()):
                if item.name != "results.zip":
                    archive.write(item, item.name)
        staging.rename(directory / task_id)
