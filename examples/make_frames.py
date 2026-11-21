"""Generate five synthetic 32 x 32 radar grids for a local demonstration."""
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    folder = Path("demo-frames")
    folder.mkdir(exist_ok=True)
    y, x = np.mgrid[:32, :32]
    for t in range(5):
        frame = 30 * np.exp(-((x - 9 - t) ** 2 + (y - 15) ** 2) / 35)
        np.save(folder / f"frame_{t:02d}.npy", frame.astype(np.float32))
    print(folder.resolve())
