import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import app as web


def upload(frame, i):
    stream = io.BytesIO()
    np.save(stream, frame)
    stream.seek(0)
    return stream, f"frame_{i}.npy"


class ForecastTest(unittest.TestCase):
    def test_uploads_five_observations_and_downloads_five_future_grids(self):
        frames = [np.roll(np.eye(8, dtype=np.float32), i, axis=0) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory, patch.object(web, "RESULTS", Path(directory)):
            client = web.app.test_client()
            data = {"npy_files": [upload(frame, i) for i, frame in enumerate(frames)]}
            result = client.post("/predict", data=data)
            self.assertEqual(result.status_code, 200)
            with client.get(result.json["comparison_url"]) as comparison:
                self.assertEqual(comparison.mimetype, "image/png")
            with client.get(result.json["zip_url"]) as download, zipfile.ZipFile(io.BytesIO(download.data)) as archive:
                for i, frame in enumerate(frames):
                    np.testing.assert_array_equal(np.load(archive.open(f"input_{i:03d}.npy")), frame)
                    prediction = np.load(archive.open(f"prediction_{i:03d}.npy"))
                    self.assertEqual(prediction.shape, frame.shape)
                    self.assertTrue(np.isfinite(prediction).all())

    def test_rejects_a_mismatched_frame_without_publishing_partial_results(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(web, "RESULTS", Path(directory)):
            frames = [np.ones((8, 8))] * 4 + [np.ones((4, 4))]
            res = web.app.test_client().post("/predict", data={"npy_files": [upload(x, i) for i, x in enumerate(frames)]})
            self.assertEqual(res.status_code, 400)
            self.assertEqual(list(Path(directory).iterdir()), [])
