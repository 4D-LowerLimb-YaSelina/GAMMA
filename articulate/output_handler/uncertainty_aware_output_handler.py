from pathlib import Path

import numpy as np

from articulate.output_handler.default_output_handler import DefaultOutputHandler


class UncertaintyAwareOutputHandler(DefaultOutputHandler):
    """Default MHR output plus a versioned confidence sidecar."""

    def store_confidence(self, mesh_name: str, confidence):
        self._mkdirs()
        path = self.out_dir / "mhr" / f"{mesh_name}.confidence.npz"
        np.savez_compressed(path, **confidence)
        return Path(path)

    def pipeline(self, estimation_refined):
        super().pipeline(estimation_refined)
        self.store_confidence(
            estimation_refined["shape"]["name"],
            estimation_refined["confidence"],
        )
