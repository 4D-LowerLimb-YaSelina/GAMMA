import numpy as np

from articulate.output_handler.smal_output_handler import SMALOutputHandler


class UncertaintyAwareSMALOutputHandler(SMALOutputHandler):
    """SMAL output plus a versioned confidence sidecar."""

    def store_confidence(self, mesh_name, confidence):
        self._mkdirs()
        path = self.out_dir / "smal" / f"{mesh_name}.confidence.npz"
        np.savez_compressed(path, **confidence)
        return path

    def pipeline(self, estimation_refined):
        super().pipeline(estimation_refined)
        self.store_confidence(
            estimation_refined["shape"]["name"],
            estimation_refined["confidence"],
        )
