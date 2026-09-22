from pathlib import Path

from articulate.output_handler.uncertainty_aware_output_handler import (
    UncertaintyAwareOutputHandler,
)


class BeCoSUncertaintyAwareOutputHandler(UncertaintyAwareOutputHandler):
    """Write articulation artifacts under ``out_dir/<split>/<pair_id>``."""

    def pipeline(self, estimation_refined):
        shape = estimation_refined["shape"]
        split = shape.get("becos_split")
        pair_id = shape.get("becos_pair_id")
        if split is None or pair_id is None:
            raise KeyError(
                "Loaded shape is missing required 'becos_split' or "
                "'becos_pair_id' metadata"
            )

        run_out_dir = self.out_dir
        self.out_dir = run_out_dir / str(split) / str(pair_id)
        try:
            super().pipeline(estimation_refined)
        finally:
            self.out_dir = run_out_dir

    def log_failed(self, mesh_path):
        mesh_path = Path(mesh_path)
        failed_root = self.out_dir / mesh_path.parent.parent.name
        failed_root.mkdir(parents=True, exist_ok=True)
        with (failed_root / "failed.txt").open("a", encoding="utf-8") as file:
            file.write(str(mesh_path) + "\n")
