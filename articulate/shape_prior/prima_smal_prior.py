import torch

from articulate.shape_prior.smal_prior import SMAL_BONE_PAIRS


class PRIMASMALShapePrior:
    """Expose the SMAL layer owned by an initialized PRIMA model."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.smal_model = None
        self.faces = None

    def load(self):
        # PRIMA owns and checkpoints its SMAL layer, so it is attached below.
        return self.smal_model

    def load_from_estimator(self, shape_estimator):
        if shape_estimator.model is None:
            shape_estimator.init_estimator()
        self.smal_model = shape_estimator.model.smal
        self.faces = shape_estimator.faces
        return self.smal_model

    def generate(self, betas, go_aa, pose_aa, R, s, t):
        if self.smal_model is None:
            raise RuntimeError(
                "SMAL prior has not been attached to the PRIMA estimator"
            )

        from prima.utils.geometry import aa_to_rotmat

        global_orient = aa_to_rotmat(go_aa.reshape(-1, 3)).reshape(-1, 1, 3, 3)
        pose = aa_to_rotmat(pose_aa.reshape(-1, 3)).reshape(-1, 34, 3, 3)
        output = self.smal_model(
            betas=betas,
            global_orient=global_orient,
            pose=pose,
        )

        x_180 = torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            device=output.vertices.device,
            dtype=output.vertices.dtype,
        )
        vertices = output.vertices @ x_180.T
        joints = output.joints @ x_180.T

        vertices = s[:, None, None] * (
            vertices @ R.permute(0, 2, 1)
        ) + t[:, None, :]
        joints = s[:, None, None] * (
            joints @ R.permute(0, 2, 1)
        ) + t[:, None, :]
        return vertices, joints

    def get_skeleton(self):
        return SMAL_BONE_PAIRS
