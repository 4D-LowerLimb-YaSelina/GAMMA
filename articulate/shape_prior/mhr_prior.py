import torch
from hydra.utils import to_absolute_path

MHR70_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_hip",
    10: "right_hip",
    11: "left_knee",
    12: "right_knee",
    13: "left_ankle",
    14: "right_ankle",
    15: "left_big_toe",
    16: "left_small_toe",
    17: "left_heel",
    18: "right_big_toe",
    19: "right_small_toe",
    20: "right_heel",
    21: "right_thumb4",
    22: "right_thumb3",
    23: "right_thumb2",
    24: "right_thumb_third_joint",
    25: "right_forefinger4",
    26: "right_forefinger3",
    27: "right_forefinger2",
    28: "right_forefinger_third_joint",
    29: "right_middle_finger4",
    30: "right_middle_finger3",
    31: "right_middle_finger2",
    32: "right_middle_finger_third_joint",
    33: "right_ring_finger4",
    34: "right_ring_finger3",
    35: "right_ring_finger2",
    36: "right_ring_finger_third_joint",
    37: "right_pinky_finger4",
    38: "right_pinky_finger3",
    39: "right_pinky_finger2",
    40: "right_pinky_finger_third_joint",
    41: "right_wrist",
    42: "left_thumb4",
    43: "left_thumb3",
    44: "left_thumb2",
    45: "left_thumb_third_joint",
    46: "left_forefinger4",
    47: "left_forefinger3",
    48: "left_forefinger2",
    49: "left_forefinger_third_joint",
    50: "left_middle_finger4",
    51: "left_middle_finger3",
    52: "left_middle_finger2",
    53: "left_middle_finger_third_joint",
    54: "left_ring_finger4",
    55: "left_ring_finger3",
    56: "left_ring_finger2",
    57: "left_ring_finger_third_joint",
    58: "left_pinky_finger4",
    59: "left_pinky_finger3",
    60: "left_pinky_finger2",
    61: "left_pinky_finger_third_joint",
    62: "left_wrist",
    63: "left_olecranon",
    64: "right_olecranon",
    65: "left_cubital_fossa",
    66: "right_cubital_fossa",
    67: "left_acromion",
    68: "right_acromion",
    69: "neck",
}

MHR70_KEYPOINT_NAME_TO_ID = {v: k for k, v in MHR70_KEYPOINT_NAMES.items()}

MHR70_BONE_PAIRS_ALL = [
    (13, 11),
    (11, 9),
    (14, 12),
    (12, 10),
    (9, 10),
    (5, 9),
    (6, 10),
    (69, 5),
    (69, 6),
    (5, 7),
    (6, 8),
    (7, 62),
    (8, 41),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 69),
    (4, 69),
    (13, 15),
    (13, 16),
    (13, 17),
    (14, 18),
    (14, 19),
    (14, 20),
    (62, 45),
    (45, 44),
    (44, 43),
    (43, 42),
    (62, 49),
    (49, 48),
    (48, 47),
    (47, 46),
    (62, 53),
    (53, 52),
    (52, 51),
    (51, 50),
    (62, 57),
    (57, 56),
    (56, 55),
    (55, 54),
    (62, 61),
    (61, 60),
    (60, 59),
    (59, 58),
    (41, 24),
    (24, 23),
    (23, 22),
    (22, 21),
    (41, 28),
    (28, 27),
    (27, 26),
    (26, 25),
    (41, 32),
    (32, 31),
    (31, 30),
    (30, 29),
    (41, 36),
    (36, 35),
    (35, 34),
    (34, 33),
    (41, 40),
    (40, 39),
    (39, 38),
    (38, 37),
]

MHR70_BONE_PAIRS_BODY = [
    (13, 11),
    (11, 9),
    (14, 12),
    (12, 10),
    (69, 9),
    (69, 10),
    (9, 10),
    (5, 69),
    (6, 69),
    (5, 7),
    (6, 8),
    (7, 62),
    (8, 41),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 69),
    (4, 69),
    (13, 15),
    (13, 16),
    (13, 17),
    (14, 18),
    (14, 19),
    (14, 20),
]

MHR70_BONE_PAIRS_CRITICAL = [
    (9, 10),
    (5, 69),
    (6, 69),
    (69, 9),
    (69, 10),
    (69, 0),
    (3, 0),
    (4, 0),
    (11, 9),
    (12, 10),
    (13, 11),
    (14, 12),
    (5, 7),
    (6, 8),
    (7, 62),
    (8, 41),
]


def get_bone_pairs(mode="all"):
    """
    Get bone pairs for MHR70 skeleton.
    """
    if mode == "all":
        return MHR70_BONE_PAIRS_ALL
    elif mode == "body":
        return MHR70_BONE_PAIRS_BODY
    elif mode == "critical":
        return MHR70_BONE_PAIRS_CRITICAL
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'all', 'body', or 'critical'")


class MHRShapePrior:
    def __init__(
        self,
        mhr_path: str = "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        device: str = "cuda",
    ):
        self.mhr_path = to_absolute_path(str(mhr_path))
        self.device = device
        self.scripted_mhr_model = None
        self.keypoint_mapping = None

    def load(self):
        if self.scripted_mhr_model is None:
            print("Loading scripted MHR model...")
            self.scripted_mhr_model = torch.jit.load(self.mhr_path).to(self.device)
        return self.scripted_mhr_model

    def load_from_estimator(self, shape_estimator):
        if shape_estimator.model is None:
            shape_estimator.init_estimator()

        self.keypoint_mapping = shape_estimator.model.head_pose.keypoint_mapping
        return self.keypoint_mapping

    def generate(
        self,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        R,
        s,
        t,
        return_keypoints=False,
        keypoint_mapping=None,
    ):
        """Generate MHR vertices with the estimated transform applied."""
        scripted_mhr_model = self.load()
        if keypoint_mapping is None:
            keypoint_mapping = self.keypoint_mapping

        vertices, skel_state = scripted_mhr_model(
            identity_coeffs, model_parameters, face_expr_coeffs
        )

        vertices /= 100
        vertices = s[:, None, None] * (vertices @ R.permute(0, 2, 1)) + t[:, None, :]

        if not return_keypoints:
            return vertices

        raw_coords, keypoint_quats, _ = torch.split(skel_state, [3, 4, 1], dim=2)
        raw_coords = raw_coords / 100
        raw_coords = (
            s[:, None, None] * (raw_coords @ R.permute(0, 2, 1)) + t[:, None, :]
        )

        model_vert_joints = torch.cat([vertices, raw_coords], dim=1)
        keypoints_coords = (
            (keypoint_mapping @ model_vert_joints.permute(1, 0, 2).flatten(1, 2))
            .reshape(-1, model_vert_joints.shape[0], 3)
            .permute(1, 0, 2)
        )

        return vertices, keypoints_coords, keypoint_quats

    def get_skeleton(self, mode="all"):
        return get_bone_pairs(mode=mode)

    def get_skeletion(self, mode="all"):
        return self.get_skeleton(mode=mode)
