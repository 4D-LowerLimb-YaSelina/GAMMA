from pathlib import Path


def load_animer(checkpoint_path):
    """Load AniMer without mutating the clean checkout."""
    import amr
    from amr.configs import get_config
    from amr.models import AMR

    model_config_path = Path(checkpoint_path).parent.parent / ".hydra/config.yaml"
    model_cfg = get_config(str(model_config_path), update_cachedir=True)

    if (
        model_cfg.MODEL.BACKBONE.TYPE == "vit"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise ValueError(
                "AniMer ViT checkpoints must use MODEL.IMAGE_SIZE=256, got "
                f"{model_cfg.MODEL.IMAGE_SIZE}"
            )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    # Upstream checkpoints contain working-directory-relative SMAL paths. Hydra
    # runs this project from its own root, so resolve the clean checkout's
    # bundled SMAL assets explicitly.
    animer_root = Path(amr.__file__).resolve().parent.parent
    smal_data = animer_root / "data/smal"
    model_cfg.defrost()
    model_cfg.SMAL.MODEL_PATH = str(smal_data / "my_smpl_00781_4_all.pkl")
    model_cfg.SMAL.SHAPE_PRIOR_PATH = str(
        smal_data / "my_smpl_data_00781_4_all.pkl"
    )
    model_cfg.SMAL.POSE_PRIOR_PATH = str(
        smal_data / "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl"
    )
    model_cfg.freeze()

    model = AMR.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        cfg=model_cfg,
        map_location="cpu",
        weights_only=False,
    )
    return model, model_cfg
