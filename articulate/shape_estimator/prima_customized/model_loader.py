from pathlib import Path


def load_prima(checkpoint_path):
    """Load PRIMA while resolving checkpoint-relative assets from its checkout."""
    import prima
    from prima.configs import get_config
    from prima.models.prima import PRIMA

    checkpoint_path = Path(checkpoint_path)
    model_config_path = checkpoint_path.parent.parent / ".hydra/config.yaml"
    model_cfg = get_config(str(model_config_path))

    if (
        model_cfg.MODEL.BACKBONE.TYPE == "vit"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise ValueError(
                "PRIMA ViT checkpoints must use MODEL.IMAGE_SIZE=256, got "
                f"{model_cfg.MODEL.IMAGE_SIZE}"
            )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    elif (
        model_cfg.MODEL.BACKBONE.TYPE == "dinov3"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise ValueError(
                "PRIMA DINOv3 checkpoints must use MODEL.IMAGE_SIZE=256, got "
                f"{model_cfg.MODEL.IMAGE_SIZE}"
            )
        model_cfg.MODEL.BBOX_SHAPE = [256, 256]
        model_cfg.freeze()
    elif (
        model_cfg.MODEL.BACKBONE.TYPE == "dinov2"
        and "BBOX_SHAPE" not in model_cfg.MODEL
    ):
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 252:
            raise ValueError(
                "PRIMA DINOv2 checkpoints must use MODEL.IMAGE_SIZE=252, got "
                f"{model_cfg.MODEL.IMAGE_SIZE}"
            )
        model_cfg.MODEL.BBOX_SHAPE = [252, 252]
        model_cfg.freeze()

    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    prima_root = Path(prima.__file__).resolve().parent.parent
    smal_data = prima_root / "data/smal"
    model_cfg.defrost()
    model_cfg.SMAL.MODEL_PATH = str(smal_data / "my_smpl_00781_4_all.pkl")
    model_cfg.SMAL.SHAPE_PRIOR_PATH = str(
        smal_data / "my_smpl_data_00781_4_all.pkl"
    )
    model_cfg.SMAL.POSE_PRIOR_PATH = str(
        smal_data / "walking_toy_symmetric_pose_prior_with_cov_35parts.pkl"
    )
    model_cfg.freeze()

    try:
        model = PRIMA.load_from_checkpoint(
            str(checkpoint_path),
            strict=False,
            cfg=model_cfg,
            map_location="cpu",
            init_renderer=False,
        )
    except (FileNotFoundError, RuntimeError) as error:
        if "bioclip" not in str(error).lower() and "huggingface" not in str(error).lower():
            raise
        raise RuntimeError(
            "PRIMA could not load its BioCLIP dependency. Allow Hugging Face "
            "access for the first run, or set HF_HUB_OFFLINE=1 when the "
            "imageomics/bioclip model is already cached."
        ) from error

    return model, model_cfg
