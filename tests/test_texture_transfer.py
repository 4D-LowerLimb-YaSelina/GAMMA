import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir

from match.metric.texture_transfer import (
    TextureTransferMetric,
    _transfer_uv_with_hard_map,
    _transfer_uv_with_spectral_permutation,
)


class TextureTransferMetricTests(unittest.TestCase):
    def test_pipeline_always_uses_final_hard_point_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metric = TextureTransferMetric(
                out_dir=temp_dir,
                texture_path=Path(temp_dir) / "unused.png",
            )
            shape_m = {
                "name": "source",
                "verts": torch.zeros((3, 3)),
            }
            shape_n = {
                "name": "target",
                "verts": torch.zeros((2, 3)),
            }
            misleading_soft_map = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 1]]),
                torch.ones(2),
                (2, 3),
            ).coalesce()

            with mock.patch.object(
                metric,
                "_texture_transfer_visualization",
            ) as visualize:
                output = metric.pipeline(
                    {
                        "shape_m": shape_m,
                        "shape_n": shape_n,
                        "p2p": torch.tensor([2, 0]),
                        "soft_map_real": misleading_soft_map,
                        "method": "test-method",
                    }
                )

            self.assertEqual(output, "exported")
            p2p_tgt_src = visualize.call_args.kwargs["p2p_tgt_src"]
            torch.testing.assert_close(
                p2p_tgt_src,
                torch.tensor([2, 0]),
            )

    def test_hard_uv_transfer_uses_linear_memory_and_preserves_nulls(self):
        uv_src = torch.tensor(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]
        )
        uv_tgt = _transfer_uv_with_hard_map(
            uv_src,
            torch.tensor([2, -1, 0]),
        )
        torch.testing.assert_close(
            uv_tgt,
            torch.tensor(
                [
                    [0.5, 0.6],
                    [0.0, 0.0],
                    [0.1, 0.2],
                ]
            ),
        )

    def test_spectral_uv_transfer_matches_original_dense_permutation(self):
        generator = torch.Generator().manual_seed(11)
        uv_src = torch.rand((5, 2), generator=generator)
        p2p_tgt_src = torch.tensor([2, 0, -1, 4])
        evecs_src = torch.rand((5, 3), generator=generator)
        evecs_tgt = torch.rand((4, 3), generator=generator)
        evecs_trans_src = torch.rand((3, 5), generator=generator)
        evecs_trans_tgt = torch.rand((3, 4), generator=generator)

        safe_p2p = p2p_tgt_src.clamp(min=0)
        matched_src = evecs_src[safe_p2p]
        matched_src[p2p_tgt_src == -1] = 0
        functional_map = evecs_trans_tgt @ matched_src
        dense_soft_permutation = evecs_tgt @ functional_map @ evecs_trans_src
        expected = dense_soft_permutation @ uv_src

        actual = _transfer_uv_with_spectral_permutation(
            uv_src,
            p2p_tgt_src,
            evecs_src,
            evecs_tgt,
            evecs_trans_src,
            evecs_trans_tgt,
        )
        torch.testing.assert_close(actual, expected)

    def test_soft_permutation_defaults_true_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            default_metric = TextureTransferMetric(
                out_dir=temp_dir,
                texture_path=Path(temp_dir) / "unused.png",
            )
            hard_metric = TextureTransferMetric(
                out_dir=temp_dir,
                texture_path=Path(temp_dir) / "unused.png",
                use_soft_permutation=False,
            )
        self.assertTrue(default_metric.use_soft_permutation)
        self.assertFalse(hard_metric.use_soft_permutation)

    def test_soft_permutation_flag_selects_transfer_from_final_p2p(self):
        shape = {
            "evecs": torch.eye(3),
            "evecs_trans": torch.eye(3),
        }
        uv_src = torch.rand((3, 2))
        p2p = torch.tensor([2, 1, 0])

        with tempfile.TemporaryDirectory() as temp_dir:
            soft_metric = TextureTransferMetric(
                out_dir=temp_dir,
                texture_path=Path(temp_dir) / "unused.png",
            )
            hard_metric = TextureTransferMetric(
                out_dir=temp_dir,
                texture_path=Path(temp_dir) / "unused.png",
                use_soft_permutation=False,
            )
            with (
                mock.patch(
                    "match.metric.texture_transfer."
                    "_transfer_uv_with_spectral_permutation",
                    return_value=torch.ones_like(uv_src),
                ) as spectral_transfer,
                mock.patch(
                    "match.metric.texture_transfer._transfer_uv_with_hard_map",
                    return_value=torch.zeros_like(uv_src),
                ) as hard_transfer,
            ):
                soft_result = soft_metric._transfer_uv(shape, shape, p2p, uv_src)
                hard_result = hard_metric._transfer_uv(shape, shape, p2p, uv_src)

        spectral_transfer.assert_called_once()
        hard_transfer.assert_called_once_with(uv_src, p2p)
        torch.testing.assert_close(soft_result, torch.ones_like(uv_src))
        torch.testing.assert_close(hard_result, torch.zeros_like(uv_src))

    def test_texture_transfer_smooths_hard_map_without_retained_soft_maps(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="partial_aware_faust_raw",
                overrides=["metrics=[geodist_s2t,texture_transfer]"],
            )

        self.assertTrue(config.metrics.texture_transfer.use_soft_permutation)
        self.assertFalse(config.map_extractor.return_soft_maps)


if __name__ == "__main__":
    unittest.main()
