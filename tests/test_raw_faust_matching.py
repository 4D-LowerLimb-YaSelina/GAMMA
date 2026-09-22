import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from match.data.raw_faust import (
    RawFAUST,
    RawFAUSTArticulatedShapeCorrDataset,
)
from match.metric.s2t_geodist import S2TGeodistMetric, _s2t_geodist_batch


TRIANGLE_OBJ = """\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
TRIANGLE_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
TRIANGLE_FACES = np.array([[0, 1, 2]], dtype=np.int64)
TEMPLATE_DIST = np.array(
    [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _write_raw_faust_fixture(
    root,
    *,
    shape_count=100,
    correspondence=(0, 1, 2),
    template_dist=TEMPLATE_DIST,
):
    data_root = root / "data"
    articulated_root = root / "articulated"
    for path in (
        data_root / "scans",
        data_root / "spectral",
        data_root / "s2t",
        data_root / "dist",
        articulated_root / "mhr",
    ):
        path.mkdir(parents=True)

    for index in range(shape_count):
        name = f"tr_scan_{index:03d}"
        (data_root / "scans" / f"{name}.obj").write_text(TRIANGLE_OBJ)

    if shape_count > 80:
        name = "tr_scan_080"
        np.savez_compressed(
            data_root / "spectral" / f"{name}.npz",
            verts=TRIANGLE_VERTICES,
            faces=TRIANGLE_FACES,
            k_eig=np.asarray(2),
            mass=np.ones(3, dtype=np.float32),
            evals=np.array([0.0, 1.0], dtype=np.float32),
            evecs=np.array(
                [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
                dtype=np.float32,
            ),
        )
        np.savetxt(
            data_root / "s2t" / f"{name}.txt",
            np.asarray(correspondence),
            fmt="%d",
        )
        (articulated_root / "mhr" / f"{name}.obj").write_text(TRIANGLE_OBJ)
        np.savez_compressed(
            articulated_root / "mhr" / f"{name}.confidence.npz",
            vertex_uncertainty=np.zeros(3),
            vertex_visibility_support=np.ones(3),
        )

    np.savez_compressed(
        data_root / "dist" / "template_dist.npz",
        dist_mat=template_dist,
    )
    return data_root, articulated_root


class RawFAUSTDatasetTests(unittest.TestCase):
    def test_component_loads_test_split_and_shared_s2t_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root, articulated_root = _write_raw_faust_fixture(Path(temp_dir))
            component = RawFAUSTArticulatedShapeCorrDataset(
                data_root=data_root,
                articulated_shape_dir=articulated_root,
                phase="test",
                return_faces=True,
                return_L=False,
                return_mass=True,
                num_evecs=2,
                return_evecs=True,
                return_grad=False,
                return_corr=True,
                return_dist=True,
                return_articulation_confidence=True,
            )

            pairs = component.pipeline()
            self.assertEqual(component.dataset_name, "faust_raw")
            self.assertEqual(len(pairs.dataset), 20)
            self.assertEqual(len(pairs), 400)
            self.assertEqual(pairs.dataset.mesh_ls[0], "tr_scan_080")
            self.assertEqual(pairs.dataset.mesh_ls[-1], "tr_scan_099")

            pair = pairs[0]
            shape = pair["first"]
            self.assertEqual(shape["name"], "tr_scan_080")
            self.assertEqual(shape["verts"].shape, (3, 3))
            self.assertEqual(shape["evecs"].shape, (3, 2))
            torch.testing.assert_close(shape["corr"], torch.arange(3))
            torch.testing.assert_close(shape["dist"], torch.from_numpy(TEMPLATE_DIST))
            self.assertEqual(
                pair["first"]["dist"].data_ptr(),
                pair["second"]["dist"].data_ptr(),
            )
            self.assertEqual(shape["articulated_faces"].shape, (1, 3))
            torch.testing.assert_close(
                shape["articulation_confidence"]["vertex_uncertainty"],
                torch.zeros(3),
            )

    def test_raw_mesh_loading_and_all_phase_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root, articulated_root = _write_raw_faust_fixture(Path(temp_dir))
            common = {
                "data_root": data_root,
                "articulated_shape_dir": articulated_root,
                "return_L": False,
                "return_mass": False,
                "return_evecs": False,
                "return_grad": False,
                "return_corr": False,
                "return_dist": False,
            }
            train = RawFAUST(phase="train", **common)
            test = RawFAUST(phase="test", **common)
            full = RawFAUST(phase="full", **common)

            self.assertEqual(len(train), 80)
            self.assertEqual(len(test), 20)
            self.assertEqual(len(full), 100)
            self.assertEqual(test[0]["verts"].shape, (3, 3))
            self.assertEqual(test[0]["faces"].shape, (1, 3))

    def test_requires_exactly_one_hundred_scans(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root, articulated_root = _write_raw_faust_fixture(
                Path(temp_dir), shape_count=99
            )
            with self.assertRaisesRegex(ValueError, "100 human body shapes"):
                RawFAUST(
                    data_root=data_root,
                    articulated_shape_dir=articulated_root,
                    return_mass=False,
                    return_evecs=False,
                    return_grad=False,
                )

    def test_rejects_malformed_correspondence_length_and_range(self):
        for correspondence, message in (
            ((0, 1), "length mismatch"),
            ((0, 1, 3), "out of range"),
        ):
            with self.subTest(correspondence=correspondence):
                with tempfile.TemporaryDirectory() as temp_dir:
                    data_root, articulated_root = _write_raw_faust_fixture(
                        Path(temp_dir), correspondence=correspondence
                    )
                    dataset = RawFAUST(
                        data_root=data_root,
                        articulated_shape_dir=articulated_root,
                        phase="test",
                        return_mass=False,
                        return_evecs=False,
                        return_grad=False,
                        return_corr=True,
                        return_dist=True,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        dataset[0]

    def test_rejects_non_square_template_distances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root, articulated_root = _write_raw_faust_fixture(
                Path(temp_dir), template_dist=np.zeros((3, 2))
            )
            with self.assertRaisesRegex(ValueError, "square template distance"):
                RawFAUST(
                    data_root=data_root,
                    articulated_shape_dir=articulated_root,
                    return_mass=False,
                    return_evecs=False,
                    return_grad=False,
                    return_dist=True,
                )


class S2TGeodistMetricTests(unittest.TestCase):
    def test_composes_predicted_map_with_target_s2t_correspondence(self):
        metric = S2TGeodistMetric()
        shape_m = {
            "name": "m",
            "dist": torch.from_numpy(TEMPLATE_DIST),
            "corr": torch.tensor([2, 0, 1]),
        }
        shape_n = {
            "name": "n",
            "corr": torch.tensor([0, 2, 1]),
        }
        output = metric.pipeline(
            {
                "dataset_name": "faust_raw",
                "shape_m": shape_m,
                "shape_n": shape_n,
                "p2p": torch.tensor([1, 0, -1]),
                "method": "partial-aware-soft-canonization",
            }
        )

        self.assertIn("error=0.000000", output)
        self.assertIn("coverage=0.667", output)
        self.assertEqual(metric.values, [0.0])
        self.assertEqual(metric.coverages, [2 / 3])

    def test_masks_null_and_out_of_range_point_map_entries(self):
        distances, accepted = _s2t_geodist_batch(
            dist_template=torch.from_numpy(TEMPLATE_DIST)[None],
            corr_x=torch.tensor([[0, 1, 2]]),
            corr_y=torch.tensor([[0, 1, 2]]),
            p2p_y2x=torch.tensor([[-1, 3, 2]]),
        )
        torch.testing.assert_close(accepted, torch.tensor([[False, False, True]]))
        self.assertEqual(float(distances[accepted].item()), 0.0)


class RawFAUSTConfigTests(unittest.TestCase):
    def test_preset_composes_for_every_refinement_mode(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for refinement_method in ("none", "spectral-icp", "zoomout"):
                with self.subTest(refinement_method=refinement_method):
                    config = compose(
                        config_name="partial_aware_faust_raw",
                        overrides=[
                            "map_extractor.refinement_method="
                            f"{refinement_method}"
                        ],
                    )
                    self.assertEqual(
                        config.data._target_,
                        "match.data.raw_faust."
                        "RawFAUSTArticulatedShapeCorrDataset",
                    )
                    self.assertEqual(config.data.dataset_name, "faust_raw")
                    self.assertEqual(config.data.raw_mesh_subdir, "scans")
                    self.assertEqual(config.data.raw_mesh_extension, "obj")
                    self.assertTrue(config.data.return_mass)
                    self.assertTrue(config.data.return_evecs)
                    self.assertEqual(
                        config.map_extractor._target_,
                        "match.map_extractor."
                        "spectral_refined_partial_aware_map_extractor."
                        "SpectrallyRefinedPartialAwareMapExtractor",
                    )
                    self.assertEqual(
                        config.map_extractor.refinement_method,
                        refinement_method,
                    )
                    self.assertEqual(
                        config.map_extractor.spectral_icp_chunk_size,
                        2048,
                    )
                    self.assertEqual(
                        config.metrics.geodist._target_,
                        "match.metric.s2t_geodist.S2TGeodistMetric",
                    )


if __name__ == "__main__":
    unittest.main()
