import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse
import torch
import trimesh
from hydra import compose, initialize_config_dir

from match.data.becos import (
    BeCoSArticulatedShapeCorrDataset,
    PairBeCoSArticulatedDataset,
)
from match.map_extractor.spectral_refined_partial_aware_map_extractor import (
    SpectrallyRefinedPartialAwareMapExtractor,
)
from match.metric.becos_geodist import (
    BeCoSGeodistMetric,
    _becos_geodist_batch,
)

VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FACES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ],
    dtype=np.int64,
)


def _write_mesh(path, vertices=VERTICES, faces=FACES):
    trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    ).export(path)


def _make_pair(
    data_root,
    pair_id,
    *,
    phase="test",
    vertices_0=VERTICES,
    vertices_1=VERTICES,
    include_corr_01=False,
    correspondence=None,
):
    pair_dir = data_root / phase / str(pair_id)
    pair_dir.mkdir(parents=True)
    _write_mesh(pair_dir / "0_shape.off", vertices_0)
    _write_mesh(pair_dir / "1_shape.off", vertices_1)
    if correspondence is None:
        correspondence = np.arange(len(vertices_1), dtype=np.int64)
    np.save(
        pair_dir / "corres_10.npy",
        np.asarray(correspondence, dtype=np.int64),
    )
    if include_corr_01:
        np.save(
            pair_dir / "corres_01.npy",
            np.arange(len(vertices_0), dtype=np.int64),
        )
    return pair_dir


def _make_artifacts(articulated_root, pair_id, *, phase="test"):
    artifact_dir = articulated_root / phase / str(pair_id) / "mhr"
    artifact_dir.mkdir(parents=True)
    for mesh_stem in ("0_shape", "1_shape"):
        _write_mesh(artifact_dir / f"{mesh_stem}.obj")
        np.savez_compressed(
            artifact_dir / f"{mesh_stem}.confidence.npz",
            vertex_uncertainty=np.arange(len(VERTICES), dtype=np.float32),
            vertex_visibility_support=np.ones(len(VERTICES), dtype=np.float32),
        )
    return artifact_dir


def _fake_operators(vertices, faces, k):
    del faces
    vertex_count = len(vertices)
    diagonal = np.arange(vertex_count)
    sparse = scipy.sparse.coo_matrix(
        (np.ones(vertex_count, dtype=np.float32), (diagonal, diagonal)),
        shape=(vertex_count, vertex_count),
    )
    sparse = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((sparse.row, sparse.col))).long(),
        torch.from_numpy(sparse.data),
        sparse.shape,
    ).coalesce()
    eigenvectors = torch.zeros((vertex_count, k), dtype=torch.float32)
    eigenvectors[:k, :k] = torch.eye(k)
    return (
        torch.ones(vertex_count),
        sparse,
        torch.arange(k, dtype=torch.float32),
        eigenvectors,
        sparse,
        sparse,
    )


def _fake_geodesic(vertices, faces):
    del faces
    indices = np.arange(len(vertices))
    return np.abs(indices[:, None] - indices[None, :]).astype(np.float64)


class BeCoSMatchingDatasetTests(unittest.TestCase):
    def test_component_loads_pair_artifacts_cache_and_benchmark_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            articulated_root = root / "articulated"
            _make_pair(data_root, 7)
            _make_artifacts(articulated_root, 7)

            component = BeCoSArticulatedShapeCorrDataset(
                data_root=data_root,
                articulated_shape_dir=articulated_root,
                phase="test",
                mode="partial_full",
                num_evecs=3,
            )
            with (
                mock.patch(
                    "match.data.becos._compute_operators",
                    side_effect=_fake_operators,
                ),
                mock.patch(
                    "match.data.becos._compute_geodesic_distances",
                    side_effect=_fake_geodesic,
                ),
            ):
                dataset = component.pipeline()
                pair = dataset[0]

            self.assertEqual(component.dataset_name, "faust_becos")
            self.assertEqual(len(dataset), 1)
            self.assertEqual(pair["pair_id"], "7")
            self.assertEqual(pair["first"]["name"], "7:0_shape")
            self.assertEqual(pair["second"]["name"], "7:1_shape")
            self.assertEqual(pair["first"]["mesh_name"], "0_shape")
            self.assertEqual(pair["first"]["evecs"].shape, (4, 3))
            self.assertEqual(pair["second"]["evecs_trans"].shape, (3, 4))
            torch.testing.assert_close(
                pair["first"]["corr"],
                torch.arange(4),
            )
            torch.testing.assert_close(
                pair["second"]["corr"],
                torch.arange(4),
            )
            self.assertEqual(pair["first"]["articulated_faces"].shape, (4, 3))
            torch.testing.assert_close(
                pair["second"]["articulation_confidence"]["vertex_uncertainty"],
                torch.arange(4, dtype=torch.float32),
            )
            torch.testing.assert_close(
                pair["first"]["dist"],
                torch.from_numpy(_fake_geodesic(VERTICES, FACES)).float() / 2,
            )

            metric = BeCoSGeodistMetric()
            output = metric.pipeline(
                {
                    "dataset_name": component.dataset_name,
                    "method": "full-coverage-soft-canonization",
                    "shape_m": pair["first"],
                    "shape_n": pair["second"],
                    "p2p": torch.arange(4),
                }
            )
            self.assertIn("error=0.000000", output)
            self.assertIn("coverage=1.000", output)

            extractor = SpectrallyRefinedPartialAwareMapExtractor(
                device="cpu",
                matching_mode="full",
                refinement_method="none",
            )
            map_result = extractor.pipeline(
                shape_m=pair["first"],
                shape_n=pair["second"],
            )
            self.assertEqual(map_result["spectral_refinement"], "none")
            self.assertEqual(map_result["p2p"].shape, (4,))
            self.assertTrue(torch.all(map_result["p2p"] >= 0))

    def test_partial_partial_sentinels_are_excluded_from_benchmark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            _make_pair(
                data_root,
                0,
                include_corr_01=True,
                correspondence=np.array([0, -1, 2, 3]),
            )
            dataset = PairBeCoSArticulatedDataset(
                data_root=data_root,
                articulated_shape_dir=root / "articulated",
                phase="test",
                mode="partial_partial",
                return_mass=False,
                return_evecs=False,
                return_dist=False,
                return_articulation_confidence=False,
                compute_cache=False,
            )
            pair = dataset[0]
            torch.testing.assert_close(
                pair["first"]["corr"],
                torch.tensor([0, -1, 2, 3]),
            )

            distances, accepted, evaluable = _becos_geodist_batch(
                dist_x=torch.from_numpy(_fake_geodesic(VERTICES, FACES))[None].float(),
                corr_x=pair["first"]["corr"][None],
                corr_y=pair["second"]["corr"][None],
                p2p_y2x=torch.arange(4)[None],
            )
            torch.testing.assert_close(
                evaluable,
                torch.tensor([[True, False, True, True]]),
            )
            torch.testing.assert_close(accepted, evaluable)
            torch.testing.assert_close(
                distances[accepted],
                torch.zeros(3),
            )

            metric = BeCoSGeodistMetric()
            output = metric.pipeline(
                {
                    "dataset_name": "faust_becos",
                    "method": "full-coverage-soft-canonization",
                    "shape_m": {
                        **pair["first"],
                        "dist": torch.from_numpy(
                            _fake_geodesic(VERTICES, FACES)
                        ).float(),
                    },
                    "shape_n": pair["second"],
                    "p2p": torch.arange(4),
                }
            )
            self.assertIn("error=0.000000", output)
            self.assertIn("coverage=1.000 (3/3)", output)

    def test_intrinsic_cache_is_shared_and_readable_without_utility_imports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            articulated_root = root / "articulated"
            transformed = VERTICES + np.array([3.0, -2.0, 1.0])
            _make_pair(data_root, 0, vertices_1=transformed)
            _make_pair(
                data_root,
                1,
                vertices_0=transformed,
                vertices_1=VERTICES,
            )
            calls = []

            def count_operators(*args):
                calls.append(args)
                return _fake_operators(*args)

            dataset = PairBeCoSArticulatedDataset(
                data_root=data_root,
                articulated_shape_dir=articulated_root,
                phase="test",
                mode="partial_full",
                num_evecs=3,
                return_dist=False,
                return_articulation_confidence=False,
                compute_cache=True,
            )
            with mock.patch(
                "match.data.becos._compute_operators",
                side_effect=count_operators,
            ):
                first_pair = dataset[0]
                second_pair = dataset[1]

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                len(list((data_root / "spectral_cache").glob("*_spectral.npz"))),
                1,
            )
            torch.testing.assert_close(
                first_pair["first"]["evecs"],
                second_pair["first"]["evecs"],
            )

            read_only = PairBeCoSArticulatedDataset(
                data_root=data_root,
                articulated_shape_dir=articulated_root,
                phase="test",
                mode="partial_full",
                num_evecs=2,
                return_dist=False,
                return_articulation_confidence=False,
                compute_cache=False,
            )
            self.assertEqual(read_only[0]["first"]["evecs"].shape, (4, 2))

    def test_partial_partial_requires_reverse_correspondence_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            _make_pair(data_root, 0, include_corr_01=False)
            _make_pair(data_root, 1, include_corr_01=True)

            dataset = PairBeCoSArticulatedDataset(
                data_root=data_root,
                articulated_shape_dir=root / "articulated",
                phase="test",
                mode="partial_partial",
                return_mass=False,
                return_evecs=False,
                return_dist=False,
                return_articulation_confidence=False,
                compute_cache=False,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.pairs[0]["pair_id"], "1")

    def test_repeated_mesh_stems_receive_pair_qualified_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            _make_pair(data_root, 0)
            _make_pair(data_root, 1)
            dataset = PairBeCoSArticulatedDataset(
                data_root=data_root,
                articulated_shape_dir=root / "articulated",
                phase="test",
                mode="partial_full",
                return_mass=False,
                return_evecs=False,
                return_dist=False,
                return_articulation_confidence=False,
                compute_cache=False,
            )
            self.assertNotEqual(
                dataset[0]["first"]["name"],
                dataset[1]["first"]["name"],
            )


class BeCoSMatchingConfigTests(unittest.TestCase):
    def test_preset_composes_for_all_refinement_methods(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for refinement_method in ("none", "spectral-icp", "zoomout"):
                with self.subTest(refinement_method=refinement_method):
                    config = compose(
                        config_name="partial_aware_becos",
                        overrides=[
                            "map_extractor.refinement_method=" f"{refinement_method}"
                        ],
                    )
                    self.assertEqual(
                        config.data._target_,
                        "match.data.becos." "BeCoSArticulatedShapeCorrDataset",
                    )
                    self.assertEqual(config.data.mode, "partial_full")
                    self.assertTrue(config.data.return_mass)
                    self.assertTrue(config.data.return_evecs)
                    self.assertEqual(config.data.num_evecs, 200)
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
                        config.metrics.geodist._target_,
                        "match.metric.becos_geodist.BeCoSGeodistMetric",
                    )

    def test_partial_partial_mode_is_available_by_override(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="partial_aware_becos",
                overrides=[
                    "data.mode=partial_partial",
                    "data.data_root=data/shape-becos/faust_partial_partial",
                ],
            )
        self.assertEqual(config.data.mode, "partial_partial")
        self.assertEqual(
            config.data.data_root,
            "data/shape-becos/faust_partial_partial",
        )


if __name__ == "__main__":
    unittest.main()
