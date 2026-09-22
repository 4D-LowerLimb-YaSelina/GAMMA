import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from match.data.raw_faust import PrefetchedPairRawFAUST
from match.map_extractor.parallel_spectral_refined_partial_aware_map_extractor import (
    ParallelSpectrallyRefinedPartialAwareMapExtractor,
)
from match.metric.cached_s2t_geodist import CachedS2TGeodistMetric
from match.metric.s2t_geodist import S2TGeodistMetric


def _raw_faust_roots(root):
    data_root = root / "data"
    articulated_root = root / "articulated"
    scans = data_root / "scans"
    scans.mkdir(parents=True)
    articulated_root.mkdir()
    for index in range(100):
        (scans / f"tr_scan_{index:03d}.obj").touch()
    return data_root, articulated_root


def _pair_dataset(root, *, max_cached_shapes=20):
    data_root, articulated_root = _raw_faust_roots(root)
    return PrefetchedPairRawFAUST(
        data_root=data_root,
        articulated_shape_dir=articulated_root,
        phase="test",
        return_faces=True,
        return_L=False,
        return_mass=False,
        return_evecs=False,
        return_grad=False,
        return_corr=False,
        return_dist=False,
        return_articulation_confidence=False,
        shape_loading_workers=4,
        pair_prefetch_depth=32,
        max_cached_shapes=max_cached_shapes,
    )


def _fallback_shape(name):
    return {
        "name": name,
        "verts": torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ),
        "articulated_verts": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ),
    }


class PrefetchedRawFAUSTTests(unittest.TestCase):
    def test_default_cache_loads_each_test_shape_once_and_preserves_pair_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pairs = _pair_dataset(Path(temp_dir))

            def load_shape(index):
                return {"name": pairs.dataset.mesh_ls[index], "index": index}

            with mock.patch.object(
                pairs.dataset,
                "__getitem__",
                side_effect=load_shape,
            ) as loader:
                loaded_pairs = list(pairs)

            self.assertEqual(len(loaded_pairs), 400)
            self.assertEqual(loader.call_count, 20)
            self.assertEqual(
                [
                    (
                        pair["first"]["index"],
                        pair["second"]["index"],
                    )
                    for pair in loaded_pairs
                ],
                pairs.combinations,
            )

    def test_small_cache_reloads_evicted_shapes_without_changing_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pairs = _pair_dataset(Path(temp_dir), max_cached_shapes=0)

            with mock.patch.object(
                pairs.dataset,
                "__getitem__",
                side_effect=lambda index: {"index": index},
            ) as loader:
                loaded_pairs = list(pairs)

            self.assertGreater(loader.call_count, 20)
            self.assertEqual(
                [
                    (
                        pair["first"]["index"],
                        pair["second"]["index"],
                    )
                    for pair in loaded_pairs
                ],
                pairs.combinations,
            )
            self.assertEqual(pairs.resident_shape_limit, 2)

    def test_rejects_invalid_parallel_loading_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for argument, value, message in (
                ("shape_loading_workers", 0, "workers"),
                ("pair_prefetch_depth", 0, "depth"),
                ("max_cached_shapes", -1, "non-negative"),
            ):
                with self.subTest(argument=argument):
                    kwargs = {
                        "shape_loading_workers": 1,
                        "pair_prefetch_depth": 1,
                        "max_cached_shapes": 2,
                    }
                    kwargs[argument] = value
                    data_root, articulated_root = _raw_faust_roots(root / argument)
                    with self.assertRaisesRegex(ValueError, message):
                        PrefetchedPairRawFAUST(
                            data_root=data_root,
                            articulated_shape_dir=articulated_root,
                            phase="test",
                            return_mass=False,
                            return_evecs=False,
                            return_grad=False,
                            return_corr=False,
                            return_dist=False,
                            **kwargs,
                        )


class ParallelPartialAwareExtractorTests(unittest.TestCase):
    def test_ckdtree_queries_receive_workers_and_are_cached_per_shape(self):
        extractor = ParallelSpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            refinement_method="none",
            ckdtree_workers=4,
        )
        source = _fallback_shape("source")
        target = _fallback_shape("target")

        tree = mock.Mock()
        tree.query.side_effect = [
            (np.zeros(2), np.array([0, 2])),
            (np.zeros(3), np.array([0, 0, 1])),
        ]
        with mock.patch(
            "match.map_extractor."
            "parallel_spectral_refined_partial_aware_map_extractor.cKDTree",
            return_value=tree,
        ) as tree_type:
            first = extractor._dense_fallback_map(source, target)
            second = extractor._dense_fallback_map(source, target)

        np.testing.assert_array_equal(first, np.array([0, 1]))
        np.testing.assert_array_equal(second, first)
        self.assertEqual(tree_type.call_count, 2)
        self.assertEqual(tree.query.call_count, 2)
        for call in tree.query.call_args_list:
            self.assertEqual(call.kwargs["workers"], 4)

    def test_ckdtree_worker_validation(self):
        for workers in (0, -2, True, 1.5):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "positive integer or -1"):
                    ParallelSpectrallyRefinedPartialAwareMapExtractor(
                        device="cpu",
                        ckdtree_workers=workers,
                    )
        for workers in (1, 4, -1):
            with self.subTest(workers=workers):
                extractor = ParallelSpectrallyRefinedPartialAwareMapExtractor(
                    device="cpu",
                    ckdtree_workers=workers,
                )
                self.assertEqual(extractor.ckdtree_workers, workers)

    def test_none_only_prepares_correspondence_for_device(self):
        extractor = ParallelSpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            refinement_method="none",
        )
        shape = {
            "name": "shape",
            "corr": torch.arange(3),
            "dist": torch.eye(3),
            "evecs": torch.eye(3),
            "evecs_trans": torch.eye(3),
            "mass": torch.ones(3),
        }
        with mock.patch.object(
            extractor,
            "_tensor_on_device",
            wraps=extractor._tensor_on_device,
        ) as transfer:
            result = extractor._prepare_result_shape(
                shape,
                torch.device("cpu"),
                role="source",
            )

        self.assertEqual([call.args[1] for call in transfer.call_args_list], ["corr"])
        self.assertIs(result["dist"], shape["dist"])
        self.assertIs(result["evecs"], shape["evecs"])

    def test_spectral_modes_prepare_only_required_source_tensors(self):
        shape = {
            "name": "shape",
            "corr": torch.arange(3),
            "evecs": torch.eye(3),
            "evecs_trans": torch.eye(3),
            "mass": torch.ones(3),
        }
        for method, expected in (
            ("spectral-icp", {"corr", "evecs", "evecs_trans"}),
            ("zoomout", {"corr", "evecs", "evecs_trans", "mass"}),
        ):
            with self.subTest(method=method):
                extractor = ParallelSpectrallyRefinedPartialAwareMapExtractor(
                    device="cpu",
                    refinement_method=method,
                )
                with mock.patch.object(
                    extractor,
                    "_tensor_on_device",
                    wraps=extractor._tensor_on_device,
                ) as transfer:
                    extractor._prepare_result_shape(
                        shape,
                        torch.device("cpu"),
                        role="source",
                    )
                self.assertEqual(
                    {call.args[1] for call in transfer.call_args_list},
                    expected,
                )


class CachedS2TGeodistTests(unittest.TestCase):
    def test_reuses_template_distance_tensor_and_matches_baseline(self):
        template_dist = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        )
        map_result = {
            "dataset_name": "faust_raw",
            "shape_m": {
                "name": "m",
                "dist": template_dist,
                "corr": torch.tensor([2, 0, 1]),
            },
            "shape_n": {
                "name": "n",
                "corr": torch.tensor([0, 2, 1]),
            },
            "p2p": torch.tensor([1, 0, -1]),
            "method": "parallel",
        }
        baseline = S2TGeodistMetric()
        cached = CachedS2TGeodistMetric()

        baseline_output = baseline.pipeline(map_result)
        first_output = cached.pipeline(map_result)
        cached_tensor = cached._template_dist_device
        second_output = cached.pipeline(map_result)

        self.assertEqual(first_output, baseline_output)
        self.assertEqual(second_output, baseline_output)
        self.assertIs(cached._template_dist_device, cached_tensor)
        self.assertEqual(cached.values, baseline.values * 2)


class ParallelRawFAUSTConfigTests(unittest.TestCase):
    def test_parallel_preset_has_bounded_workers_and_lean_map_outputs(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="partial_aware_faust_raw_parallel")

        self.assertEqual(
            config.data._target_,
            "match.data.raw_faust." "ParallelRawFAUSTArticulatedShapeCorrDataset",
        )
        self.assertTrue(config.data.return_mass)
        self.assertFalse(config.data.return_evecs)
        self.assertEqual(config.data.shape_loading_workers, 4)
        self.assertEqual(config.data.pair_prefetch_depth, 32)
        self.assertEqual(config.data.max_cached_shapes, 20)
        self.assertEqual(config.map_extractor.ckdtree_workers, 4)
        self.assertEqual(config.map_extractor.refinement_method, "none")
        self.assertFalse(config.map_extractor.return_soft_maps)
        self.assertEqual(
            config.metrics.geodist._target_,
            "match.metric.cached_s2t_geodist.CachedS2TGeodistMetric",
        )

    def test_spectral_refinement_can_be_enabled_with_matching_data_overrides(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="partial_aware_faust_raw_parallel",
                overrides=[
                    "map_extractor.refinement_method=zoomout",
                    "data.return_evecs=true",
                    "data.return_mass=true",
                ],
            )

        self.assertEqual(config.map_extractor.refinement_method, "zoomout")
        self.assertTrue(config.data.return_evecs)
        self.assertTrue(config.data.return_mass)


if __name__ == "__main__":
    unittest.main()
