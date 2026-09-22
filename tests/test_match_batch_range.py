import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import match.run as match_run


class _TrackingDataset:
    def __init__(self, size):
        self.pairs = [
            {
                "first": {
                    "name": f"first-{index}",
                    "articulated_verts": object(),
                },
                "second": {
                    "name": f"second-{index}",
                    "articulated_verts": object(),
                },
            }
            for index in range(size)
        ]
        self.visited = []

    def __len__(self):
        return len(self.pairs)

    def __iter__(self):
        for index, pair in enumerate(self.pairs):
            self.visited.append(index)
            yield pair


class _Data:
    dataset_name = "test"

    def __init__(self, dataset):
        self.dataset = dataset

    def pipeline(self):
        return self.dataset


class _MapExtractor:
    method = "test-map"

    def __init__(self):
        self.processed = []

    def pipeline(self, shape_m, shape_n):
        self.processed.append(int(shape_m["name"].removeprefix("first-")))
        return {
            "shape_m": shape_m,
            "shape_n": shape_n,
            "method": self.method,
        }


class _Metric:
    def __init__(self):
        self.pair_indices = []
        self.finalized = False

    def pipeline(self, map_result, pair_index):
        self.pair_indices.append(pair_index)
        return None

    def finalize(self):
        self.finalized = True


class MatchBatchRangeTests(unittest.TestCase):
    def _run(self, batch_start, batch_end):
        dataset = _TrackingDataset(size=6)
        data = _Data(dataset)
        map_extractor = _MapExtractor()
        metric = _Metric()
        cfg = OmegaConf.create(
            {
                "data": {},
                "map_extractor": {},
                "metrics": {"test": {}},
                "batch_start": batch_start,
                "batch_end": batch_end,
            }
        )

        with (
            mock.patch(
                "match.run.instantiate",
                side_effect=[data, map_extractor, metric],
            ),
            mock.patch(
                "match.run.tqdm",
                side_effect=lambda iterable, **kwargs: iterable,
            ),
        ):
            match_run.main.__wrapped__(cfg)

        self.assertTrue(metric.finalized)
        self.assertEqual(map_extractor.processed, metric.pair_indices)
        return dataset.visited, metric.pair_indices

    def test_batch_bounds_select_inclusive_zero_based_pair_indices(self):
        visited, processed = self._run(batch_start=2, batch_end=4)

        self.assertEqual(processed, [2, 3, 4])
        self.assertEqual(visited, [0, 1, 2, 3, 4, 5])

    def test_each_batch_bound_is_optional(self):
        cases = (
            (None, None, [0, 1, 2, 3, 4, 5]),
            (3, None, [3, 4, 5]),
            (None, 2, [0, 1, 2]),
        )
        for batch_start, batch_end, expected in cases:
            with self.subTest(batch_start=batch_start, batch_end=batch_end):
                _, processed = self._run(batch_start, batch_end)
                self.assertEqual(processed, expected)

    def test_all_match_presets_expose_batch_bounds(self):
        config_dir = Path(__file__).resolve().parents[1] / "match" / "config"
        config_names = sorted(path.stem for path in config_dir.glob("*.yaml"))

        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir),
        ):
            for config_name in config_names:
                with self.subTest(config_name=config_name):
                    config = compose(
                        config_name=config_name,
                        overrides=["batch_start=1", "batch_end=2"],
                    )
                    self.assertEqual(config.batch_start, 1)
                    self.assertEqual(config.batch_end, 2)


if __name__ == "__main__":
    unittest.main()
