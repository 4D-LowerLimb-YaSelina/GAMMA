import unittest
from pathlib import Path
from unittest import mock

import torch
from hydra import compose, initialize_config_dir

from match.map_extractor.partial_aware_soft_map_extractor import (
    PartialAwareSoftCanonizationMapExtractor,
)
from match.map_extractor.spectral_refined_partial_aware_map_extractor import (
    SpectrallyRefinedPartialAwareMapExtractor,
    _initial_functional_map,
    _zoomout_refine,
)
from match.map_extractor.spectral_map_extractor import _spectral_icp_refine


def _spectral_shape(vertex_count=3):
    eigenvectors = torch.eye(vertex_count)
    return {
        "evecs": eigenvectors,
        "evecs_trans": eigenvectors.T,
        "mass": torch.ones(vertex_count),
    }


def _coarse_result(vertex_count=3):
    indices = torch.arange(vertex_count)
    soft_real = torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(vertex_count),
        (vertex_count, vertex_count),
    ).coalesce()
    soft_with_null = torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        torch.ones(vertex_count),
        (vertex_count, vertex_count + 1),
    ).coalesce()
    return {
        "shape_m": _spectral_shape(vertex_count),
        "shape_n": _spectral_shape(vertex_count),
        "p2p": indices,
        "soft_map": soft_with_null,
        "soft_map_real": soft_real,
        "method": "full-coverage-soft-canonization",
    }


def _pipeline_shape(name):
    vertices = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    shape = {
        "name": name,
        "verts": vertices,
        "faces": torch.tensor([[0, 1, 2]]),
        "articulated_verts": vertices.clone(),
        "articulated_faces": torch.tensor([[0, 1, 2]]),
        "articulation_confidence": {
            "vertex_uncertainty": torch.zeros(3),
            "vertex_visibility_support": torch.ones(3),
        },
    }
    shape.update(_spectral_shape())
    return shape


class SpectralRefinementAlgorithmTests(unittest.TestCase):
    @staticmethod
    def _spectral_icp_unchunked(cmn, evecs_m, evecs_n, n_iters):
        phi_m = evecs_m.T
        phi_n = evecs_n.T
        functional_map = cmn.clone()

        for _ in range(n_iters):
            mapped_m = functional_map @ phi_m
            matches = torch.cdist(mapped_m.T, phi_n.T).argmin(dim=1)
            matched_n = phi_n[:, matches]
            left, _, right_t = torch.linalg.svd(matched_n @ phi_m.T)
            functional_map = left @ right_t

        mapped_m = evecs_m @ functional_map.T
        return torch.cdist(mapped_m, evecs_n).argmin(dim=-2)

    def test_spectral_icp_chunking_matches_unchunked_reference(self):
        generator = torch.Generator().manual_seed(7)
        evecs_m = torch.randn(7, 3, generator=generator)
        evecs_n = torch.randn(6, 3, generator=generator)
        functional_map = torch.randn(3, 3, generator=generator)

        expected = self._spectral_icp_unchunked(
            functional_map,
            evecs_m,
            evecs_n,
            n_iters=2,
        )
        actual = _spectral_icp_refine(
            functional_map,
            evecs_m,
            evecs_n,
            n_iters=2,
            chunk_size=2,
        )
        torch.testing.assert_close(actual, expected)

    def test_spectral_icp_chunks_iteration_and_final_discretization(self):
        eigenvectors = torch.eye(5)
        cdist_query_sizes = []
        original_cdist = torch.cdist

        def tracked_cdist(query, target):
            cdist_query_sizes.append(query.shape[0])
            return original_cdist(query, target)

        with mock.patch(
            "match.map_extractor.spectral_map_extractor.torch.cdist",
            side_effect=tracked_cdist,
        ):
            point_map = _spectral_icp_refine(
                torch.eye(5),
                eigenvectors,
                eigenvectors,
                n_iters=1,
                chunk_size=2,
            )

        torch.testing.assert_close(point_map, torch.arange(5))
        self.assertEqual(cdist_query_sizes, [2, 2, 1, 2, 2, 1])

    def test_spectral_icp_rejects_invalid_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
            _spectral_icp_refine(
                torch.eye(3),
                torch.eye(3),
                torch.eye(3),
                chunk_size=0,
            )

    def test_initial_functional_map_uses_only_valid_correspondences(self):
        eigenvectors = torch.eye(3)
        functional_map = _initial_functional_map(
            torch.tensor([0, -1, 2]),
            eigenvectors,
            eigenvectors.T,
            basis_size=3,
        )
        torch.testing.assert_close(
            functional_map,
            torch.diag(torch.tensor([1.0, 0.0, 1.0])),
        )

    def test_initial_functional_map_rejects_all_unmatched(self):
        eigenvectors = torch.eye(3)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            _initial_functional_map(
                torch.full((3,), -1),
                eigenvectors,
                eigenvectors.T,
                basis_size=3,
            )

    def test_zoomout_reaches_full_basis_and_returns_dense_map(self):
        eigenvectors = torch.eye(3)
        point_map = _zoomout_refine(
            torch.tensor([0, -1, 2]),
            eigenvectors,
            eigenvectors,
            eigenvectors.T,
            torch.ones(3),
            k_init=1,
            k_max=3,
            step=1,
            chunk_size=2,
        )
        self.assertEqual(point_map.shape, (3,))
        self.assertTrue(torch.all(point_map >= 0))
        torch.testing.assert_close(point_map, torch.arange(3))

    def test_zoomout_rejects_insufficient_basis(self):
        eigenvectors = torch.eye(3)
        with self.assertRaisesRegex(ValueError, "requires 4 eigenvectors"):
            _zoomout_refine(
                torch.arange(3),
                eigenvectors,
                eigenvectors,
                eigenvectors.T,
                torch.ones(3),
                k_init=2,
                k_max=4,
                step=2,
                chunk_size=2,
            )


class SpectrallyRefinedExtractorTests(unittest.TestCase):
    def test_none_preserves_soft_maps_and_coarse_result(self):
        coarse = _coarse_result()
        extractor = SpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            matching_mode="full",
            refinement_method="none",
        )
        with mock.patch.object(
            PartialAwareSoftCanonizationMapExtractor,
            "pipeline",
            return_value=coarse,
        ):
            result = extractor.pipeline({}, {})

        torch.testing.assert_close(result["p2p"], torch.arange(3))
        torch.testing.assert_close(result["p2p_coarse"], torch.arange(3))
        self.assertIn("soft_map", result)
        self.assertIn("soft_map_real", result)
        self.assertEqual(result["spectral_refinement"], "none")
        self.assertEqual(result["method"], "full-coverage-soft-canonization")

    def test_spectral_icp_replaces_p2p_and_renames_coarse_soft_maps(self):
        extractor = SpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            matching_mode="full",
            refinement_method="spectral-icp",
        )
        with mock.patch.object(
            PartialAwareSoftCanonizationMapExtractor,
            "pipeline",
            return_value=_coarse_result(),
        ):
            result = extractor.pipeline({}, {})

        torch.testing.assert_close(result["p2p"], torch.arange(3))
        torch.testing.assert_close(result["p2p_coarse"], torch.arange(3))
        self.assertNotIn("soft_map", result)
        self.assertNotIn("soft_map_real", result)
        self.assertIn("coarse_soft_map", result)
        self.assertIn("coarse_soft_map_real", result)
        self.assertEqual(result["spectral_refinement"], "spectral-icp")
        self.assertEqual(
            result["method"],
            "full-coverage-soft-canonization+spectral-icp",
        )

    def test_spectral_icp_accepts_partial_initializer_and_returns_dense_map(self):
        coarse = _coarse_result()
        coarse["p2p"] = torch.tensor([0, -1, 2])
        extractor = SpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            matching_mode="partial",
            refinement_method="spectral-icp",
        )
        with mock.patch.object(
            PartialAwareSoftCanonizationMapExtractor,
            "pipeline",
            return_value=coarse,
        ):
            result = extractor.pipeline({}, {})

        torch.testing.assert_close(result["p2p_coarse"], torch.tensor([0, -1, 2]))
        self.assertTrue(torch.all(result["p2p"] >= 0))

    def test_zoomout_accepts_partial_initializer_and_returns_dense_map(self):
        coarse = _coarse_result()
        coarse["p2p"] = torch.tensor([0, -1, 2])
        extractor = SpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            matching_mode="partial",
            refinement_method="zoomout",
            zoomout_k_init=1,
            zoomout_k_max=3,
            zoomout_step=1,
            zoomout_chunk_size=2,
        )
        with mock.patch.object(
            PartialAwareSoftCanonizationMapExtractor,
            "pipeline",
            return_value=coarse,
        ):
            result = extractor.pipeline({}, {})

        torch.testing.assert_close(result["p2p_coarse"], torch.tensor([0, -1, 2]))
        self.assertTrue(torch.all(result["p2p"] >= 0))
        torch.testing.assert_close(result["p2p"], torch.arange(3))

    def test_real_pipeline_refines_hard_map_without_retaining_soft_maps(self):
        extractor = SpectrallyRefinedPartialAwareMapExtractor(
            device="cpu",
            matching_mode="full",
            use_visibility_term=False,
            refinement_method="spectral-icp",
        )
        result = extractor.pipeline(
            _pipeline_shape("target"),
            _pipeline_shape("source"),
        )

        torch.testing.assert_close(result["p2p_coarse"], torch.arange(3))
        torch.testing.assert_close(result["p2p"], torch.arange(3))
        self.assertNotIn("coarse_soft_map", result)
        self.assertNotIn("coarse_soft_map_real", result)
        self.assertNotIn("soft_map", result)
        self.assertNotIn("soft_map_real", result)

    def test_constructor_validates_zoomout_schedule(self):
        with self.assertRaisesRegex(ValueError, "evenly divide"):
            SpectrallyRefinedPartialAwareMapExtractor(
                device="cpu",
                zoomout_k_init=2,
                zoomout_k_max=5,
                zoomout_step=2,
            )

    def test_constructor_validates_spectral_icp_chunk_size(self):
        with self.assertRaisesRegex(
            ValueError, "spectral_icp_chunk_size must be positive"
        ):
            SpectrallyRefinedPartialAwareMapExtractor(
                device="cpu",
                spectral_icp_chunk_size=0,
            )


class SpectralRefinementConfigTests(unittest.TestCase):
    def test_mhr_and_smal_variants_compose_for_every_method(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name in (
                "partial_aware_spectral",
                "partial_aware_smal_spectral",
            ):
                for refinement_method in ("none", "spectral-icp", "zoomout"):
                    with self.subTest(
                        config_name=config_name,
                        refinement_method=refinement_method,
                    ):
                        config = compose(
                            config_name=config_name,
                            overrides=[
                                "map_extractor.refinement_method="
                                f"{refinement_method}"
                            ],
                        )
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
                        self.assertTrue(config.data.return_mass)
                        self.assertTrue(config.data.return_evecs)
                        self.assertFalse(config.data.return_L)
                        self.assertFalse(config.data.return_grad)

    def test_smal_variant_keeps_smal_template_configuration(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "match" / "config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name="partial_aware_smal_spectral")
        self.assertEqual(
            config.map_extractor.smooth_kernel_template_type,
            "smal",
        )
        self.assertEqual(
            config.map_extractor.smooth_kernel_template_path,
            "pkg/AniMer/data/smal/my_smpl_00781_4_all.pkl",
        )


if __name__ == "__main__":
    unittest.main()
