import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse
import torch
import trimesh
from hydra import compose, initialize_config_dir

from articulate.output_handler.uncertainty_aware_output_handler import (
    UncertaintyAwareOutputHandler,
)
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
    _softmax_negative,
    confidence_weighted_fusion,
    vertex_visibility_from_depth,
)
from match.data.articulated_shape_corr import ShapeDatasetFast
from match.map_extractor.partial_aware_soft_map_extractor import (
    PartialAwareSoftCanonizationMapExtractor,
    build_template_smooth_kernel,
    canonicalization_confidence,
    compose_soft_map,
    dense_canonical_fallback_map,
    hard_map_from_soft,
    load_smooth_kernel_cache,
    mesh_vertex_area_weights,
    parameterized_smooth_kernel_cache_path,
    point_to_triangle_assignment,
    reciprocal_consistency,
    reverse_canonical_map,
    save_smooth_kernel_cache,
)
from match.metric.geodist import GeodistMetric

TRIANGLE_VERTICES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
TRIANGLE_FACES = np.array([[0, 1, 2]])


class UncertaintyAwareComponentTests(unittest.TestCase):
    def test_negative_softmax_prefers_small_residual(self):
        weights = _softmax_negative(np.array([0.0, 1.0]), temperature=1.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])
        np.testing.assert_allclose(
            _softmax_negative(np.array([np.inf, np.inf]), temperature=1.0),
            [0.5, 0.5],
        )

    def test_visibility_uses_z_buffer(self):
        vertices = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
        depth = np.zeros((3, 3), dtype=float)
        depth[1, 1] = 1.0
        camera = {
            "R_wc": np.eye(3),
            "t_wc": np.zeros(3),
            "intrinsics": {
                "focal_length": 1.0,
                "camera_center": (1.0, 1.0),
            },
        }
        visibility = vertex_visibility_from_depth(
            vertices,
            depth,
            camera,
            depth_tolerance=1e-6,
            pixel_radius=0,
        )
        np.testing.assert_array_equal(visibility, [True, False])

    def test_weighted_fusion_falls_back_for_unobserved_vertex(self):
        vertices = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )
        fused, uncertainty, support, fallback = confidence_weighted_fusion(
            vertices,
            view_weights=np.array([0.75, 0.25]),
            visibility=np.array([[1, 0], [0, 0]]),
        )
        np.testing.assert_allclose(fused[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(uncertainty, [0.0, 3.0])
        np.testing.assert_allclose(support, [0.75, 0.0])
        np.testing.assert_array_equal(fallback, [False, True])

    def test_weighted_fusion_can_ignore_vertex_visibility(self):
        vertices = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )
        fused, uncertainty, support, fallback = confidence_weighted_fusion(
            vertices,
            view_weights=np.array([0.75, 0.25]),
            visibility=np.array([[1, 0], [0, 0]]),
            apply_visibility=False,
        )
        np.testing.assert_allclose(fused[:, 0], [0.5, 1.0])
        np.testing.assert_allclose(uncertainty, [0.75, 3.0])
        np.testing.assert_allclose(support, [1.0, 1.0])
        np.testing.assert_array_equal(fallback, [False, False])

    def test_uncertainty_aware_presets_disable_visibility_weighting(self):
        config_dir = str(Path(__file__).parents[1] / "articulate/config")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            configs = [
                compose(config_name=config_name)
                for config_name in (
                    "uncertainty_aware",
                    "uncertainty_aware_animer",
                    "uncertainty_aware_prima",
                )
            ]
        for config in configs:
            self.assertFalse(config.shape_refiner.use_visibility_in_vertex_confidence)

    def test_barycentric_assignment_has_three_weights(self):
        point = np.array([[0.25, 0.25, 1.0]])
        assignment = point_to_triangle_assignment(
            point, TRIANGLE_VERTICES, TRIANGLE_FACES
        )
        matrix, closest, distances, triangle_ids, barycentric = assignment
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(matrix.nnz, 3)
        np.testing.assert_allclose(matrix.toarray(), [[0.5, 0.25, 0.25]])
        np.testing.assert_allclose(closest, point)
        np.testing.assert_allclose(distances, 0.0)
        np.testing.assert_array_equal(triangle_ids, [0])
        np.testing.assert_allclose(barycentric.sum(axis=1), 1.0)

    def test_full_canonicalization_confidence(self):
        assignment = point_to_triangle_assignment(
            TRIANGLE_VERTICES, TRIANGLE_VERTICES, TRIANGLE_FACES
        )
        matrix, _, distances, triangle_ids, barycentric = assignment
        diagnostics = canonicalization_confidence(
            raw_vertices=TRIANGLE_VERTICES,
            raw_faces=TRIANGLE_FACES,
            proxy_vertices=TRIANGLE_VERTICES,
            proxy_faces=TRIANGLE_FACES,
            assignment=matrix,
            distances=distances,
            triangle_ids=triangle_ids,
            barycentric=barycentric,
            vertex_uncertainty=np.zeros(3),
            vertex_visibility_support=np.ones(3),
            distance_scale=0.02,
            uncertainty_scale=0.02,
        )
        np.testing.assert_allclose(diagnostics["confidence"], 1.0)
        np.testing.assert_allclose(diagnostics["normal_term"], 1.0)

    def test_canonicalization_can_ignore_visibility(self):
        assignment = point_to_triangle_assignment(
            TRIANGLE_VERTICES, TRIANGLE_VERTICES, TRIANGLE_FACES
        )
        matrix, _, distances, triangle_ids, barycentric = assignment
        diagnostics = canonicalization_confidence(
            raw_vertices=TRIANGLE_VERTICES,
            raw_faces=TRIANGLE_FACES,
            proxy_vertices=TRIANGLE_VERTICES,
            proxy_faces=TRIANGLE_FACES,
            assignment=matrix,
            distances=distances,
            triangle_ids=triangle_ids,
            barycentric=barycentric,
            vertex_uncertainty=np.zeros(3),
            vertex_visibility_support=np.zeros(3),
            distance_scale=0.02,
            uncertainty_scale=0.02,
            use_visibility_term=False,
        )
        np.testing.assert_allclose(diagnostics["visibility_support"], 0.0)
        np.testing.assert_allclose(diagnostics["visibility_term"], 1.0)
        np.testing.assert_allclose(diagnostics["confidence"], 1.0)

    def test_reverse_map_uses_area_confidence_evidence(self):
        assignment = scipy.sparse.eye(3, format="csr")
        area_weights, total_area = mesh_vertex_area_weights(
            TRIANGLE_VERTICES, TRIANGLE_FACES
        )
        reverse, evidence, occupancy, kappa = reverse_canonical_map(
            assignment,
            area_weights,
            confidence=np.ones(3),
            total_surface_area=total_area,
            kappa_factor=1.0,
            eps=1e-12,
        )
        expected_kappa = total_area / 3.0
        np.testing.assert_allclose(evidence, area_weights)
        self.assertAlmostEqual(kappa, expected_kappa)
        np.testing.assert_allclose(occupancy, 1.0 - np.exp(-1.0))
        np.testing.assert_allclose(reverse.toarray().sum(axis=1), 1.0, atol=1e-11)

    def test_hard_map_emits_minus_one_when_null_wins(self):
        soft = scipy.sparse.csr_matrix([[0.8, 0.1, 0.1], [0.2, 0.1, 0.7]])
        point_map, best, null = hard_map_from_soft(soft, target_vertex_count=2)
        np.testing.assert_array_equal(point_map, [0, -1])
        np.testing.assert_allclose(best, [0.8, 0.2])
        np.testing.assert_allclose(null, [0.1, 0.7])

    def test_dense_canonical_fallback_uses_shared_topology(self):
        target_vertices = TRIANGLE_VERTICES[[2, 0, 1]]
        fallback = dense_canonical_fallback_map(
            TRIANGLE_VERTICES,
            TRIANGLE_VERTICES,
            TRIANGLE_VERTICES,
            target_vertices,
        )
        np.testing.assert_array_equal(fallback, [1, 2, 0])

    def test_smooth_kernel_uses_geodesic_gaussian_and_pruning(self):
        kernel, diagnostics = build_template_smooth_kernel(
            TRIANGLE_VERTICES,
            TRIANGLE_FACES,
            sigma_factor=1.0,
            min_weight=0.1,
            batch_size=2,
        )
        np.testing.assert_allclose(kernel.toarray().sum(axis=1), 1.0, atol=2e-7)
        expected_ratio = np.exp(-1.0 / (2.0 * diagnostics["sigma"] ** 2))
        self.assertAlmostEqual(
            float(kernel[0, 1] / kernel[0, 0]), expected_ratio, places=6
        )

        pruned, _ = build_template_smooth_kernel(
            TRIANGLE_VERTICES,
            TRIANGLE_FACES,
            sigma_factor=0.1,
            min_weight=0.5,
            batch_size=2,
        )
        np.testing.assert_allclose(pruned.toarray(), np.eye(3))

    def test_sparse_smooth_kernel_cache_round_trip_at_npy_path(self):
        kernel, diagnostics = build_template_smooth_kernel(
            TRIANGLE_VERTICES,
            TRIANGLE_FACES,
            sigma_factor=1.0,
            min_weight=0.1,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "MHR_smooth_kernel.npy"
            model_path = Path(temp_dir) / "mhr_model.pt"
            save_smooth_kernel_cache(
                cache_path,
                kernel,
                sigma_factor=1.0,
                min_weight=0.1,
                template_diagonal=diagnostics["template_diagonal"],
                mhr_model_path=model_path,
            )
            self.assertTrue(cache_path.is_file())
            loaded, metadata = load_smooth_kernel_cache(cache_path)
            np.testing.assert_allclose(loaded.toarray(), kernel.toarray())
            self.assertEqual(metadata["mhr_model_path"], str(model_path))
            self.assertEqual(metadata["sigma_factor"], 1.0)
            self.assertEqual(metadata["min_weight"], 0.1)

    def test_smooth_kernel_cache_path_encodes_kernel_parameters(self):
        base_path = Path("output/cache/MHR_smooth_kernel.npy")
        first = parameterized_smooth_kernel_cache_path(
            base_path, sigma_factor=0.01, min_weight=1e-3
        )
        second = parameterized_smooth_kernel_cache_path(
            base_path, sigma_factor=0.005, min_weight=1e-2
        )
        self.assertEqual(
            first,
            Path("output/cache/MHR_smooth_kernel-sigma0p01-min0p001.npy"),
        )
        self.assertNotEqual(first, second)

    def test_extractor_reuses_valid_cache_without_loading_mhr(self):
        kernel, diagnostics = build_template_smooth_kernel(
            TRIANGLE_VERTICES,
            TRIANGLE_FACES,
            sigma_factor=1.0,
            min_weight=0.1,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "MHR_smooth_kernel.npy"
            missing_model_path = root / "missing_mhr_model.pt"
            extractor = PartialAwareSoftCanonizationMapExtractor(
                device="cpu",
                use_smooth_kernel=True,
                smooth_kernel_cache_path=cache_path,
                smooth_kernel_mhr_path=missing_model_path,
                smooth_kernel_sigma_factor=1.0,
                smooth_kernel_min_weight=0.1,
            )
            save_smooth_kernel_cache(
                extractor.smooth_kernel_cache_path,
                kernel,
                sigma_factor=1.0,
                min_weight=0.1,
                template_diagonal=diagnostics["template_diagonal"],
                mhr_model_path=missing_model_path,
            )
            reused = extractor._get_smooth_kernel(3)
            np.testing.assert_allclose(reused.toarray(), kernel.toarray())

    def test_extractor_rejects_incompatible_cached_topology(self):
        kernel = scipy.sparse.eye(2, format="csr", dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_path = root / "MHR_smooth_kernel.npy"
            model_path = root / "missing_mhr_model.pt"
            extractor = PartialAwareSoftCanonizationMapExtractor(
                device="cpu",
                use_smooth_kernel=True,
                smooth_kernel_cache_path=cache_path,
                smooth_kernel_mhr_path=model_path,
                smooth_kernel_sigma_factor=1.0,
                smooth_kernel_min_weight=0.1,
            )
            save_smooth_kernel_cache(
                extractor.smooth_kernel_cache_path,
                kernel,
                sigma_factor=1.0,
                min_weight=0.1,
                template_diagonal=1.0,
                mhr_model_path=model_path,
            )
            with self.assertRaisesRegex(ValueError, "topology"):
                extractor._get_smooth_kernel(3)

    def test_smooth_composition_inserts_kernel(self):
        assignment = scipy.sparse.eye(3, format="csr")
        reverse = scipy.sparse.hstack(
            [scipy.sparse.eye(3, format="csr"), scipy.sparse.csr_matrix((3, 1))],
            format="csr",
        )
        kernel = scipy.sparse.csr_matrix(
            [[0.75, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        baseline = compose_soft_map(assignment, reverse)
        smoothed = compose_soft_map(assignment, reverse, kernel)
        np.testing.assert_allclose(baseline.toarray(), np.c_[np.eye(3), np.zeros(3)])
        np.testing.assert_allclose(
            smoothed.toarray(), np.c_[kernel.toarray(), np.zeros(3)]
        )

    def test_reciprocal_consistency_rejects_many_to_one_match(self):
        forward = scipy.sparse.csr_matrix(
            [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.8, 0.1]]
        )
        backward = scipy.sparse.csr_matrix([[0.9, 0.1, 0.0, 0.0], [0.0, 0.2, 0.7, 0.1]])
        forward_p2p, _, _ = hard_map_from_soft(forward, target_vertex_count=2)
        backward_p2p, _, _ = hard_map_from_soft(backward, target_vertex_count=3)
        filtered, consistent, scores, selected = reciprocal_consistency(
            forward,
            backward,
            forward_p2p,
            backward_p2p,
            forward_target_vertex_count=2,
            backward_target_vertex_count=3,
        )
        np.testing.assert_array_equal(forward_p2p, [0, 0, 1])
        np.testing.assert_array_equal(backward_p2p, [0, 2])
        np.testing.assert_array_equal(filtered, [0, -1, 1])
        np.testing.assert_array_equal(consistent, [True, False, True])
        self.assertAlmostEqual(float(scores[0, 0]), 0.9)
        self.assertAlmostEqual(float(selected[1]), np.sqrt(0.08))

    def test_partial_aware_pipeline_composes_target_direction(self):
        vertices = torch.tensor(TRIANGLE_VERTICES, dtype=torch.float32)
        faces = torch.tensor(TRIANGLE_FACES, dtype=torch.long)

        def shape(name):
            return {
                "name": name,
                "verts": vertices.clone(),
                "faces": faces.clone(),
                "articulated_verts": vertices.clone(),
                "articulated_faces": faces.clone(),
                "articulation_confidence": {
                    "vertex_uncertainty": torch.zeros(3),
                    "vertex_visibility_support": torch.ones(3),
                },
            }

        result = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            return_soft_maps=True,
        ).pipeline(shape("source"), shape("target"))
        torch.testing.assert_close(result["p2p"], torch.arange(3))
        self.assertTrue(result["soft_map"].is_sparse)
        self.assertEqual(result["soft_map"].shape, (3, 4))
        self.assertEqual(result["soft_map_real"].shape, (3, 3))
        self.assertTrue(result["visibility_term_enabled"])
        self.assertFalse(result["smooth_kernel_enabled"])
        self.assertFalse(result["reciprocal_consistency_enabled"])

    def test_full_matching_mode_falls_back_to_dense_map(self):
        vertices = torch.tensor(TRIANGLE_VERTICES, dtype=torch.float32)
        faces = torch.tensor(TRIANGLE_FACES, dtype=torch.long)

        def shape(name):
            return {
                "name": name,
                "verts": vertices.clone(),
                "faces": faces.clone(),
                "articulated_verts": vertices.clone(),
                "articulated_faces": faces.clone(),
                "articulation_confidence": {
                    "vertex_uncertainty": torch.zeros(3),
                    "vertex_visibility_support": torch.zeros(3),
                },
            }

        result = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            matching_mode="full",
            use_visibility_term=True,
        ).pipeline(shape("target"), shape("source"))
        torch.testing.assert_close(result["p2p"], torch.arange(3))
        self.assertTrue(torch.all(result["dense_fallback_used"]))
        self.assertTrue(torch.all(result["target_occupancy"] == 1))
        self.assertTrue(torch.all(result["null_probability"] == 0))
        self.assertEqual(result["matching_mode"], "full")
        self.assertNotIn("soft_map", result)
        self.assertNotIn("soft_map_real", result)

    def test_reciprocal_pipeline_preserves_forward_soft_map(self):
        vertices = torch.tensor(TRIANGLE_VERTICES, dtype=torch.float32)
        faces = torch.tensor(TRIANGLE_FACES, dtype=torch.long)

        def shape(name):
            return {
                "name": name,
                "verts": vertices.clone(),
                "faces": faces.clone(),
                "articulated_verts": vertices.clone(),
                "articulated_faces": faces.clone(),
                "articulation_confidence": {
                    "vertex_uncertainty": torch.zeros(3),
                    "vertex_visibility_support": torch.ones(3),
                },
            }

        baseline = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            return_soft_maps=True,
        ).pipeline(shape("target"), shape("source"))
        reciprocal = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            use_reciprocal_consistency=True,
            return_soft_maps=True,
        ).pipeline(shape("target"), shape("source"))
        torch.testing.assert_close(
            baseline["soft_map"].to_dense(), reciprocal["soft_map"].to_dense()
        )
        torch.testing.assert_close(reciprocal["p2p"], torch.arange(3))
        torch.testing.assert_close(reciprocal["p2p_pre_reciprocal"], torch.arange(3))
        self.assertTrue(torch.all(reciprocal["reciprocal_consistent"]))
        self.assertTrue(reciprocal["reciprocal_score_map"].is_sparse)

        lean_reciprocal = PartialAwareSoftCanonizationMapExtractor(
            device="cpu",
            use_reciprocal_consistency=True,
        ).pipeline(shape("target"), shape("source"))
        torch.testing.assert_close(lean_reciprocal["p2p"], reciprocal["p2p"])
        self.assertNotIn("soft_map", lean_reciprocal)
        self.assertNotIn("soft_map_real", lean_reciprocal)
        self.assertNotIn("reciprocal_score_map", lean_reciprocal)

    def test_confidence_sidecar_round_trip(self):
        confidence = {
            "schema_version": np.asarray(1),
            "vertex_uncertainty": np.array([0.1, 0.2]),
            "vertex_visibility_support": np.array([1.0, 0.5]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = UncertaintyAwareOutputHandler(out_dir=temp_dir)
            path = handler.store_confidence("person", confidence)
            self.assertEqual(path.name, "person.confidence.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["vertex_uncertainty"], [0.1, 0.2])

    def test_dataset_loads_raw_mesh_without_spectral_precomputation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data" / "off").mkdir(parents=True)
            (root / "articulated" / "mhr").mkdir(parents=True)
            mesh = trimesh.Trimesh(
                vertices=TRIANGLE_VERTICES,
                faces=TRIANGLE_FACES,
                process=False,
            )
            mesh.export(root / "data" / "off" / "person.off")
            mesh.export(root / "articulated" / "mhr" / "person.obj")
            np.savez_compressed(
                root / "articulated" / "mhr" / "person.confidence.npz",
                vertex_uncertainty=np.zeros(3),
                vertex_visibility_support=np.ones(3),
            )

            dataset = ShapeDatasetFast(
                data_root=root / "data",
                articulated_shape_dir=root / "articulated",
                return_faces=True,
                return_L=False,
                return_mass=False,
                return_evecs=False,
                return_grad=False,
                return_articulation_confidence=True,
            )
            item = dataset[0]
            self.assertFalse((root / "data" / "spectral").exists())
            self.assertEqual(item["verts"].shape, (3, 3))
            self.assertEqual(item["articulated_faces"].shape, (1, 3))
            torch.testing.assert_close(
                item["articulation_confidence"]["vertex_uncertainty"],
                torch.zeros(3),
            )

    def test_mocked_refiner_lifecycle_returns_confidence(self):
        class FakePrior:
            def generate(
                self,
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                R,
                s,
                t,
                return_keypoints=False,
            ):
                base = torch.tensor(
                    TRIANGLE_VERTICES,
                    dtype=identity_coeffs.dtype,
                    device=identity_coeffs.device,
                )
                vertices = base.unsqueeze(0).repeat(len(identity_coeffs), 1, 1)
                if not return_keypoints:
                    return vertices
                keypoints = torch.zeros(
                    len(identity_coeffs), 70, 3, dtype=vertices.dtype
                )
                return vertices, keypoints, None

        class TestRefiner(UncertaintyAwareMHRShapeRefiner):
            def _render_visibility(self, vertices, results, faces, body_diagonal):
                return np.ones(vertices.shape[:2], dtype=bool)

        result = {
            "sam3d_output": {
                "shape_params": np.zeros(4),
                "mhr_model_params": np.zeros(6),
                "expr_params": np.zeros(3),
            },
            "alignment": {"R": np.eye(3), "t": np.zeros(3), "s": 1.0},
            "confidence": {"alignment_mse": 0.0, "inlier_ratio": 1.0},
            "iou": 1.0,
        }
        shape = {
            "name": "person",
            "path": "person.obj",
            "mesh": trimesh.Trimesh(
                vertices=TRIANGLE_VERTICES,
                faces=TRIANGLE_FACES,
                process=False,
            ),
        }
        estimation = {
            "shape": shape,
            "results": {(0.0, 0.0): result, (180.0, 0.0): result},
            "camera_configs": {"h": 512},
            "axis_order": "xyz",
            "estimator": SimpleNamespace(faces=TRIANGLE_FACES),
        }
        refiner = TestRefiner(
            device="cpu",
            optimize_keypoints=False,
            optimize_surface=False,
            merge_method="confidence_weighted",
        )
        self.assertFalse(refiner.use_visibility_in_vertex_confidence)
        refined = refiner.pipeline(estimation, FakePrior())
        self.assertIn("confidence", refined)
        np.testing.assert_allclose(refined["confidence"]["view_weights"], [0.5, 0.5])
        np.testing.assert_allclose(refined["confidence"]["vertex_uncertainty"], 0.0)

    def test_geodist_masks_null_and_reports_coverage(self):
        metric = GeodistMetric()
        shape_m = {
            "name": "m",
            "dist": torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
            "corr": torch.arange(3),
        }
        shape_n = {"name": "n", "corr": torch.arange(3)}
        output = metric.pipeline(
            {
                "shape_m": shape_m,
                "shape_n": shape_n,
                "p2p": torch.tensor([0, -1, 2]),
                "method": "partial-aware-soft-canonization",
            }
        )
        self.assertIn("coverage=0.667", output)
        self.assertEqual(metric.coverages, [2 / 3])
        self.assertEqual(metric.values, [0.0])

        unmatched_metric = GeodistMetric()
        unmatched_output = unmatched_metric.pipeline(
            {
                "shape_m": shape_m,
                "shape_n": shape_n,
                "p2p": torch.full((3,), -1),
                "method": "partial-aware-soft-canonization",
            }
        )
        self.assertIn("coverage=0.000", unmatched_output)
        self.assertEqual(unmatched_metric.values, [])
        self.assertEqual(unmatched_metric.coverages, [0.0])


if __name__ == "__main__":
    unittest.main()
