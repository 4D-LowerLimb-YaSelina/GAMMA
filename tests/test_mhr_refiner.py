import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import trimesh

from articulate.shape_refiner.mhr_refiner import MHRShapeRefiner
from articulate.shape_refiner.uncertainty_aware_mhr_refiner import (
    UncertaintyAwareMHRShapeRefiner,
)


class MHRParameterHandoffTests(unittest.TestCase):
    refiner_types = (MHRShapeRefiner, UncertaintyAwareMHRShapeRefiner)

    @staticmethod
    def _results(count):
        results = {}
        for index in range(count):
            results[f"view_{index}"] = {
                "sam3d_output": {
                    "shape_params": np.full(2, 100.0 + index),
                    "mhr_model_params": np.full(3, 200.0 + index),
                    "expr_params": np.full(1, 300.0 + index),
                }
            }
        return results

    @staticmethod
    def _parameters(count):
        identity = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
        model = (
            torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) + 20.0
        ).requires_grad_()
        expression = torch.arange(count, dtype=torch.float32).reshape(count, 1) + 40.0
        return identity, model, expression

    @staticmethod
    def _alignment(count):
        return (
            torch.eye(3).repeat(count, 1, 1),
            torch.zeros(count, 3),
            torch.ones(count),
        )

    def test_selected_views_preserve_current_parameters_in_selected_order(self):
        results = self._results(4)
        selected_results = {
            "view_2": results["view_2"],
            "view_0": results["view_0"],
            "view_3": results["view_3"],
        }
        selected_indices = torch.tensor([2, 0, 3])
        mesh = trimesh.Trimesh(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]), process=False
        )

        for refiner_type in self.refiner_types:
            with self.subTest(refiner=refiner_type.__name__):
                refiner = refiner_type(device="cpu", n_views_surface=3)
                identity, model, expression = self._parameters(4)
                captured = {}

                def generate(*_args, **_kwargs):
                    return torch.zeros(4, 3, 3)

                def optimize_surface(**kwargs):
                    captured.update(kwargs)
                    return (
                        kwargs["identity_coeffs"],
                        kwargs["model_parameters"],
                        kwargs["face_expr_coeffs"],
                    )

                with (
                    mock.patch.object(
                        refiner,
                        "stack_alignment_params",
                        return_value=self._alignment(4),
                    ),
                    mock.patch.object(
                        refiner,
                        "select_best_views_by_chamfer",
                        return_value=selected_results,
                    ),
                    mock.patch.object(
                        refiner,
                        "stack_mhr_params",
                        side_effect=AssertionError(
                            "surface selection must not reload estimator parameters"
                        ),
                    ) as stack_mhr_params,
                    mock.patch.object(
                        refiner,
                        "_optimize_mhr_surface",
                        side_effect=optimize_surface,
                    ),
                ):
                    output = refiner.shape_optim(
                        results=results,
                        mesh=mesh,
                        X_hat=np.zeros((70, 3), dtype=np.float32),
                        shape_prior=SimpleNamespace(generate=generate),
                        estimator=SimpleNamespace(faces=mesh.faces),
                        identity_coeffs=identity,
                        model_parameters=model,
                        face_expr_coeffs=expression,
                        device="cpu",
                    )

                stack_mhr_params.assert_not_called()
                self.assertEqual(list(captured["results"]), list(selected_results))
                self.assertEqual(list(output[3]), list(selected_results))
                torch.testing.assert_close(
                    captured["identity_coeffs"],
                    identity.index_select(0, selected_indices),
                )
                torch.testing.assert_close(
                    captured["model_parameters"],
                    model.index_select(0, selected_indices),
                )
                torch.testing.assert_close(
                    captured["face_expr_coeffs"],
                    expression.index_select(0, selected_indices),
                )
                for name in (
                    "identity_coeffs",
                    "model_parameters",
                    "face_expr_coeffs",
                ):
                    parameter = captured[name]
                    self.assertEqual(parameter.shape[0], len(selected_results))
                    self.assertTrue(parameter.is_leaf)
                    self.assertFalse(parameter.requires_grad)

    def test_no_selection_preserves_all_current_parameters(self):
        results = self._results(3)
        mesh = trimesh.Trimesh(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]), process=False
        )

        for refiner_type in self.refiner_types:
            with self.subTest(refiner=refiner_type.__name__):
                refiner = refiner_type(device="cpu", n_views_surface=3)
                identity, model, expression = self._parameters(3)
                captured = {}

                def optimize_surface(**kwargs):
                    captured.update(kwargs)
                    return (
                        kwargs["identity_coeffs"],
                        kwargs["model_parameters"],
                        kwargs["face_expr_coeffs"],
                    )

                with (
                    mock.patch.object(
                        refiner,
                        "select_best_views_by_chamfer",
                        side_effect=AssertionError("view selection should be skipped"),
                    ),
                    mock.patch.object(
                        refiner,
                        "_optimize_mhr_surface",
                        side_effect=optimize_surface,
                    ),
                ):
                    output = refiner.shape_optim(
                        results=results,
                        mesh=mesh,
                        X_hat=np.zeros((70, 3), dtype=np.float32),
                        shape_prior=mock.sentinel.shape_prior,
                        estimator=SimpleNamespace(faces=mesh.faces),
                        identity_coeffs=identity,
                        model_parameters=model,
                        face_expr_coeffs=expression,
                        device="cpu",
                    )

                self.assertIs(captured["identity_coeffs"], identity)
                self.assertIs(captured["model_parameters"], model)
                self.assertIs(captured["face_expr_coeffs"], expression)
                self.assertEqual(list(captured["results"]), list(results))
                self.assertEqual(list(output[3]), list(results))


if __name__ == "__main__":
    unittest.main()
