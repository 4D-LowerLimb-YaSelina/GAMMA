import torch

from articulate.shape_refiner.dense_anchor_mhr_refiner import (
    DenseAnchorMHRShapeRefiner,
)
from articulate.shape_refiner.mhr_refiner import (
    bone_length_loss,
    chamfer_dist_subsample_mesh,
)


def _selected_parameter_rows(results, selected_results, *parameters):
    """Keep optimized parameter rows in the selected legacy view order."""
    result_keys = list(results)
    indices = [result_keys.index(key) for key in selected_results]
    return tuple(parameter[indices] for parameter in parameters)


class DenseAnchorLegacyChamferMHRShapeRefiner(DenseAnchorMHRShapeRefiner):
    """Dense-anchor initialization followed by the legacy MHR surface stage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._legacy_surface_target = None

    def _dense_surface_loss(
        self,
        results,
        mesh,
        shape_prior,
        identity_coeffs,
        vertices,
        keypoints,
    ):
        del identity_coeffs
        loss = chamfer_dist_subsample_mesh(
            verts_src=vertices,
            mesh_tgt=mesh,
            src_subsample_num=self.src_subsample_num,
            tgt_subsample_num=self.tgt_subsample_num,
            src_to_tgt=self.src_to_tgt,
            tgt_to_src=True,
            squared=True,
            reduction="mean",
        )
        target = self._legacy_surface_target
        if target is None:
            return loss

        keypoints_gt = torch.as_tensor(
            target,
            device=keypoints.device,
            dtype=keypoints.dtype,
        ).repeat(len(results), 1, 1)
        if self.bone_term_weight > 0.0:
            loss = loss + self.bone_term_weight * bone_length_loss(
                keypoints[:, :70],
                keypoints_gt,
                bone_pairs=shape_prior.get_skeleton(mode="all"),
                reduction="mean",
            )
        if self.keypoint_term_weight > 0.0:
            loss = loss + self.keypoint_term_weight * (
                (keypoints[:, :70] - keypoints_gt).square().sum(dim=2).mean()
            )
        return loss

    def shape_optim(
        self,
        results,
        mesh,
        X_hat,
        shape_prior,
        estimator,
        identity_coeffs,
        model_parameters,
        face_expr_coeffs,
        device="cuda",
    ):
        self._canonical_vertices = self._neutral_template(
            shape_prior,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
        )
        self._anchor_indices = self._load_or_create_anchors(
            self._canonical_vertices, estimator.faces
        )
        self._triangulate_anchors(results, mesh)
        R, t, s = self.stack_alignment_params(
            results, device, dtype=identity_coeffs.dtype
        )
        with torch.no_grad():
            post_joint_vertices = (
                shape_prior.generate(
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                    R,
                    s,
                    t,
                    return_keypoints=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
        self._prepare_grounding(mesh, post_joint_vertices, estimator.faces)
        identity_coeffs, model_parameters, face_expr_coeffs = (
            self._sparse_anchor_optim(
                results,
                shape_prior,
                identity_coeffs,
                model_parameters,
                face_expr_coeffs,
                device,
            )
        )

        results_for_surface = results
        if 0 < self.n_views_surface < len(results):
            with torch.no_grad():
                vertices = shape_prior.generate(
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                    R,
                    s,
                    t,
                    return_keypoints=False,
                )
            results_for_surface = self.select_best_views_by_chamfer(
                results=results,
                vertices_optimized=vertices,
                mesh_target=mesh,
                estimator=estimator,
                n_views=self.n_views_surface,
            )
            identity_coeffs, model_parameters, face_expr_coeffs = (
                _selected_parameter_rows(
                    results,
                    results_for_surface,
                    identity_coeffs,
                    model_parameters,
                    face_expr_coeffs,
                )
            )

        self._legacy_surface_target = X_hat
        identity_coeffs, model_parameters, face_expr_coeffs = self._dense_optim(
            results_for_surface,
            mesh,
            shape_prior,
            estimator,
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            device,
        )
        return (
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            results_for_surface,
        )
