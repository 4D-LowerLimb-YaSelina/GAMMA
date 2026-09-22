import torch

from articulate.shape_refiner.dense_anchor_smal_refiner import (
    DenseAnchorPRIMASMALShapeRefiner,
    DenseAnchorSMALShapeRefiner,
)
from articulate.shape_refiner.smal_refiner import (
    bone_length_loss,
    chamfer_dist_subsample_mesh,
)


def _selected_parameter_rows(results, selected_results, *parameters):
    """Keep optimized parameter rows in the selected legacy view order."""
    result_keys = list(results)
    indices = [result_keys.index(key) for key in selected_results]
    return tuple(parameter[indices] for parameter in parameters)


class _DenseAnchorLegacyChamferSMALMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._legacy_surface_target = None

    def _dense_surface_loss(
        self,
        results,
        mesh,
        shape_prior,
        betas,
        vertices,
        keypoints,
    ):
        del betas
        loss = chamfer_dist_subsample_mesh(
            vertices,
            mesh,
            src_subsample_num=self.src_subsample_num,
            tgt_subsample_num=self.tgt_subsample_num,
            src_to_tgt=self.src_to_tgt,
            tgt_to_src=True,
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
                keypoints,
                keypoints_gt,
                shape_prior.get_skeleton(),
            )
        if self.keypoint_term_weight > 0.0:
            loss = loss + self.keypoint_term_weight * (
                (keypoints - keypoints_gt).square().sum(dim=2).mean()
            )
        return loss

    def shape_optim(
        self,
        results,
        mesh,
        triangulated,
        shape_prior,
        betas,
        global_orient,
        pose,
    ):
        self._canonical_vertices = self._neutral_template(
            shape_prior, betas, global_orient, pose
        )
        self._anchor_indices = self._load_or_create_anchors(
            self._canonical_vertices, shape_prior.faces
        )
        self._triangulate_anchors(results, mesh)
        rotation, translation, scale = self.stack_alignment_params(
            results, dtype=betas.dtype
        )
        with torch.no_grad():
            post_joint_vertices, _ = shape_prior.generate(
                betas,
                global_orient,
                pose,
                rotation,
                scale,
                translation,
            )
        self._prepare_grounding(
            mesh,
            post_joint_vertices.detach().cpu().numpy(),
            shape_prior.faces,
        )
        betas, global_orient, pose = self._sparse_anchor_optim(
            results, shape_prior, betas, global_orient, pose
        )

        results_for_surface = results
        if 0 < self.n_views_surface < len(results):
            with torch.no_grad():
                vertices, _ = shape_prior.generate(
                    betas,
                    global_orient,
                    pose,
                    rotation,
                    scale,
                    translation,
                )
            results_for_surface = self.select_best_views(
                results, vertices, mesh, shape_prior.faces
            )
            betas, global_orient, pose = _selected_parameter_rows(
                results,
                results_for_surface,
                betas,
                global_orient,
                pose,
            )

        self._legacy_surface_target = triangulated
        betas, global_orient, pose = self._dense_optim(
            results_for_surface,
            mesh,
            shape_prior,
            betas,
            global_orient,
            pose,
        )
        return betas, global_orient, pose, results_for_surface


class DenseAnchorLegacyChamferSMALShapeRefiner(
    _DenseAnchorLegacyChamferSMALMixin, DenseAnchorSMALShapeRefiner
):
    """AniMer anchors followed by its legacy Chamfer surface stage."""


class DenseAnchorLegacyChamferPRIMASMALShapeRefiner(
    _DenseAnchorLegacyChamferSMALMixin, DenseAnchorPRIMASMALShapeRefiner
):
    """PRIMA anchors followed by its legacy Chamfer surface stage."""
