import torch

from match.map_extractor.partial_aware_soft_map_extractor import (
    PartialAwareSoftCanonizationMapExtractor,
)
from match.map_extractor.spectral_map_extractor import (
    _nn_argmin_chunked,
    _spectral_icp_refine,
)


def _initial_functional_map(p2p_n2m, evecs_m, evecs_trans_n, basis_size):
    """Project the valid entries of an n-to-m point map into an LBO basis."""
    if p2p_n2m.ndim != 1:
        raise ValueError(f"Expected a one-dimensional point map, got {p2p_n2m.shape}")
    if evecs_m.ndim != 2 or evecs_trans_n.ndim != 2:
        raise ValueError("Spectral eigenvectors must be two-dimensional")
    if len(p2p_n2m) != evecs_trans_n.shape[1]:
        raise ValueError(
            "Point-map source size does not match source eigenvectors: "
            f"{len(p2p_n2m)} vs {evecs_trans_n.shape[1]}"
        )
    if basis_size > evecs_m.shape[1] or basis_size > evecs_trans_n.shape[0]:
        raise ValueError(
            f"Requested {basis_size} eigenvectors, but only "
            f"{evecs_m.shape[1]} target and {evecs_trans_n.shape[0]} source "
            "eigenvectors are available"
        )

    valid = p2p_n2m >= 0
    if not torch.any(valid):
        raise ValueError(
            "Spectral refinement requires at least one valid initial correspondence"
        )
    valid_targets = p2p_n2m[valid]
    if torch.any(valid_targets >= evecs_m.shape[0]):
        raise ValueError("Initial point map contains an out-of-range target index")

    return evecs_trans_n[:basis_size, valid] @ evecs_m[valid_targets, :basis_size]


def _functional_map_to_point_map(
    functional_map,
    evecs_m,
    evecs_n,
    chunk_size,
):
    basis_size = functional_map.shape[0]
    mapped_m = evecs_m[:, :basis_size] @ functional_map.T
    return _nn_argmin_chunked(
        evecs_n[:, :basis_size],
        mapped_m,
        chunk_size,
    )


def _zoomout_refine(
    p2p_n2m,
    evecs_m,
    evecs_n,
    evecs_trans_n,
    mass_n,
    *,
    k_init,
    k_max,
    step,
    chunk_size,
):
    """Refine an arbitrary initial point map with the baseline ZoomOut updates."""
    available = min(evecs_m.shape[1], evecs_n.shape[1])
    if k_max > available:
        raise ValueError(
            f"ZoomOut requires {k_max} eigenvectors, but only {available} are "
            "available on both shapes"
        )
    if len(mass_n) != evecs_n.shape[0]:
        raise ValueError(
            "Source mass size does not match source eigenvectors: "
            f"{len(mass_n)} vs {evecs_n.shape[0]}"
        )

    functional_map = _initial_functional_map(
        p2p_n2m,
        evecs_m,
        evecs_trans_n,
        k_init,
    )
    current_basis_size = k_init

    for new_basis_size in range(k_init + step, k_max + 1, step):
        point_map = _functional_map_to_point_map(
            functional_map,
            evecs_m[:, :current_basis_size],
            evecs_n[:, :current_basis_size],
            chunk_size,
        )
        functional_map = (
            evecs_n[:, :new_basis_size] * mass_n.unsqueeze(1)
        ).T @ evecs_m[point_map, :new_basis_size]
        current_basis_size = new_basis_size

    return _functional_map_to_point_map(
        functional_map,
        evecs_m,
        evecs_n,
        chunk_size,
    )


class SpectrallyRefinedPartialAwareMapExtractor(
    PartialAwareSoftCanonizationMapExtractor
):
    """Optionally refine a partial-aware canonical map in the spectral domain."""

    REFINEMENT_METHODS = {"none", "spectral-icp", "zoomout"}

    def __init__(
        self,
        refinement_method: str = "none",
        spectral_icp_n_iters: int = 1,
        spectral_icp_chunk_size: int = 2048,
        zoomout_k_init: int = 20,
        zoomout_k_max: int = 200,
        zoomout_step: int = 5,
        zoomout_chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.refinement_method = str(refinement_method).lower()
        self.spectral_icp_n_iters = int(spectral_icp_n_iters)
        self.spectral_icp_chunk_size = int(spectral_icp_chunk_size)
        self.zoomout_k_init = int(zoomout_k_init)
        self.zoomout_k_max = int(zoomout_k_max)
        self.zoomout_step = int(zoomout_step)
        self.zoomout_chunk_size = int(zoomout_chunk_size)

        if self.refinement_method not in self.REFINEMENT_METHODS:
            valid = ", ".join(sorted(self.REFINEMENT_METHODS))
            raise ValueError(
                f"Unknown refinement_method {refinement_method!r}. Valid: {valid}"
            )
        if self.spectral_icp_n_iters <= 0:
            raise ValueError("spectral_icp_n_iters must be positive")
        if self.spectral_icp_chunk_size <= 0:
            raise ValueError("spectral_icp_chunk_size must be positive")
        if self.zoomout_k_init <= 0:
            raise ValueError("zoomout_k_init must be positive")
        if self.zoomout_k_max < self.zoomout_k_init:
            raise ValueError("zoomout_k_max must be at least zoomout_k_init")
        if self.zoomout_step <= 0:
            raise ValueError("zoomout_step must be positive")
        if (self.zoomout_k_max - self.zoomout_k_init) % self.zoomout_step:
            raise ValueError(
                "zoomout_step must evenly divide zoomout_k_max - zoomout_k_init"
            )
        if self.zoomout_chunk_size <= 0:
            raise ValueError("zoomout_chunk_size must be positive")

        if self.refinement_method != "none":
            self.method = f"{self.method}+{self.refinement_method}"

    @staticmethod
    def _spectral_tensors(shape_m, shape_n, *, require_mass):
        required_m = {"evecs"}
        required_n = {"evecs", "evecs_trans"}
        if require_mass:
            required_n.add("mass")
        missing_m = required_m.difference(shape_m)
        missing_n = required_n.difference(shape_n)
        if missing_m or missing_n:
            details = []
            if missing_m:
                details.append(f"target shape missing {sorted(missing_m)}")
            if missing_n:
                details.append(f"source shape missing {sorted(missing_n)}")
            raise KeyError(
                "Spectral refinement inputs unavailable: " + "; ".join(details)
            )

        evecs_m = shape_m["evecs"]
        evecs_n = shape_n["evecs"]
        evecs_trans_n = shape_n["evecs_trans"]
        if evecs_m.shape[1] != evecs_n.shape[1]:
            raise ValueError(
                "Spectral refinement requires the same number of eigenvectors on "
                f"both shapes, got {evecs_m.shape[1]} and {evecs_n.shape[1]}"
            )
        return evecs_m, evecs_n, evecs_trans_n

    def _refine(self, result):
        shape_m = result["shape_m"]
        shape_n = result["shape_n"]
        p2p_coarse = result["p2p"]
        require_mass = self.refinement_method == "zoomout"
        evecs_m, evecs_n, evecs_trans_n = self._spectral_tensors(
            shape_m,
            shape_n,
            require_mass=require_mass,
        )

        if self.refinement_method == "spectral-icp":
            functional_map = _initial_functional_map(
                p2p_coarse,
                evecs_m,
                evecs_trans_n,
                evecs_m.shape[1],
            )
            return _spectral_icp_refine(
                functional_map,
                evecs_m,
                evecs_n,
                n_iters=self.spectral_icp_n_iters,
                chunk_size=self.spectral_icp_chunk_size,
            )

        return _zoomout_refine(
            p2p_coarse,
            evecs_m,
            evecs_n,
            evecs_trans_n,
            shape_n["mass"],
            k_init=self.zoomout_k_init,
            k_max=self.zoomout_k_max,
            step=self.zoomout_step,
            chunk_size=self.zoomout_chunk_size,
        )

    def pipeline(self, shape_m, shape_n):
        result = super().pipeline(shape_m=shape_m, shape_n=shape_n)
        result["p2p_coarse"] = result["p2p"].clone()
        result["spectral_refinement"] = self.refinement_method
        result["method"] = self.method

        if self.refinement_method == "none":
            return result

        with torch.no_grad():
            result["p2p"] = self._refine(result)

        for name in ("soft_map", "soft_map_real"):
            if name in result:
                result[f"coarse_{name}"] = result.pop(name)
        return result
