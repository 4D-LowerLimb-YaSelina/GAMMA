import numbers

import numpy as np
import torch
from scipy.spatial import cKDTree

from match.map_extractor.partial_aware_soft_map_extractor import _to_numpy
from match.map_extractor.spectral_refined_partial_aware_map_extractor import (
    SpectrallyRefinedPartialAwareMapExtractor,
)


def _validate_ckdtree_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, numbers.Integral):
        raise ValueError("ckdtree_workers must be a positive integer or -1")
    workers = int(workers)
    if workers == 0 or workers < -1:
        raise ValueError("ckdtree_workers must be a positive integer or -1")
    return workers


def _validate_vertices(vertices, name):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(
            f"Dense fallback {name} must be a non-empty array shaped (V, 3)"
        )
    return vertices


class ParallelSpectrallyRefinedPartialAwareMapExtractor(
    SpectrallyRefinedPartialAwareMapExtractor
):
    """Partial-aware extractor with cached, parallel dense fallback queries."""

    def __init__(
        self,
        ckdtree_workers: int = 4,
        cache_device_shape_tensors: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ckdtree_workers = _validate_ckdtree_workers(ckdtree_workers)
        self.cache_device_shape_tensors = bool(cache_device_shape_tensors)
        self._source_to_proxy_cache = {}
        self._proxy_to_target_cache = {}
        self._device_tensor_cache = {}

    @staticmethod
    def _shape_key(shape):
        name = shape.get("name")
        if name is None:
            return None
        return str(name)

    def _cached_nearest(self, cache, shape, *, query_key, reference_key):
        query = _validate_vertices(_to_numpy(shape[query_key]), query_key)
        reference = _validate_vertices(
            _to_numpy(shape[reference_key]),
            reference_key,
        )
        shape_key = self._shape_key(shape)
        cache_key = None
        if shape_key is not None:
            cache_key = (
                shape_key,
                query_key,
                reference_key,
                len(query),
                len(reference),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        nearest = np.asarray(
            cKDTree(reference).query(
                query,
                workers=self.ckdtree_workers,
            )[1],
            dtype=np.int64,
        )
        if cache_key is not None:
            cache[cache_key] = nearest
        return nearest

    def _dense_fallback_map(self, source_shape, target_shape):
        source_proxy = _validate_vertices(
            _to_numpy(source_shape["articulated_verts"]),
            "source articulated_verts",
        )
        target_proxy = _validate_vertices(
            _to_numpy(target_shape["articulated_verts"]),
            "target articulated_verts",
        )
        if len(source_proxy) != len(target_proxy):
            raise ValueError(
                "Source and target proxy meshes must share topology for dense "
                f"fallback: {len(source_proxy)} vs {len(target_proxy)} vertices"
            )

        source_to_proxy = self._cached_nearest(
            self._source_to_proxy_cache,
            source_shape,
            query_key="verts",
            reference_key="articulated_verts",
        )
        proxy_to_target = self._cached_nearest(
            self._proxy_to_target_cache,
            target_shape,
            query_key="articulated_verts",
            reference_key="verts",
        )
        return np.asarray(proxy_to_target[source_to_proxy], dtype=np.int64)

    def _should_prepare_dense_fallback(self, missing):
        return bool(np.any(missing))

    def _tensor_on_device(self, shape, key, tensor, device, role):
        if not self.cache_device_shape_tensors:
            return tensor.to(device)
        shape_key = self._shape_key(shape)
        if shape_key is None:
            return tensor.to(device)
        cache_key = (
            shape_key,
            role,
            key,
            str(device),
            tuple(tensor.shape),
            tensor.dtype,
        )
        cached = self._device_tensor_cache.get(cache_key)
        if cached is None:
            cached = tensor.to(device)
            self._device_tensor_cache[cache_key] = cached
        return cached

    def _prepare_result_shape(self, shape, device, *, role):
        device_keys = {"corr"}
        if self.refinement_method != "none":
            device_keys.add("evecs")
            if role == "source":
                device_keys.add("evecs_trans")
                if self.refinement_method == "zoomout":
                    device_keys.add("mass")

        result = dict(shape)
        for key in device_keys:
            value = result.get(key)
            if torch.is_tensor(value):
                result[key] = self._tensor_on_device(
                    shape,
                    key,
                    value,
                    device,
                    role,
                )
        return result
