import numbers
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
import trimesh
from hydra.utils import to_absolute_path
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

SMOOTH_KERNEL_CACHE_VERSION = 1


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, numbers.Number):
        return np.asarray(value)
    raise TypeError(f"Unsupported array type: {type(value)}")


def _scipy_to_torch_sparse(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        device=device,
    ).coalesce()


def _diagnostics_to_torch(diagnostics, device):
    return {
        name: torch.from_numpy(np.asarray(value)).float().to(device)
        for name, value in diagnostics.items()
    }


def _closest_points(mesh: trimesh.Trimesh, points: np.ndarray, fallback_chunk_size=128):
    try:
        return trimesh.proximity.closest_point(mesh, points)
    except (ImportError, ModuleNotFoundError):
        chunks = [
            trimesh.proximity.closest_point_naive(
                mesh, points[start : start + fallback_chunk_size]
            )
            for start in range(0, len(points), fallback_chunk_size)
        ]
        return tuple(np.concatenate(values, axis=0) for values in zip(*chunks))


def point_to_triangle_assignment(points, proxy_vertices, proxy_faces):
    """Build a sparse raw-point to canonical-vertex barycentric assignment."""
    points = np.asarray(points, dtype=np.float64)
    proxy_vertices = np.asarray(proxy_vertices, dtype=np.float64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    proxy = trimesh.Trimesh(
        vertices=proxy_vertices,
        faces=proxy_faces,
        process=False,
    )
    closest, distances, triangle_ids = _closest_points(proxy, points)
    triangles = proxy_vertices[proxy_faces[triangle_ids]]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, closest)
    barycentric = np.clip(barycentric, 0.0, 1.0)
    barycentric /= np.maximum(barycentric.sum(axis=1, keepdims=True), 1e-12)

    rows = np.repeat(np.arange(len(points), dtype=np.int64), 3)
    columns = proxy_faces[triangle_ids].reshape(-1)
    values = barycentric.reshape(-1)
    assignment = scipy.sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(points), len(proxy_vertices)),
    )
    assignment.eliminate_zeros()
    return assignment, closest, distances, triangle_ids, barycentric


def mesh_vertex_area_weights(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangle_areas = trimesh.triangles.area(vertices[faces])
    weights = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(weights, faces[:, corner], triangle_areas / 3.0)
    return weights, float(triangle_areas.sum())


def _normalize_rows(vectors, eps=1e-12):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def canonicalization_confidence(
    *,
    raw_vertices,
    raw_faces,
    proxy_vertices,
    proxy_faces,
    assignment,
    distances,
    triangle_ids,
    barycentric,
    vertex_uncertainty,
    vertex_visibility_support,
    distance_scale,
    uncertainty_scale,
    use_visibility_term=True,
    eps=1e-12,
):
    raw_vertices = np.asarray(raw_vertices, dtype=np.float64)
    raw_faces = np.asarray(raw_faces, dtype=np.int64)
    proxy_vertices = np.asarray(proxy_vertices, dtype=np.float64)
    proxy_faces = np.asarray(proxy_faces, dtype=np.int64)
    vertex_uncertainty = np.asarray(vertex_uncertainty, dtype=np.float64)
    vertex_visibility_support = np.asarray(vertex_visibility_support, dtype=np.float64)

    body_diagonal = float(np.linalg.norm(np.ptp(proxy_vertices, axis=0)))
    s_distance = max(float(distance_scale) * body_diagonal, eps)
    s_uncertainty = max(float(uncertainty_scale) * body_diagonal, eps)
    distance_term = np.exp(-(np.asarray(distances) ** 2) / (2.0 * s_distance**2))

    interpolated_uncertainty = np.asarray(assignment @ vertex_uncertainty).reshape(-1)
    uncertainty_term = np.exp(-interpolated_uncertainty / (2.0 * s_uncertainty**2))

    raw_mesh = trimesh.Trimesh(
        vertices=raw_vertices,
        faces=raw_faces,
        process=False,
    )
    proxy_mesh = trimesh.Trimesh(
        vertices=proxy_vertices,
        faces=proxy_faces,
        process=False,
    )
    raw_normals = _normalize_rows(np.asarray(raw_mesh.vertex_normals))
    proxy_vertex_normals = np.asarray(proxy_mesh.vertex_normals)
    assigned_proxy_normals = (
        proxy_vertex_normals[proxy_faces[triangle_ids]] * barycentric[:, :, None]
    ).sum(axis=1)
    assigned_proxy_normals = _normalize_rows(assigned_proxy_normals)
    normal_term = np.clip(
        np.sum(raw_normals * assigned_proxy_normals, axis=1), 0.0, 1.0
    )

    interpolated_visibility = np.clip(
        np.asarray(assignment @ vertex_visibility_support).reshape(-1),
        0.0,
        1.0,
    )
    visibility_term = (
        interpolated_visibility
        if use_visibility_term
        else np.ones_like(interpolated_visibility)
    )
    confidence = distance_term * uncertainty_term * normal_term * visibility_term
    return {
        "confidence": np.clip(confidence, 0.0, 1.0),
        "distance": np.asarray(distances, dtype=np.float64),
        "uncertainty": interpolated_uncertainty,
        "distance_term": distance_term,
        "uncertainty_term": uncertainty_term,
        "normal_term": normal_term,
        "visibility_support": interpolated_visibility,
        "visibility_term": visibility_term,
    }


def reverse_canonical_map(
    assignment,
    area_weights,
    confidence,
    *,
    total_surface_area,
    kappa_factor=1.0,
    force_full_occupancy=False,
    eps=1e-8,
):
    """Construct Q with one final null column from target evidence."""
    assignment = assignment.tocsr()
    area_weights = np.asarray(area_weights, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    canonical_size = assignment.shape[1]
    if kappa_factor <= 0:
        raise ValueError(f"kappa_factor must be positive, got {kappa_factor}")

    weighted_evidence = area_weights * confidence
    evidence = np.asarray(assignment.T @ weighted_evidence).reshape(-1)
    kappa = max(
        float(kappa_factor) * float(total_surface_area) / canonical_size,
        float(eps),
    )
    occupancy = (
        np.ones(canonical_size, dtype=np.float64)
        if force_full_occupancy
        else 1.0 - np.exp(-evidence / kappa)
    )

    weighted_assignment = assignment.multiply(weighted_evidence[:, None])
    reverse_real = weighted_assignment.T.tocsr()
    reverse_real = scipy.sparse.diags(occupancy / (evidence + eps)) @ reverse_real
    reverse_null = scipy.sparse.csr_matrix((1.0 - occupancy)[:, None])
    reverse = scipy.sparse.hstack([reverse_real, reverse_null], format="csr")
    reverse.eliminate_zeros()
    return reverse, evidence, occupancy, kappa


def hard_map_from_soft(soft_map, target_vertex_count, *, allow_null_matches=True):
    soft_map = soft_map.tocsr()
    real = soft_map[:, :target_vertex_count].tocsr()
    null = np.asarray(soft_map[:, target_vertex_count].todense()).reshape(-1)
    point_map = np.full(soft_map.shape[0], -1, dtype=np.int64)
    best_real = np.zeros(soft_map.shape[0], dtype=np.float64)

    for row in range(real.shape[0]):
        start, end = real.indptr[row], real.indptr[row + 1]
        if start == end:
            continue
        values = real.data[start:end]
        best_offset = int(np.argmax(values))
        best_real[row] = values[best_offset]
        if not allow_null_matches or best_real[row] > null[row]:
            point_map[row] = real.indices[start + best_offset]
    return point_map, best_real, null


def dense_canonical_fallback_map(
    source_vertices,
    source_proxy_vertices,
    target_proxy_vertices,
    target_vertices,
    *,
    workers=1,
):
    """Compose dense nearest-neighbor maps through shared proxy topology."""
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_proxy_vertices = np.asarray(source_proxy_vertices, dtype=np.float64)
    target_proxy_vertices = np.asarray(target_proxy_vertices, dtype=np.float64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    if len(source_proxy_vertices) != len(target_proxy_vertices):
        raise ValueError(
            "Source and target proxy meshes must share topology for dense fallback: "
            f"{len(source_proxy_vertices)} vs {len(target_proxy_vertices)} vertices"
        )
    if not all(
        vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0
        for vertices in (
            source_vertices,
            source_proxy_vertices,
            target_proxy_vertices,
            target_vertices,
        )
    ):
        raise ValueError("Dense fallback expects non-empty vertex arrays shaped (V, 3)")

    source_to_proxy = cKDTree(source_proxy_vertices).query(
        source_vertices,
        workers=workers,
    )[1]
    proxy_to_target = cKDTree(target_vertices).query(
        target_proxy_vertices,
        workers=workers,
    )[1]
    return np.asarray(proxy_to_target[source_to_proxy], dtype=np.int64)


def template_edge_graph(vertices, faces):
    """Build the weighted mesh-edge graph used for template geodesics."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices with shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Expected triangular faces with shape (F, 3), got {faces.shape}"
        )
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Template vertices and faces must be non-empty")

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("Template contains zero-length or non-finite mesh edges")

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([lengths, lengths])
    graph = scipy.sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(vertices), len(vertices))
    )
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("Template mesh-edge graph is not connected")
    return graph


def build_template_smooth_kernel(
    vertices,
    faces,
    *,
    sigma_factor=0.02,
    min_weight=1e-4,
    batch_size=128,
):
    """Build a pruned, row-normalized Gaussian kernel from mesh geodesics."""
    sigma_factor = float(sigma_factor)
    min_weight = float(min_weight)
    batch_size = int(batch_size)
    if sigma_factor <= 0:
        raise ValueError(f"sigma_factor must be positive, got {sigma_factor}")
    if not 0 < min_weight < 1:
        raise ValueError(f"min_weight must be in (0, 1), got {min_weight}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    vertices = np.asarray(vertices, dtype=np.float64)
    body_diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if not np.isfinite(body_diagonal) or body_diagonal <= 0:
        raise ValueError(f"Invalid template bounding-box diagonal: {body_diagonal}")
    sigma = sigma_factor * body_diagonal
    distance_limit = sigma * np.sqrt(-2.0 * np.log(min_weight))
    graph = template_edge_graph(vertices, faces)

    row_parts = []
    column_parts = []
    value_parts = []
    for start in range(0, len(vertices), batch_size):
        stop = min(start + batch_size, len(vertices))
        distances = dijkstra(
            graph,
            directed=False,
            indices=np.arange(start, stop),
            limit=distance_limit,
        )
        local_rows, columns = np.nonzero(np.isfinite(distances))
        values = np.exp(-(distances[local_rows, columns] ** 2) / (2.0 * sigma**2))
        retained = values >= min_weight
        row_parts.append(local_rows[retained] + start)
        column_parts.append(columns[retained])
        value_parts.append(values[retained])

    kernel = scipy.sparse.csr_matrix(
        (
            np.concatenate(value_parts).astype(np.float32),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(len(vertices), len(vertices)),
        dtype=np.float32,
    )
    kernel.eliminate_zeros()
    row_sums = np.asarray(kernel.sum(axis=1)).reshape(-1)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Smooth kernel contains an empty or non-finite row")
    kernel = (scipy.sparse.diags(1.0 / row_sums) @ kernel).tocsr()
    kernel.sort_indices()
    return kernel, {
        "template_diagonal": body_diagonal,
        "sigma": sigma,
        "distance_limit": distance_limit,
    }


def save_smooth_kernel_cache(
    path,
    kernel,
    *,
    sigma_factor,
    min_weight,
    template_diagonal,
    mhr_model_path,
    template_type=None,
    template_path=None,
):
    """Store CSR arrays in compressed NPZ content at the configured path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kernel = kernel.tocsr()
    temporary_path = path.with_name(f"{path.name}.tmp")
    with temporary_path.open("wb") as cache_file:
        template_path = mhr_model_path if template_path is None else template_path
        template_type = "mhr" if template_type is None else template_type
        np.savez_compressed(
            cache_file,
            cache_version=np.asarray(SMOOTH_KERNEL_CACHE_VERSION, dtype=np.int64),
            data=kernel.data.astype(np.float32, copy=False),
            indices=kernel.indices.astype(np.int32, copy=False),
            indptr=kernel.indptr.astype(np.int32, copy=False),
            shape=np.asarray(kernel.shape, dtype=np.int64),
            sigma_factor=np.asarray(sigma_factor, dtype=np.float64),
            min_weight=np.asarray(min_weight, dtype=np.float64),
            template_diagonal=np.asarray(template_diagonal, dtype=np.float64),
            mhr_model_path=np.asarray(str(mhr_model_path)),
            template_type=np.asarray(str(template_type)),
            template_path=np.asarray(str(template_path)),
        )
    temporary_path.replace(path)


def parameterized_smooth_kernel_cache_path(path, *, sigma_factor, min_weight):
    """Return a cache path that uniquely identifies smoothing parameters."""
    path = Path(path)

    def parameter_token(value):
        return format(float(value), ".12g").replace("-", "m").replace(".", "p")

    return path.with_name(
        f"{path.stem}"
        f"-sigma{parameter_token(sigma_factor)}"
        f"-min{parameter_token(min_weight)}"
        f"{path.suffix}"
    )


def load_smooth_kernel_cache(path):
    """Load the sparse kernel and metadata without enabling pickle."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "cache_version",
                "data",
                "indices",
                "indptr",
                "shape",
                "sigma_factor",
                "min_weight",
                "template_diagonal",
                "mhr_model_path",
            }
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(f"missing fields {sorted(missing)}")
            version = int(cache["cache_version"])
            if version != SMOOTH_KERNEL_CACHE_VERSION:
                raise ValueError(
                    f"unsupported cache version {version}; "
                    f"expected {SMOOTH_KERNEL_CACHE_VERSION}"
                )
            shape = tuple(int(value) for value in cache["shape"])
            kernel = scipy.sparse.csr_matrix(
                (cache["data"], cache["indices"], cache["indptr"]), shape=shape
            )
            metadata = {
                "sigma_factor": float(cache["sigma_factor"]),
                "min_weight": float(cache["min_weight"]),
                "template_diagonal": float(cache["template_diagonal"]),
                "mhr_model_path": str(cache["mhr_model_path"]),
                "template_type": (
                    str(cache["template_type"])
                    if "template_type" in cache.files
                    else "mhr"
                ),
                "template_path": (
                    str(cache["template_path"])
                    if "template_path" in cache.files
                    else str(cache["mhr_model_path"])
                ),
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Invalid smooth-kernel cache '{path}': {error}") from error

    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Smooth-kernel cache must be square, got {kernel.shape}")
    if not np.all(np.isfinite(kernel.data)) or np.any(kernel.data < 0):
        raise ValueError("Smooth-kernel cache contains invalid weights")
    row_sums = np.asarray(kernel.sum(axis=1)).reshape(-1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Smooth-kernel cache rows are not normalized")
    return kernel, metadata


def neutral_mhr_template(mhr_model_path):
    """Generate the zero-identity, zero-expression, neutral-pose MHR template."""
    model = torch.jit.load(str(mhr_model_path), map_location="cpu").eval()
    identity_count = int(model.character_torch.blend_shape.shape_vectors.shape[0])
    transformed_parameter_count = int(
        model.character_torch.parameter_transform.parameter_transform.shape[1]
    )
    parameter_count = transformed_parameter_count - identity_count
    expression_count = int(model.face_expressions_model.shape_vectors.shape[0])
    with torch.no_grad():
        vertices, _ = model(
            torch.zeros((1, identity_count), dtype=torch.float32),
            torch.zeros((1, parameter_count), dtype=torch.float32),
            torch.zeros((1, expression_count), dtype=torch.float32),
        )
    faces = model.character_torch.mesh.faces
    return vertices[0].cpu().numpy(), faces.cpu().numpy()


def neutral_smal_template(smal_model_path):
    """Load the neutral SMAL vertices and faces from the trusted model asset."""
    import inspect

    compatibility = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    added_numpy_names = []
    added_getargspec = False
    try:
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
            added_getargspec = True
        for name, value in compatibility.items():
            if name not in np.__dict__:
                setattr(np, name, value)
                added_numpy_names.append(name)
        with Path(smal_model_path).open("rb") as model_file:
            model = pickle.load(model_file, encoding="latin1")
        vertices = np.asarray(model["v_template"], dtype=np.float64)
        faces = np.asarray(model["f"], dtype=np.int64)
    finally:
        for name in added_numpy_names:
            delattr(np, name)
        if added_getargspec:
            delattr(inspect, "getargspec")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"Invalid neutral SMAL vertices in {smal_model_path}: {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Invalid neutral SMAL faces in {smal_model_path}: {faces.shape}"
        )
    return vertices, faces


def compose_soft_map(source_assignment, reverse_target, smooth_kernel=None):
    """Compose source-to-canonical and canonical-to-target sparse maps."""
    source_assignment = source_assignment.tocsr()
    if smooth_kernel is not None:
        source_assignment = (source_assignment @ smooth_kernel).tocsr()
    soft_map = (source_assignment @ reverse_target).tocsr()
    soft_map.eliminate_zeros()
    return soft_map


def reciprocal_consistency(
    forward_soft,
    backward_soft,
    forward_point_map,
    backward_point_map,
    *,
    forward_target_vertex_count,
    backward_target_vertex_count,
):
    """Filter hard matches by mutual selection and compute symmetric scores."""
    forward_real = forward_soft[:, :forward_target_vertex_count].tocsr()
    backward_real = backward_soft[:, :backward_target_vertex_count].tocsr()
    if forward_real.shape != backward_real.T.shape:
        raise ValueError(
            "Forward/backward real soft maps have incompatible shapes: "
            f"{forward_real.shape} vs {backward_real.shape}"
        )

    score_map = forward_real.multiply(backward_real.T).tocsr()
    score_map.data = np.sqrt(score_map.data)
    score_map.eliminate_zeros()

    forward_point_map = np.asarray(forward_point_map, dtype=np.int64)
    backward_point_map = np.asarray(backward_point_map, dtype=np.int64)
    source_indices = np.arange(len(forward_point_map), dtype=np.int64)
    valid = forward_point_map >= 0
    consistent = np.zeros(len(forward_point_map), dtype=bool)
    consistent[valid] = (
        backward_point_map[forward_point_map[valid]] == source_indices[valid]
    )
    filtered_point_map = forward_point_map.copy()
    filtered_point_map[~consistent] = -1

    selected_scores = np.zeros(len(forward_point_map), dtype=np.float64)
    selected_scores[valid] = np.asarray(
        score_map[source_indices[valid], forward_point_map[valid]]
    ).reshape(-1)
    return filtered_point_map, consistent, score_map, selected_scores


class PartialAwareSoftCanonizationMapExtractor:
    """LBO-free sparse canonical composition with an explicit null target."""

    def __init__(
        self,
        device: str = "cuda",
        distance_scale: float = 0.02,
        uncertainty_scale: float = 0.02,
        kappa_factor: float = 1.0,
        eps: float = 1e-8,
        cache_canonicalization: bool = True,
        matching_mode: str = "partial",
        use_visibility_term: bool = True,
        use_smooth_kernel: bool = False,
        use_reciprocal_consistency: bool = False,
        return_soft_maps: bool = False,
        smooth_kernel_cache_path: str = "output/cache/MHR_smooth_kernel.npy",
        smooth_kernel_mhr_path: str = (
            "pkg/sam_3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
        ),
        smooth_kernel_template_type: str = "mhr",
        smooth_kernel_template_path: str = None,
        smooth_kernel_sigma_factor: float = 0.01,
        smooth_kernel_min_weight: float = 1e-3,
        smooth_kernel_batch_size: int = 128,
    ):
        self.device = device
        self.distance_scale = float(distance_scale)
        self.uncertainty_scale = float(uncertainty_scale)
        self.kappa_factor = float(kappa_factor)
        self.eps = float(eps)
        self.cache_canonicalization = bool(cache_canonicalization)
        self.matching_mode = str(matching_mode)
        self.use_visibility_term = bool(use_visibility_term)
        self.use_smooth_kernel = bool(use_smooth_kernel)
        self.use_reciprocal_consistency = bool(use_reciprocal_consistency)
        self.return_soft_maps = bool(return_soft_maps)
        smooth_kernel_cache_base_path = Path(
            to_absolute_path(str(smooth_kernel_cache_path))
        )
        self.smooth_kernel_mhr_path = Path(
            to_absolute_path(str(smooth_kernel_mhr_path))
        )
        self.smooth_kernel_template_type = str(smooth_kernel_template_type).lower()
        configured_template_path = (
            smooth_kernel_mhr_path
            if smooth_kernel_template_path is None
            else smooth_kernel_template_path
        )
        self.smooth_kernel_template_path = Path(
            to_absolute_path(str(configured_template_path))
        )
        self.smooth_kernel_sigma_factor = float(smooth_kernel_sigma_factor)
        self.smooth_kernel_min_weight = float(smooth_kernel_min_weight)
        self.smooth_kernel_batch_size = int(smooth_kernel_batch_size)
        self.smooth_kernel_cache_path = parameterized_smooth_kernel_cache_path(
            smooth_kernel_cache_base_path,
            sigma_factor=self.smooth_kernel_sigma_factor,
            min_weight=self.smooth_kernel_min_weight,
        )
        if self.distance_scale <= 0 or self.uncertainty_scale <= 0:
            raise ValueError("distance_scale and uncertainty_scale must be positive")
        if self.kappa_factor <= 0:
            raise ValueError("kappa_factor must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.matching_mode not in {"partial", "full"}:
            raise ValueError(
                "matching_mode must be 'partial' or 'full', got "
                f"{self.matching_mode!r}"
            )
        if self.smooth_kernel_sigma_factor <= 0:
            raise ValueError("smooth_kernel_sigma_factor must be positive")
        if not 0 < self.smooth_kernel_min_weight < 1:
            raise ValueError("smooth_kernel_min_weight must be in (0, 1)")
        if self.smooth_kernel_batch_size <= 0:
            raise ValueError("smooth_kernel_batch_size must be positive")
        if self.smooth_kernel_template_type not in {"mhr", "smal"}:
            raise ValueError(
                "smooth_kernel_template_type must be 'mhr' or 'smal', got "
                f"{self.smooth_kernel_template_type!r}"
            )
        self.method = (
            "partial-aware-soft-canonization"
            if self.matching_mode == "partial"
            else "full-coverage-soft-canonization"
        )
        self._canonical_cache = {}
        self._smooth_kernel = None

    def _dense_fallback_map(self, source_shape, target_shape):
        return dense_canonical_fallback_map(
            _to_numpy(source_shape["verts"]),
            _to_numpy(source_shape["articulated_verts"]),
            _to_numpy(target_shape["articulated_verts"]),
            _to_numpy(target_shape["verts"]),
        )

    def _should_prepare_dense_fallback(self, missing):
        del missing
        return True

    def _prepare_result_shape(self, shape, device, *, role):
        del role
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in shape.items()
        }

    def _canonicalize(self, shape):
        cache_key = shape.get("name")
        if (
            self.cache_canonicalization
            and cache_key is not None
            and cache_key in self._canonical_cache
        ):
            return self._canonical_cache[cache_key]

        required = {
            "verts",
            "faces",
            "articulated_verts",
            "articulated_faces",
            "articulation_confidence",
        }
        missing = required.difference(shape)
        if missing:
            raise KeyError(f"Shape {shape.get('name')} is missing {sorted(missing)}")

        raw_vertices = _to_numpy(shape["verts"])
        raw_faces = _to_numpy(shape["faces"])
        proxy_vertices = _to_numpy(shape["articulated_verts"])
        proxy_faces = _to_numpy(shape["articulated_faces"])
        confidence_artifact = shape["articulation_confidence"]
        vertex_uncertainty = _to_numpy(confidence_artifact["vertex_uncertainty"])
        visibility_support = _to_numpy(confidence_artifact["vertex_visibility_support"])
        if len(vertex_uncertainty) != len(proxy_vertices):
            raise ValueError(
                f"Uncertainty topology for {shape.get('name')} has "
                f"{len(vertex_uncertainty)} values for {len(proxy_vertices)} vertices"
            )

        assignment = point_to_triangle_assignment(
            raw_vertices, proxy_vertices, proxy_faces
        )
        matrix, _, distances, triangle_ids, barycentric = assignment
        diagnostics = canonicalization_confidence(
            raw_vertices=raw_vertices,
            raw_faces=raw_faces,
            proxy_vertices=proxy_vertices,
            proxy_faces=proxy_faces,
            assignment=matrix,
            distances=distances,
            triangle_ids=triangle_ids,
            barycentric=barycentric,
            vertex_uncertainty=vertex_uncertainty,
            vertex_visibility_support=visibility_support,
            distance_scale=self.distance_scale,
            uncertainty_scale=self.uncertainty_scale,
            use_visibility_term=self.use_visibility_term,
            eps=self.eps,
        )
        area_weights, total_area = mesh_vertex_area_weights(raw_vertices, raw_faces)
        canonicalization = {
            "assignment": matrix,
            "diagnostics": diagnostics,
            "area_weights": area_weights,
            "total_area": total_area,
        }
        if self.cache_canonicalization and cache_key is not None:
            self._canonical_cache[cache_key] = canonicalization
        return canonicalization

    def _reverse_target(self, canonicalization):
        if "reverse" not in canonicalization:
            reverse, evidence, occupancy, kappa = reverse_canonical_map(
                canonicalization["assignment"],
                canonicalization["area_weights"],
                canonicalization["diagnostics"]["confidence"],
                total_surface_area=canonicalization["total_area"],
                kappa_factor=self.kappa_factor,
                force_full_occupancy=self.matching_mode == "full",
                eps=self.eps,
            )
            canonicalization["reverse"] = reverse
            canonicalization["evidence"] = evidence
            canonicalization["occupancy"] = occupancy
            canonicalization["kappa"] = kappa
        return (
            canonicalization["reverse"],
            canonicalization["evidence"],
            canonicalization["occupancy"],
            canonicalization["kappa"],
        )

    def _get_smooth_kernel(self, canonical_size):
        if not self.use_smooth_kernel:
            return None
        if self._smooth_kernel is not None:
            if self._smooth_kernel.shape != (canonical_size, canonical_size):
                raise ValueError(
                    "Cached in-memory smooth kernel has shape "
                    f"{self._smooth_kernel.shape}, expected "
                    f"{(canonical_size, canonical_size)}"
                )
            return self._smooth_kernel

        rebuild = True
        if self.smooth_kernel_cache_path.is_file():
            cached, metadata = load_smooth_kernel_cache(self.smooth_kernel_cache_path)
            if cached.shape != (canonical_size, canonical_size):
                raise ValueError(
                    "Smooth-kernel cache topology does not match articulated meshes: "
                    f"{cached.shape} vs {(canonical_size, canonical_size)}"
                )
            rebuild = not (
                np.isclose(
                    metadata["sigma_factor"],
                    self.smooth_kernel_sigma_factor,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.isclose(
                    metadata["min_weight"],
                    self.smooth_kernel_min_weight,
                    rtol=0.0,
                    atol=1e-12,
                )
                and metadata["template_type"] == self.smooth_kernel_template_type
                and metadata["template_path"] == str(self.smooth_kernel_template_path)
            )
            if not rebuild:
                self._smooth_kernel = cached

        if rebuild:
            if not self.smooth_kernel_template_path.is_file():
                raise FileNotFoundError(
                    "Cannot build smooth kernel because the neutral template "
                    f"asset is missing: {self.smooth_kernel_template_path}"
                )
            if self.smooth_kernel_template_type == "mhr":
                vertices, faces = neutral_mhr_template(self.smooth_kernel_template_path)
            else:
                vertices, faces = neutral_smal_template(
                    self.smooth_kernel_template_path
                )
            if len(vertices) != canonical_size:
                raise ValueError(
                    "Neutral template topology does not match articulated meshes: "
                    f"{len(vertices)} vs {canonical_size} vertices"
                )
            kernel, diagnostics = build_template_smooth_kernel(
                vertices,
                faces,
                sigma_factor=self.smooth_kernel_sigma_factor,
                min_weight=self.smooth_kernel_min_weight,
                batch_size=self.smooth_kernel_batch_size,
            )
            save_smooth_kernel_cache(
                self.smooth_kernel_cache_path,
                kernel,
                sigma_factor=self.smooth_kernel_sigma_factor,
                min_weight=self.smooth_kernel_min_weight,
                template_diagonal=diagnostics["template_diagonal"],
                mhr_model_path=self.smooth_kernel_template_path,
                template_type=self.smooth_kernel_template_type,
                template_path=self.smooth_kernel_template_path,
            )
            self._smooth_kernel = kernel
        return self._smooth_kernel

    def pipeline(self, shape_m, shape_n):
        canonical_m = self._canonicalize(shape_m)
        canonical_n = self._canonicalize(shape_n)
        if canonical_m["assignment"].shape[1] != canonical_n["assignment"].shape[1]:
            raise ValueError(
                "Articulated meshes do not share canonical topology: "
                f"{canonical_m['assignment'].shape[1]} vs "
                f"{canonical_n['assignment'].shape[1]} vertices"
            )

        canonical_size = canonical_m["assignment"].shape[1]
        smooth_kernel = self._get_smooth_kernel(canonical_size)
        target_vertex_count = len(_to_numpy(shape_m["verts"]))
        source_vertex_count = len(_to_numpy(shape_n["verts"]))
        reverse_m, evidence_m, occupancy_m, kappa_m = self._reverse_target(canonical_m)
        soft_n2m = compose_soft_map(canonical_n["assignment"], reverse_m, smooth_kernel)
        point_map, best_real, null = hard_map_from_soft(
            soft_n2m,
            target_vertex_count,
            allow_null_matches=self.matching_mode == "partial",
        )

        fallback_map = None
        fallback_used = np.zeros(source_vertex_count, dtype=bool)
        if self.matching_mode == "full":
            fallback_used = point_map < 0
            if self._should_prepare_dense_fallback(fallback_used):
                fallback_map = self._dense_fallback_map(shape_n, shape_m)
            if np.any(fallback_used):
                point_map[fallback_used] = fallback_map[fallback_used]
        point_map_before_reciprocal = point_map.copy()

        reciprocal_output = None
        if self.use_reciprocal_consistency:
            reverse_n, _, _, _ = self._reverse_target(canonical_n)
            soft_m2n = compose_soft_map(
                canonical_m["assignment"], reverse_n, smooth_kernel
            )
            backward_point_map, _, _ = hard_map_from_soft(
                soft_m2n,
                source_vertex_count,
                allow_null_matches=self.matching_mode == "partial",
            )
            if self.matching_mode == "full":
                backward_missing = backward_point_map < 0
                backward_fallback = None
                if self._should_prepare_dense_fallback(backward_missing):
                    backward_fallback = self._dense_fallback_map(shape_m, shape_n)
                if np.any(backward_missing):
                    backward_point_map[backward_missing] = backward_fallback[
                        backward_missing
                    ]
            point_map, consistent, score_map, selected_scores = reciprocal_consistency(
                soft_n2m,
                soft_m2n,
                point_map,
                backward_point_map,
                forward_target_vertex_count=target_vertex_count,
                backward_target_vertex_count=source_vertex_count,
            )
            reciprocal_output = {
                "p2p_pre_reciprocal": torch.from_numpy(
                    point_map_before_reciprocal
                ).long(),
                "backward_p2p": torch.from_numpy(backward_point_map).long(),
                "reciprocal_consistent": torch.from_numpy(consistent),
                "reciprocal_score": torch.from_numpy(selected_scores).float(),
            }
            if self.return_soft_maps:
                reciprocal_output["reciprocal_score_map"] = score_map

        if self.matching_mode == "full":
            fallback_after_reciprocal = point_map < 0
            fallback_used |= fallback_after_reciprocal
            if np.any(fallback_after_reciprocal):
                if fallback_map is None:
                    fallback_map = self._dense_fallback_map(shape_n, shape_m)
                point_map[fallback_after_reciprocal] = fallback_map[
                    fallback_after_reciprocal
                ]
            if np.any(point_map < 0):
                raise RuntimeError(
                    "Full matching mode produced an incomplete point map"
                )

        device = torch.device(self.device)
        result = {
            "shape_m": self._prepare_result_shape(
                shape_m,
                device,
                role="target",
            ),
            "shape_n": self._prepare_result_shape(
                shape_n,
                device,
                role="source",
            ),
            "p2p": torch.from_numpy(point_map).long().to(device),
            "source_confidence": torch.from_numpy(
                canonical_n["diagnostics"]["confidence"]
            )
            .float()
            .to(device),
            "target_confidence": torch.from_numpy(
                canonical_m["diagnostics"]["confidence"]
            )
            .float()
            .to(device),
            "source_confidence_terms": _diagnostics_to_torch(
                canonical_n["diagnostics"], device
            ),
            "target_confidence_terms": _diagnostics_to_torch(
                canonical_m["diagnostics"], device
            ),
            "target_evidence": torch.from_numpy(evidence_m).float().to(device),
            "target_occupancy": torch.from_numpy(occupancy_m).float().to(device),
            "best_real_probability": torch.from_numpy(best_real).float().to(device),
            "null_probability": torch.from_numpy(null).float().to(device),
            "kappa": kappa_m,
            "matching_mode": self.matching_mode,
            "dense_fallback_used": torch.from_numpy(fallback_used).to(device),
            "visibility_term_enabled": self.use_visibility_term,
            "smooth_kernel_enabled": self.use_smooth_kernel,
            "reciprocal_consistency_enabled": self.use_reciprocal_consistency,
            "method": self.method,
        }
        if self.return_soft_maps:
            result["soft_map"] = _scipy_to_torch_sparse(soft_n2m, device)
            result["soft_map_real"] = _scipy_to_torch_sparse(
                soft_n2m[:, :target_vertex_count],
                device,
            )
        if smooth_kernel is not None:
            result["smooth_kernel_nnz"] = smooth_kernel.nnz
            result["smooth_kernel_cache_path"] = str(self.smooth_kernel_cache_path)
        if reciprocal_output is not None:
            result.update(
                {
                    key: (
                        _scipy_to_torch_sparse(value, device)
                        if scipy.sparse.issparse(value)
                        else value.to(device)
                    )
                    for key, value in reciprocal_output.items()
                }
            )
        return result
