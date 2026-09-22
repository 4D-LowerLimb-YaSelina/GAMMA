import fcntl
import hashlib
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import potpourri3d as pp3d
import scipy
import scipy.sparse.linalg as sla
import torch
import trimesh
from hydra.utils import to_absolute_path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
from torch.utils.data import Dataset


def _absolute_path(path):
    return Path(to_absolute_path(str(path)))


def _torch_to_numpy(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().cpu().numpy()


def _sparse_numpy_to_torch(matrix):
    matrix = matrix.tocoo()
    values = torch.from_numpy(matrix.data)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size(matrix.shape),
    ).coalesce()


def _sparse_torch_to_numpy(matrix):
    if len(matrix.shape) != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(matrix.shape)}")
    matrix = matrix.coalesce()
    indices = _torch_to_numpy(matrix.indices())
    values = _torch_to_numpy(matrix.values())
    return scipy.sparse.coo_matrix(
        (values, indices),
        shape=tuple(matrix.shape),
    ).tocsc()


def _read_sparse_matrix(archive, prefix):
    return scipy.sparse.csc_matrix(
        (
            archive[f"{prefix}_data"],
            archive[f"{prefix}_indices"],
            archive[f"{prefix}_indptr"],
        ),
        shape=tuple(int(value) for value in archive[f"{prefix}_shape"]),
    )


def _dot(first, second, keepdim=False):
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} != {second.shape}")
    return torch.sum(first * second, dim=-1, keepdim=keepdim)


def _normalize(vectors, eps=1e-12):
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors / (norm + eps)


def _face_normals(vertices, faces):
    coordinates = vertices[faces]
    edge_a = coordinates[:, 1] - coordinates[:, 0]
    edge_b = coordinates[:, 2] - coordinates[:, 0]
    return _normalize(torch.cross(edge_a, edge_b, dim=-1))


def _mesh_vertex_normals(vertices, faces):
    face_normals = _torch_to_numpy(
        _face_normals(torch.from_numpy(vertices), torch.from_numpy(faces))
    )
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    normal_lengths = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    return vertex_normals / (normal_lengths + 1e-12)


def _vertex_normals(vertices, faces):
    vertices_np = _torch_to_numpy(vertices)
    faces_np = _torch_to_numpy(faces)
    normals = _mesh_vertex_normals(vertices_np, faces_np)

    bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_mask.any():
        extent = np.amax(vertices_np, axis=0) - np.amin(vertices_np, axis=0)
        scale = np.linalg.norm(extent) * 1e-4
        random_state = np.random.RandomState(seed=777)
        wiggle = (random_state.rand(*vertices.shape) - 0.5) * scale
        normals = _mesh_vertex_normals(
            vertices_np + bad_mask * wiggle,
            faces_np,
        )

    bad_mask = np.isnan(normals).any(axis=1)
    if bad_mask.any():
        random_state = np.random.RandomState(seed=777)
        normals[bad_mask] = (random_state.rand(*vertices.shape) - 0.5)[bad_mask]
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12

    normals = torch.from_numpy(normals).to(
        device=vertices.device,
        dtype=vertices.dtype,
    )
    if torch.isnan(normals).any():
        raise ValueError("NaN vertex normals")
    return normals


def _build_tangent_frames(vertices, faces):
    vertex_normals = _vertex_normals(vertices, faces)
    vertex_count = len(vertices)
    candidate_x = torch.tensor(
        [1, 0, 0],
        device=vertices.device,
        dtype=vertices.dtype,
    ).expand(vertex_count, -1)
    candidate_y = torch.tensor(
        [0, 1, 0],
        device=vertices.device,
        dtype=vertices.dtype,
    ).expand(vertex_count, -1)
    basis_x = torch.where(
        torch.abs(_dot(vertex_normals, candidate_x, keepdim=True)) < 0.9,
        candidate_x,
        candidate_y,
    )
    basis_x -= _dot(basis_x, vertex_normals, keepdim=True) * vertex_normals
    basis_x = _normalize(basis_x)
    basis_y = torch.cross(vertex_normals, basis_x, dim=-1)
    frames = torch.stack((basis_x, basis_y, vertex_normals), dim=-2)
    if torch.isnan(frames).any():
        raise ValueError("NaN coordinate frame; mesh may be degenerate")
    return frames


def _edge_tangent_vectors(vertices, frames, edges):
    edge_vectors = vertices[edges[1]] - vertices[edges[0]]
    basis_x = frames[edges[0], 0]
    basis_y = frames[edges[0], 1]
    return torch.stack(
        (
            _dot(edge_vectors, basis_x),
            _dot(edge_vectors, basis_y),
        ),
        dim=-1,
    )


def _build_gradient(vertices, edges, edge_tangent_vectors):
    vertex_count = len(vertices)
    outgoing_edges = [[] for _ in range(vertex_count)]
    for edge_index in range(edges.shape[1]):
        tail = edges[0, edge_index]
        tip = edges[1, edge_index]
        if tip != tail:
            outgoing_edges[tail].append(edge_index)

    row_indices = []
    column_indices = []
    values = []
    regularization = 1e-5

    for vertex_index in range(vertex_count):
        edge_indices = outgoing_edges[vertex_index]
        neighbor_count = len(edge_indices)
        lhs = np.zeros((neighbor_count, 2))
        rhs = np.zeros((neighbor_count, neighbor_count + 1))
        vertex_lookup = [vertex_index]

        for neighbor_index, edge_index in enumerate(edge_indices):
            neighbor_vertex = edges[1, edge_index]
            vertex_lookup.append(neighbor_vertex)
            lhs[neighbor_index] = edge_tangent_vectors[edge_index]
            rhs[neighbor_index, 0] = -1
            rhs[neighbor_index, neighbor_index + 1] = 1

        lhs_transposed = lhs.T
        lhs_inverse = (
            np.linalg.inv(lhs_transposed @ lhs + regularization * np.identity(2))
            @ lhs_transposed
        )
        solution = lhs_inverse @ rhs
        coefficients = (solution[0] + 1j * solution[1]).T

        for local_index, global_index in enumerate(vertex_lookup):
            row_indices.append(vertex_index)
            column_indices.append(global_index)
            values.append(coefficients[local_index])

    return scipy.sparse.coo_matrix(
        (np.asarray(values), (row_indices, column_indices)),
        shape=(vertex_count, vertex_count),
    ).tocsc()


def _compute_operators(vertices, faces, k):
    """Compute the BeCoS spectral-cache payload without utility-project imports."""
    if k <= 0:
        raise ValueError(f"num_evecs must be positive, got {k}")
    if k >= len(vertices):
        raise ValueError(
            f"num_evecs must be smaller than the vertex count, got {k} for "
            f"{len(vertices)} vertices"
        )

    device = vertices.device
    dtype = vertices.dtype
    epsilon = 1e-8
    vertices_np = _torch_to_numpy(vertices).astype(np.float64)
    faces_np = _torch_to_numpy(faces).astype(np.int64)
    frames = _build_tangent_frames(vertices, faces)

    laplacian = pp3d.cotan_laplacian(
        vertices_np,
        faces_np,
        denom_eps=1e-10,
    )
    mass_np = pp3d.vertex_areas(vertices_np, faces_np)
    mass_np += epsilon * np.mean(mass_np)
    if np.isnan(laplacian.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(mass_np).any():
        raise RuntimeError("NaN mass matrix")

    laplacian_eigsh = (
        laplacian + epsilon * scipy.sparse.identity(laplacian.shape[0])
    ).tocsc()
    mass_matrix = scipy.sparse.diags(mass_np)
    failure_count = 0
    while True:
        try:
            eigenvalues_np, eigenvectors_np = sla.eigsh(
                laplacian_eigsh,
                k=k,
                M=mass_matrix,
                sigma=epsilon,
            )
            eigenvalues_np = np.clip(
                eigenvalues_np,
                a_min=0,
                a_max=float("inf"),
            )
            break
        except Exception as error:
            if failure_count > 3:
                raise ValueError("Failed to compute eigendecomposition") from error
            failure_count += 1
            laplacian_eigsh = laplacian_eigsh + (
                epsilon * 10**failure_count * scipy.sparse.identity(laplacian.shape[0])
            )

    laplacian_coo = laplacian.tocoo()
    edges = torch.tensor(
        np.stack((laplacian_coo.row, laplacian_coo.col)),
        device=device,
        dtype=faces.dtype,
    )
    tangent_vectors = _edge_tangent_vectors(vertices, frames, edges)
    gradient_np = _build_gradient(
        vertices_np,
        _torch_to_numpy(edges),
        _torch_to_numpy(tangent_vectors),
    )

    return (
        torch.from_numpy(mass_np).to(device=device, dtype=dtype),
        _sparse_numpy_to_torch(laplacian).to(device=device, dtype=dtype),
        torch.from_numpy(eigenvalues_np).to(device=device, dtype=dtype),
        torch.from_numpy(eigenvectors_np).to(device=device, dtype=dtype),
        _sparse_numpy_to_torch(np.real(gradient_np)).to(
            device=device,
            dtype=dtype,
        ),
        _sparse_numpy_to_torch(np.imag(gradient_np)).to(
            device=device,
            dtype=dtype,
        ),
    )


def _compute_geodesic_distances(vertices, faces):
    """Match the edge-graph geodesic benchmark used by the BeCoS baseline."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    adjacency = mesh.vertex_adjacency_graph
    if not nx.is_connected(adjacency):
        raise ValueError("BeCoS mesh graph is not connected")

    vertex_count = len(vertices)
    adjacency_matrix = nx.adjacency_matrix(adjacency, range(vertex_count))
    knn_graph = neighbors.kneighbors_graph(
        vertices,
        n_neighbors=min(500, vertex_count - 1),
        mode="distance",
        include_self=False,
    )
    distance_adjacency = csr_matrix((vertex_count, vertex_count)).tolil()
    distance_adjacency[adjacency_matrix != 0] = knn_graph[adjacency_matrix != 0]
    distances = shortest_path(distance_adjacency, directed=False)
    if np.isinf(distances).any():
        raise ValueError("Infinite values in BeCoS geodesic distance matrix")
    return distances


class PairBeCoSArticulatedDataset(Dataset):
    """BeCoS pairs augmented with uncertainty-aware articulated proxies."""

    MODES = {"partial_full", "partial_partial"}
    PHASES = {"train", "val", "test"}
    SPECTRAL_FIELDS = {
        "k_eig",
        "mass",
        "evals",
        "evecs",
        "L_data",
        "L_indices",
        "L_indptr",
        "L_shape",
        "gradX_data",
        "gradX_indices",
        "gradX_indptr",
        "gradX_shape",
        "gradY_data",
        "gradY_indices",
        "gradY_indptr",
        "gradY_shape",
    }

    def __init__(
        self,
        data_root,
        articulated_shape_dir,
        phase="test",
        mode="partial_full",
        articulated_subdir="mhr",
        articulated_extension=".obj",
        return_faces=True,
        return_L=False,
        return_mass=True,
        num_evecs=200,
        return_evecs=True,
        return_grad=False,
        return_corr=True,
        return_dist=True,
        return_articulation_confidence=True,
        articulation_confidence_suffix=".confidence.npz",
        compute_cache=True,
        normalize_dist_by_area=True,
    ):
        self.data_root = _absolute_path(data_root)
        self.articulated_shape_dir = _absolute_path(articulated_shape_dir)
        self.phase = str(phase)
        self.mode = str(mode)
        self.articulated_subdir = articulated_subdir
        self.articulated_extension = str(articulated_extension)
        self.return_faces = bool(return_faces)
        self.return_L = bool(return_L)
        self.return_mass = bool(return_mass)
        self.num_evecs = int(num_evecs)
        self.return_evecs = bool(return_evecs)
        self.return_grad = bool(return_grad)
        self.return_corr = bool(return_corr)
        self.return_dist = bool(return_dist)
        self.return_articulation_confidence = bool(return_articulation_confidence)
        self.articulation_confidence_suffix = str(articulation_confidence_suffix)
        self.compute_cache = bool(compute_cache)
        self.normalize_dist_by_area = bool(normalize_dist_by_area)
        self.cache_dir = self.data_root / "spectral_cache"

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Invalid BeCoS data root: {self.data_root}")
        if self.phase not in self.PHASES:
            raise ValueError(
                f"phase must be one of {sorted(self.PHASES)}, got {self.phase!r}"
            )
        if self.mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.MODES)}, got {self.mode!r}"
            )
        if not self.articulated_extension.startswith("."):
            raise ValueError(
                "articulated_extension must include a leading dot, got "
                f"{self.articulated_extension!r}"
            )
        if self.num_evecs <= 0:
            raise ValueError("num_evecs must be positive")

        self.pairs = self._collect_pairs()
        if not self.pairs:
            raise FileNotFoundError(
                f"No usable BeCoS pairs found under {self.data_root / self.phase}"
            )

    @staticmethod
    def _sort_key(path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    @staticmethod
    def _single_file(pair_dir, pattern):
        paths = sorted(pair_dir.glob(pattern))
        return paths[0] if paths else None

    def _collect_pairs(self):
        phase_dir = self.data_root / self.phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(f"Invalid BeCoS phase directory: {phase_dir}")

        pairs = []
        pair_dirs = sorted(
            (path for path in phase_dir.iterdir() if path.is_dir()),
            key=self._sort_key,
        )
        for pair_dir in pair_dirs:
            mesh_0 = self._single_file(pair_dir, "0_*.off")
            mesh_1 = self._single_file(pair_dir, "1_*.off")
            corr_10 = pair_dir / "corres_10.npy"
            if mesh_0 is None or mesh_1 is None or not corr_10.is_file():
                continue
            if (
                self.mode == "partial_partial"
                and not (pair_dir / "corres_01.npy").is_file()
            ):
                continue
            pairs.append(
                {
                    "pair_id": pair_dir.name,
                    "pair_dir": pair_dir,
                    "mesh_0": mesh_0,
                    "mesh_1": mesh_1,
                    "corr_10": corr_10,
                }
            )
        return pairs

    @staticmethod
    def _load_mesh(mesh_path):
        mesh = trimesh.load(
            mesh_path,
            process=False,
            maintain_order=True,
            force="mesh",
        )
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a mesh at {mesh_path}, got {type(mesh)}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(
                f"Malformed mesh vertices at {mesh_path}: {vertices.shape}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"Malformed mesh faces at {mesh_path}: {faces.shape}")
        return vertices, faces

    @staticmethod
    def _intrinsic_fingerprint(vertices, faces):
        face_vertices = vertices[faces]
        squared_edge_lengths = np.stack(
            [
                np.sum(
                    (face_vertices[:, 0] - face_vertices[:, 1]) ** 2,
                    axis=1,
                ),
                np.sum(
                    (face_vertices[:, 1] - face_vertices[:, 2]) ** 2,
                    axis=1,
                ),
                np.sum(
                    (face_vertices[:, 2] - face_vertices[:, 0]) ** 2,
                    axis=1,
                ),
            ],
            axis=1,
        )
        digest = hashlib.sha256()
        digest.update(np.asarray([len(vertices)], dtype="<i8").tobytes())
        digest.update(np.asarray(faces, dtype="<i8").tobytes())
        digest.update(
            np.asarray(
                np.round(squared_edge_lengths, decimals=12),
                dtype="<f8",
            ).tobytes()
        )
        return digest.hexdigest()

    def _spectral_cache_file(self, fingerprint):
        return self.cache_dir / f"{fingerprint}_spectral.npz"

    def _distance_cache_file(self, fingerprint):
        suffix = "geodesic_area_norm" if self.normalize_dist_by_area else "geodesic_raw"
        return self.cache_dir / f"{fingerprint}_{suffix}.npz"

    @staticmethod
    @contextmanager
    def _cache_lock(cache_file):
        lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")
        with lock_file.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_savez(cache_file, **arrays):
        temporary_file = cache_file.with_name(
            f".{cache_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_file.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_file, cache_file)
        finally:
            temporary_file.unlink(missing_ok=True)

    def _load_spectral_cache(self, cache_file, vertex_count):
        try:
            with np.load(cache_file, allow_pickle=False) as archive:
                if not self.SPECTRAL_FIELDS.issubset(archive.files):
                    return None
                if int(archive["k_eig"]) < self.num_evecs:
                    return None

                mass_np = archive["mass"]
                eigenvalues_np = archive["evals"]
                eigenvectors_np = archive["evecs"]
                if mass_np.shape != (vertex_count,):
                    return None
                if eigenvalues_np.ndim != 1 or len(eigenvalues_np) < self.num_evecs:
                    return None
                if (
                    eigenvectors_np.ndim != 2
                    or eigenvectors_np.shape[0] != vertex_count
                    or eigenvectors_np.shape[1] < self.num_evecs
                ):
                    return None

                item = {"mass": torch.from_numpy(mass_np).float()}
                if self.return_L:
                    laplacian = _read_sparse_matrix(archive, "L")
                    if laplacian.shape != (vertex_count, vertex_count):
                        return None
                    item["L"] = _sparse_numpy_to_torch(laplacian).float()
                if self.return_evecs:
                    eigenvectors = torch.from_numpy(
                        eigenvectors_np[:, : self.num_evecs]
                    ).float()
                    item["evecs"] = eigenvectors
                    item["evecs_trans"] = eigenvectors.T * item["mass"][None]
                    item["evals"] = torch.from_numpy(
                        eigenvalues_np[: self.num_evecs]
                    ).float()
                if self.return_grad:
                    grad_x = _read_sparse_matrix(archive, "gradX")
                    grad_y = _read_sparse_matrix(archive, "gradY")
                    if grad_x.shape != (vertex_count, vertex_count) or grad_y.shape != (
                        vertex_count,
                        vertex_count,
                    ):
                        return None
                    item["gradX"] = _sparse_numpy_to_torch(grad_x).float()
                    item["gradY"] = _sparse_numpy_to_torch(grad_y).float()
                return item
        except (EOFError, KeyError, OSError, ValueError):
            return None

    def _compute_and_cache_spectral(self, vertices, faces, cache_file):
        mass, laplacian, eigenvalues, eigenvectors, grad_x, grad_y = _compute_operators(
            vertices.float(), faces.long(), self.num_evecs
        )
        laplacian_np = _sparse_torch_to_numpy(laplacian).astype(np.float32)
        grad_x_np = _sparse_torch_to_numpy(grad_x).astype(np.float32)
        grad_y_np = _sparse_torch_to_numpy(grad_y).astype(np.float32)
        self._atomic_savez(
            cache_file,
            k_eig=np.int32(self.num_evecs),
            mass=_torch_to_numpy(mass).astype(np.float32),
            evals=_torch_to_numpy(eigenvalues).astype(np.float32),
            evecs=_torch_to_numpy(eigenvectors).astype(np.float32),
            L_data=laplacian_np.data,
            L_indices=laplacian_np.indices,
            L_indptr=laplacian_np.indptr,
            L_shape=laplacian_np.shape,
            gradX_data=grad_x_np.data,
            gradX_indices=grad_x_np.indices,
            gradX_indptr=grad_x_np.indptr,
            gradX_shape=grad_x_np.shape,
            gradY_data=grad_y_np.data,
            gradY_indices=grad_y_np.indices,
            gradY_indptr=grad_y_np.indptr,
            gradY_shape=grad_y_np.shape,
        )
        item = {"mass": mass.float()}
        if self.return_L:
            item["L"] = laplacian.float()
        if self.return_evecs:
            eigenvectors = eigenvectors[:, : self.num_evecs].float()
            item["evecs"] = eigenvectors
            item["evecs_trans"] = eigenvectors.T * item["mass"][None]
            item["evals"] = eigenvalues[: self.num_evecs].float()
        if self.return_grad:
            item["gradX"] = grad_x.float()
            item["gradY"] = grad_y.float()
        return item

    def _get_spectral(self, vertices, faces, fingerprint):
        cache_file = self._spectral_cache_file(fingerprint)
        if cache_file.is_file():
            item = self._load_spectral_cache(cache_file, len(vertices))
            if item is not None:
                return item
        if not self.compute_cache:
            raise FileNotFoundError(
                f"Missing usable spectral cache {cache_file}; "
                "set compute_cache=true to compute it"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(cache_file):
            if cache_file.is_file():
                item = self._load_spectral_cache(cache_file, len(vertices))
                if item is not None:
                    return item
            return self._compute_and_cache_spectral(
                vertices,
                faces,
                cache_file,
            )

    @staticmethod
    def _load_distance_cache(cache_file, vertex_count):
        try:
            with np.load(cache_file, allow_pickle=False) as archive:
                if "dist_mat" not in archive.files:
                    return None
                distances = archive["dist_mat"]
                if distances.shape != (vertex_count, vertex_count):
                    return None
                return torch.from_numpy(distances).float()
        except (EOFError, KeyError, OSError, ValueError):
            return None

    def _get_distances(self, vertices, faces, mass, fingerprint):
        cache_file = self._distance_cache_file(fingerprint)
        if cache_file.is_file():
            distances = self._load_distance_cache(cache_file, len(vertices))
            if distances is not None:
                return distances
        if not self.compute_cache:
            raise FileNotFoundError(
                f"Missing geodesic cache {cache_file}; "
                "set compute_cache=true to compute it"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(cache_file):
            if cache_file.is_file():
                distances = self._load_distance_cache(
                    cache_file,
                    len(vertices),
                )
                if distances is not None:
                    return distances

            distances = _compute_geodesic_distances(
                _torch_to_numpy(vertices).astype(np.float64),
                _torch_to_numpy(faces).astype(np.int64),
            )
            if self.normalize_dist_by_area:
                area = float(mass.sum())
                if not np.isfinite(area) or area <= 0:
                    raise ValueError(f"Invalid mesh area for geodesic scaling: {area}")
                distances = distances / np.sqrt(area)
            distances = distances.astype(np.float32)
            self._atomic_savez(cache_file, dist_mat=distances)
            return torch.from_numpy(distances)

    def _artifact_root(self, pair_id):
        root = self.articulated_shape_dir / self.phase / str(pair_id)
        if self.articulated_subdir:
            root = root / str(self.articulated_subdir)
        return root

    def _load_articulation(self, pair_id, mesh_stem):
        artifact_root = self._artifact_root(pair_id)
        mesh_path = artifact_root / f"{mesh_stem}{self.articulated_extension}"
        if not mesh_path.is_file():
            return None, None, None

        vertices_np, faces_np = self._load_mesh(mesh_path)
        vertices = torch.from_numpy(vertices_np.astype(np.float32))
        faces = torch.from_numpy(faces_np).long()
        confidence = None
        if self.return_articulation_confidence:
            confidence_path = artifact_root / (
                mesh_stem + self.articulation_confidence_suffix
            )
            if not confidence_path.is_file():
                raise FileNotFoundError(
                    f"Missing uncertainty-aware articulation artifact: "
                    f"{confidence_path}"
                )
            with np.load(confidence_path, allow_pickle=False) as archive:
                required = {
                    "vertex_uncertainty",
                    "vertex_visibility_support",
                }
                missing = required.difference(archive.files)
                if missing:
                    raise ValueError(
                        f"Confidence artifact {confidence_path} is missing "
                        f"{sorted(missing)}"
                    )
                confidence = {
                    name: torch.from_numpy(np.asarray(archive[name])).float()
                    for name in required
                }
            lengths = {name: len(value) for name, value in confidence.items()}
            if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != len(
                vertices
            ):
                raise ValueError(
                    f"Confidence topology mismatch in {confidence_path}: "
                    f"{lengths} for {len(vertices)} articulated vertices"
                )
        return vertices, faces, confidence

    def _build_shape(
        self,
        *,
        pair_id,
        mesh_path,
        vertices,
        faces,
        spectral,
    ):
        mesh_stem = mesh_path.stem
        item = {
            "name": f"{pair_id}:{mesh_stem}",
            "mesh_name": mesh_stem,
            "pair_id": str(pair_id),
            "verts": vertices.float(),
        }
        if self.return_faces:
            item["faces"] = faces.long()
        if self.return_L:
            item["L"] = spectral["L"]
        if self.return_mass:
            item["mass"] = spectral["mass"]
        if self.return_evecs:
            for name in ("evecs", "evecs_trans", "evals"):
                item[name] = spectral[name]
        if self.return_grad:
            item["gradX"] = spectral["gradX"]
            item["gradY"] = spectral["gradY"]

        articulated_vertices, articulated_faces, confidence = self._load_articulation(
            pair_id, mesh_stem
        )
        item["articulated_verts"] = articulated_vertices
        item["articulated_faces"] = articulated_faces
        if self.return_articulation_confidence:
            item["articulation_confidence"] = confidence
        return item

    def __getitem__(self, index):
        pair = self.pairs[index]
        vertices_0_np, faces_0_np = self._load_mesh(pair["mesh_0"])
        vertices_1_np, faces_1_np = self._load_mesh(pair["mesh_1"])
        fingerprint_0 = self._intrinsic_fingerprint(vertices_0_np, faces_0_np)
        fingerprint_1 = self._intrinsic_fingerprint(vertices_1_np, faces_1_np)
        vertices_0 = torch.from_numpy(vertices_0_np.astype(np.float32))
        vertices_1 = torch.from_numpy(vertices_1_np.astype(np.float32))
        faces_0 = torch.from_numpy(faces_0_np).long()
        faces_1 = torch.from_numpy(faces_1_np).long()

        needs_spectral = any(
            (
                self.return_L,
                self.return_mass,
                self.return_evecs,
                self.return_grad,
                self.return_dist,
            )
        )
        if needs_spectral:
            spectral_0 = self._get_spectral(
                vertices_0,
                faces_0,
                fingerprint_0,
            )
            spectral_1 = self._get_spectral(
                vertices_1,
                faces_1,
                fingerprint_1,
            )
        else:
            spectral_0 = {}
            spectral_1 = {}

        first = self._build_shape(
            pair_id=pair["pair_id"],
            mesh_path=pair["mesh_0"],
            vertices=vertices_0,
            faces=faces_0,
            spectral=spectral_0,
        )
        second = self._build_shape(
            pair_id=pair["pair_id"],
            mesh_path=pair["mesh_1"],
            vertices=vertices_1,
            faces=faces_1,
            spectral=spectral_1,
        )

        if self.return_corr:
            correspondence = np.load(pair["corr_10"], allow_pickle=False)
            correspondence = np.asarray(correspondence, dtype=np.int64)
            if correspondence.ndim != 1:
                raise ValueError(
                    f"corres_10.npy must be one-dimensional: {pair['corr_10']}"
                )
            if len(correspondence) != len(vertices_1):
                raise ValueError(
                    "corres_10.npy length must match shape 1 vertex count: "
                    f"{len(correspondence)} vs {len(vertices_1)} at "
                    f"{pair['pair_dir']}"
                )
            if len(correspondence):
                minimum = int(correspondence.min())
                maximum = int(correspondence.max())
                if minimum < -1 or maximum >= len(vertices_0):
                    raise ValueError(
                        "corres_10.npy contains out-of-range shape 0 indices at "
                        f"{pair['pair_dir']}: [{minimum}, {maximum}] for "
                        f"{len(vertices_0)} vertices; -1 is the only allowed "
                        "unmatched sentinel"
                    )
            first["corr"] = torch.from_numpy(correspondence).long()
            second["corr"] = torch.arange(len(vertices_1), dtype=torch.long)

        if self.return_dist:
            first["dist"] = self._get_distances(
                vertices_0,
                faces_0,
                spectral_0["mass"],
                fingerprint_0,
            )

        return {
            "first": first,
            "second": second,
            "pair_id": pair["pair_id"],
        }

    def __len__(self):
        return len(self.pairs)


class BeCoSArticulatedShapeCorrDataset:
    """Hydra component for uncertainty-aware matching on BeCoS pairs."""

    def __init__(
        self,
        data_root,
        articulated_shape_dir,
        dataset_name="faust_becos",
        mode="partial_full",
        phase="test",
        articulated_subdir="mhr",
        articulated_extension=".obj",
        return_faces=True,
        return_L=False,
        return_mass=True,
        num_evecs=200,
        return_evecs=True,
        return_grad=False,
        return_corr=True,
        return_dist=True,
        return_articulation_confidence=True,
        articulation_confidence_suffix=".confidence.npz",
        compute_cache=True,
        normalize_dist_by_area=True,
    ):
        self.dataset_name = str(dataset_name)
        self.dataset_kwargs = {
            "data_root": data_root,
            "articulated_shape_dir": articulated_shape_dir,
            "mode": mode,
            "phase": phase,
            "articulated_subdir": articulated_subdir,
            "articulated_extension": articulated_extension,
            "return_faces": return_faces,
            "return_L": return_L,
            "return_mass": return_mass,
            "num_evecs": num_evecs,
            "return_evecs": return_evecs,
            "return_grad": return_grad,
            "return_corr": return_corr,
            "return_dist": return_dist,
            "return_articulation_confidence": (return_articulation_confidence),
            "articulation_confidence_suffix": (articulation_confidence_suffix),
            "compute_cache": compute_cache,
            "normalize_dist_by_area": normalize_dist_by_area,
        }

    def pipeline(self):
        return PairBeCoSArticulatedDataset(**self.dataset_kwargs)
