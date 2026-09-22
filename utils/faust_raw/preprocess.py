"""Batch-compute spectral operators for raw meshes and geodesic distance for the first template shape."""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy
import scipy.sparse.linalg as sla
import torch
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn import neighbors
from tqdm.auto import tqdm


# These helpers are adapted from baselines/fmap/src/utils so this standalone
# preprocessor can run from the repository root without modifying PYTHONPATH.


def torch2np(tensor):
    """Detach a tensor and return it as a CPU NumPy array."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    return tensor.detach().cpu().numpy()


def sparse_np_to_torch(matrix):
    """Convert a SciPy sparse matrix to a coalesced torch sparse tensor."""
    matrix = matrix.tocoo()
    values = torch.from_numpy(matrix.data)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col))).long()
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size(matrix.shape),
    ).coalesce()


def sparse_torch_to_np(matrix):
    """Convert a two-dimensional torch sparse tensor to SciPy CSC format."""
    if len(matrix.shape) != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(matrix.shape)}")
    matrix = matrix.coalesce()
    indices = torch2np(matrix.indices())
    values = torch2np(matrix.values())
    return scipy.sparse.coo_matrix(
        (values, indices),
        shape=tuple(matrix.shape),
    ).tocsc()


def _dot(first, second, keepdim=False):
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} != {second.shape}")
    return torch.sum(first * second, dim=-1, keepdim=keepdim)


def _normalize(vectors, eps=1e-12):
    return vectors / (torch.linalg.vector_norm(vectors, dim=-1, keepdim=True) + eps)


def _face_normals(verts, faces):
    coords = verts[faces]
    edge_a = coords[:, 1] - coords[:, 0]
    edge_b = coords[:, 2] - coords[:, 0]
    return _normalize(torch.cross(edge_a, edge_b, dim=-1))


def _mesh_vertex_normals(verts, faces):
    face_normals = torch2np(
        _face_normals(torch.from_numpy(verts), torch.from_numpy(faces))
    )
    vertex_normals = np.zeros_like(verts)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    return vertex_normals / (
        np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + 1e-12
    )


def _find_knn(source, target, k, omit_diagonal=False):
    source_np = torch2np(source)
    target_np = torch2np(target)
    tree = neighbors.KDTree(target_np)
    search_k = k + 1 if omit_diagonal else k
    _, indices = tree.query(source_np, k=search_k)

    if omit_diagonal:
        mask = indices != np.arange(indices.shape[0])[:, None]
        mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
        indices = indices[mask].reshape(indices.shape[0], indices.shape[1] - 1)

    return torch.tensor(indices, device=source.device, dtype=torch.long)


def _vertex_normals(verts, faces, n_neighbors=30):
    verts_np = torch2np(verts)

    if faces is None:
        neighbor_indices = _find_knn(
            verts,
            verts,
            n_neighbors,
            omit_diagonal=True,
        )
        neighborhoods = verts_np[torch2np(neighbor_indices)] - verts_np[:, None]
        _, _, vh = np.linalg.svd(neighborhoods, full_matrices=False)
        normals = vh[:, 2]
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    else:
        faces_np = torch2np(faces)
        normals = _mesh_vertex_normals(verts_np, faces_np)

        bad_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_mask.any():
            extent = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(extent) * 1e-4
            random_state = np.random.RandomState(seed=777)
            wiggle = (random_state.rand(*verts.shape) - 0.5) * scale
            normals = _mesh_vertex_normals(
                verts_np + bad_mask * wiggle,
                faces_np,
            )

        bad_mask = np.isnan(normals).any(axis=1)
        if bad_mask.any():
            random_state = np.random.RandomState(seed=777)
            normals[bad_mask] = (
                random_state.rand(*verts.shape) - 0.5
            )[bad_mask]
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12

    normals = torch.from_numpy(normals).to(
        device=verts.device,
        dtype=verts.dtype,
    )
    if torch.isnan(normals).any():
        raise ValueError("NaN vertex normals")
    return normals


def _build_tangent_frames(verts, faces, vertex_normals=None):
    if vertex_normals is None:
        vertex_normals = _vertex_normals(verts, faces)

    vertex_count = len(verts)
    candidate_x = torch.tensor(
        [1, 0, 0],
        device=verts.device,
        dtype=verts.dtype,
    ).expand(vertex_count, -1)
    candidate_y = torch.tensor(
        [0, 1, 0],
        device=verts.device,
        dtype=verts.dtype,
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


def _edge_tangent_vectors(verts, frames, edges):
    edge_vectors = verts[edges[1]] - verts[edges[0]]
    basis_x = frames[edges[0], 0]
    basis_y = frames[edges[0], 1]
    return torch.stack(
        (
            _dot(edge_vectors, basis_x),
            _dot(edge_vectors, basis_y),
        ),
        dim=-1,
    )


def _build_gradient(verts, edges, edge_tangent_vectors):
    vertex_count = len(verts)
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
        lhs_inverse = np.linalg.inv(
            lhs_transposed @ lhs + regularization * np.identity(2)
        ) @ lhs_transposed
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


def _build_point_cloud_gradient(verts, frames, n_neighbors=30):
    neighbor_indices = _find_knn(
        verts,
        verts,
        n_neighbors,
        omit_diagonal=True,
    )
    edges = np.stack(
        (
            np.repeat(np.arange(len(verts)), n_neighbors),
            torch2np(neighbor_indices).ravel(),
        )
    )
    tangent_vectors = _edge_tangent_vectors(verts, frames, edges)
    return _build_gradient(verts, edges, torch2np(tangent_vectors))


def compute_operators(verts, faces, k=120, normals=None):
    """Compute mass, Laplacian, eigenbasis, and gradient operators."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    device = verts.device
    dtype = verts.dtype
    is_cloud = faces is None
    epsilon = 1e-8
    verts_np = torch2np(verts).astype(np.float64)
    faces_np = torch2np(faces) if faces is not None else None
    frames = _build_tangent_frames(verts, faces, vertex_normals=normals)

    if is_cloud:
        laplacian, mass = robust_laplacian.point_cloud_laplacian(verts_np)
        mass_np = mass.diagonal()
    else:
        laplacian = pp3d.cotan_laplacian(
            verts_np,
            faces_np,
            denom_eps=1e-10,
        )
        mass_np = pp3d.vertex_areas(verts_np, faces_np)
        mass_np += epsilon * np.mean(mass_np)

    if np.isnan(laplacian.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(mass_np).any():
        raise RuntimeError("NaN mass matrix")

    laplacian_coo = laplacian.tocoo()
    if k > 0:
        laplacian_eigsh = (
            laplacian
            + epsilon * scipy.sparse.identity(laplacian.shape[0])
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
                    raise ValueError(
                        "Failed to compute eigendecomposition"
                    ) from error
                failure_count += 1
                print("Decomposition failed; adding epsilon")
                laplacian_eigsh = laplacian_eigsh + (
                    epsilon
                    * 10**failure_count
                    * scipy.sparse.identity(laplacian.shape[0])
                )
    else:
        eigenvalues_np = np.zeros(0)
        eigenvectors_np = np.zeros((len(verts), 0))

    if is_cloud:
        gradient_np = _build_point_cloud_gradient(verts, frames)
    else:
        edges = torch.tensor(
            np.stack((laplacian_coo.row, laplacian_coo.col)),
            device=device,
            dtype=faces.dtype,
        )
        tangent_vectors = _edge_tangent_vectors(verts, frames, edges)
        gradient_np = _build_gradient(
            verts_np,
            torch2np(edges),
            torch2np(tangent_vectors),
        )

    mass = torch.from_numpy(mass_np).to(device=device, dtype=dtype)
    laplacian = sparse_np_to_torch(laplacian).to(device=device, dtype=dtype)
    eigenvalues = torch.from_numpy(eigenvalues_np).to(
        device=device,
        dtype=dtype,
    )
    eigenvectors = torch.from_numpy(eigenvectors_np).to(
        device=device,
        dtype=dtype,
    )
    grad_x = sparse_np_to_torch(np.real(gradient_np)).to(
        device=device,
        dtype=dtype,
    )
    grad_y = sparse_np_to_torch(np.imag(gradient_np)).to(
        device=device,
        dtype=dtype,
    )
    return (
        frames,
        mass,
        laplacian,
        eigenvalues,
        eigenvectors,
        grad_x,
        grad_y,
    )


def compute_geodesic_distmat(verts, faces):
    """Compute all-pairs geodesic distances over mesh edges."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    adjacency = mesh.vertex_adjacency_graph
    if not nx.is_connected(adjacency):
        raise ValueError("Registration mesh graph is not connected")

    vertex_count = len(verts)
    adjacency_matrix = nx.adjacency_matrix(
        adjacency,
        range(vertex_count),
    )
    knn_graph = neighbors.kneighbors_graph(
        verts,
        n_neighbors=min(500, vertex_count - 1),
        mode="distance",
        include_self=False,
    )
    distance_adjacency = csr_matrix((vertex_count, vertex_count)).tolil()
    distance_adjacency[adjacency_matrix != 0] = knn_graph[
        adjacency_matrix != 0
    ]
    distances = shortest_path(distance_adjacency, directed=False)
    if np.isinf(distances).any():
        raise ValueError("Infinite values in geodesic distance matrix")
    return distances


def _plot_lbo(mesh, name, output_dir, lbo, shape_disp, cmap, camera_position, camera_zoom, window_size, max_k):
    """Plot LBO eigenvectors on a trimesh mesh and save as PNG.

    Uses pv.wrap() to convert the trimesh object directly, avoiding the vertex
    count mismatch that occurs when pv.read() re-reads the file from disk.
    """
    import pyvista as pv

    pv_mesh = pv.wrap(mesh)

    lbo_min = lbo[:, :max_k].min().cpu().numpy()
    lbo_max = lbo[:, :max_k].max().cpu().numpy()

    pl = pv.Plotter(off_screen=True)
    for dim in range(max_k):
        pv_mesh[f'lbo-{dim}'] = lbo[:, dim].cpu().numpy()
        pl.add_mesh(
            mesh=pv_mesh.translate(dim * shape_disp),
            scalars=f'lbo-{dim}',
            cmap=cmap,
            clim=[lbo_min, lbo_max],
            show_scalar_bar=False,
        )
    pl.camera_position = camera_position
    pl.zoom_camera(camera_zoom)
    pl.screenshot(output_dir / f'{name}.png', window_size=window_size, return_img=False)
    pl.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-compute spectral operators for raw meshes and geodesic distance for the first template shape."
    )
    parser.add_argument("--raw_mesh_folder", type=str, required=True,
                        help="Folder with raw mesh files")
    parser.add_argument("--raw_mesh_type", type=str, default="ply",
                        help="Mesh file extension (default: ply)")
    parser.add_argument("--regist_mesh_folder", type=str, required=True,
                        help="Folder with template/registration mesh files")
    parser.add_argument("--regist_mesh_type", type=str, default="ply",
                        help="Mesh file extension (default: ply)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Root output directory (default: parent of raw_mesh_folder)")
    parser.add_argument("--k_eig", type=int, default=128,
                        help="Number of eigenvectors/values to compute (default: 128)")
    parser.add_argument("--plot_lbo", action="store_true",
                        help="Export LBO visualization")
    parser.add_argument("--pcd_mode", action="store_true",
                        help="Compute point clouds LBO")
    args = parser.parse_args()

    raw_mesh_folder = Path(args.raw_mesh_folder)
    regist_mesh_folder = Path(args.regist_mesh_folder)
    output_dir = Path(args.output_dir) if args.output_dir else raw_mesh_folder.parent
    k_eig = args.k_eig

    assert k_eig > 0, f"invalid k_eig: {k_eig}"
    assert raw_mesh_folder.is_dir(), f"invalid raw_mesh_folder: {raw_mesh_folder}"
    assert regist_mesh_folder.is_dir(), f"invalid regist_mesh_folder: {regist_mesh_folder}"

    spectral_dir = output_dir / "spectral"
    dist_dir = output_dir / "dist"
    spectral_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Spectral operators for each raw mesh ---
    mesh_files = sorted(raw_mesh_folder.glob(f"*.{args.raw_mesh_type}"))
    assert len(mesh_files) > 0, f"No *.{args.raw_mesh_type} files found in {raw_mesh_folder}"

    for mesh_file in tqdm(mesh_files, desc="Spectral operators"):
        try:
            stem = mesh_file.stem

            mesh = trimesh.load(mesh_file)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            frames, mass_vec, L, evals, evecs, gradX, gradY = compute_operators(
                torch.from_numpy(verts).float(),
                torch.from_numpy(faces).long() if not args.pcd_mode else None,
                k=k_eig,
            )

            frames_np = torch2np(frames).astype(np.float32)
            mass_np = torch2np(mass_vec).astype(np.float32)
            evals_np = torch2np(evals).astype(np.float32)
            evecs_np = torch2np(evecs).astype(np.float32)
            L_np = sparse_torch_to_np(L).astype(np.float32)
            gradX_np = sparse_torch_to_np(gradX).astype(np.float32)
            gradY_np = sparse_torch_to_np(gradY).astype(np.float32)

            np.savez(
                spectral_dir / f"{stem}.npz",
                verts=verts,
                faces=faces,
                k_eig=k_eig,
                frames=frames_np,
                mass=mass_np,
                evals=evals_np,
                evecs=evecs_np,
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
            )

            # Plot LBO if requested
            if args.plot_lbo:
                lbo_vis_dir = output_dir / "lbo_vis"
                lbo_vis_dir.mkdir(parents=True, exist_ok=True)

                _plot_lbo(
                    mesh=mesh,
                    name=stem,
                    output_dir=lbo_vis_dir,
                    lbo=evecs,
                    shape_disp=np.array([1, 0, 0]),
                    cmap="coolwarm",
                    camera_position="xy",
                    camera_zoom=2.0,
                    window_size=[1024, 512],
                    max_k=min(10, k_eig),
                )

        except Exception as e:
            print(f"Error processing {mesh_file}: {e}")

    # --- 2. Template geodesic distance (first sorted registration mesh) ---
    regist_files = sorted(regist_mesh_folder.glob(f"*.{args.regist_mesh_type}"))
    assert len(regist_files) > 0, f"No *.{args.regist_mesh_type} files found in {regist_mesh_folder}"

    template_file = regist_files[0]
    print(f"Computing geodesic distance matrix for template: {template_file.name}")

    mesh = trimesh.load(template_file)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    dist_mat = compute_geodesic_distmat(verts, faces)
    np.savez(
        dist_dir / "template_dist.npz",
        dist_mat=dist_mat,
    )

    print(f"Done. Spectral: {spectral_dir}, Dist: {dist_dir}")


if __name__ == "__main__":
    main()
