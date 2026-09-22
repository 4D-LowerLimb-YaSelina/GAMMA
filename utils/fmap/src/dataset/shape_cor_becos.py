import fcntl
import hashlib
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from src.infra.registry import DATASET_REGISTRY
from src.utils.geometry import compute_operators
from src.utils.shape import compute_geodesic_distmat
from src.utils.tensor import read_sp_mat, sparse_np_to_torch, sparse_torch_to_np, torch2np


@DATASET_REGISTRY.register()
class PairBeCoS(Dataset):
    """
    BeCoS pair dataset for partial_full and partial_partial outputs.

    Direction is deterministic: shape 1 -> shape 0 using corres_10.npy.
    """

    def __init__(
            self,
            data_root,
            phase: str = 'train',
            mode: str = 'partial_full',
            return_faces: bool = True,
            return_L: bool = False,
            return_mass: bool = True,
            num_evecs: int = 200,
            return_evecs: bool = True,
            return_grad: bool = True,
            return_corr: bool = False,
            return_dist: bool = False,
            compute_cache: bool = False,
            normalize_dist_by_area: bool = True,
        ):
        self.data_root = Path(data_root)
        assert self.data_root.is_dir(), f'invalid data root: {data_root}.'

        assert phase in ['train', 'val', 'test'], f'invalid phase: {phase}'
        assert mode in ['partial_full', 'partial_partial'], f'invalid BeCoS mode: {mode}'

        self.phase = phase
        self.mode = mode
        self.return_faces = return_faces
        self.return_L = return_L
        self.return_mass = return_mass
        self.num_evecs = num_evecs
        self.return_evecs = return_evecs
        self.return_grad = return_grad
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.compute_cache = compute_cache
        self.normalize_dist_by_area = normalize_dist_by_area
        self.cache_dir = self.data_root / 'spectral_cache'

        self.pairs = self._collect_pairs()
        assert len(self.pairs) > 0, f'no usable BeCoS pairs found under {self.data_root / phase}'

    @staticmethod
    def _sort_key(path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    @staticmethod
    def _single_file(pair_dir, pattern):
        files = sorted(pair_dir.glob(pattern))
        return files[0] if files else None

    def _collect_pairs(self):
        phase_dir = self.data_root / self.phase
        assert phase_dir.is_dir(), f'invalid BeCoS phase folder: {phase_dir}'

        pairs = []
        for pair_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()], key=self._sort_key):
            mesh_0 = self._single_file(pair_dir, '0_*.off')
            mesh_1 = self._single_file(pair_dir, '1_*.off')
            corr_10 = pair_dir / 'corres_10.npy'

            if mesh_0 is None or mesh_1 is None or not corr_10.is_file():
                continue

            if self.mode == 'partial_partial' and not (pair_dir / 'corres_01.npy').is_file():
                continue

            pairs.append({
                'pair_dir': pair_dir,
                'mesh_0': mesh_0,
                'mesh_1': mesh_1,
                'corr_10': corr_10,
            })

        return pairs

    @staticmethod
    def _load_mesh(mesh_path):
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True, force='mesh')
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
            raise ValueError(f'malformed mesh vertices at {mesh_path}: {verts.shape}')
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f'malformed mesh faces at {mesh_path}: {faces.shape}')

        fingerprint = PairBeCoS._intrinsic_fingerprint(verts, faces)
        return (
            torch.from_numpy(verts.astype(np.float32)),
            torch.from_numpy(faces),
            fingerprint,
        )

    @staticmethod
    def _intrinsic_fingerprint(verts, faces):
        face_verts = verts[faces]
        squared_edge_lengths = np.stack(
            [
                np.sum((face_verts[:, 0] - face_verts[:, 1]) ** 2, axis=1),
                np.sum((face_verts[:, 1] - face_verts[:, 2]) ** 2, axis=1),
                np.sum((face_verts[:, 2] - face_verts[:, 0]) ** 2, axis=1),
            ],
            axis=1,
        )

        digest = hashlib.sha256()
        digest.update(np.asarray([verts.shape[0]], dtype='<i8').tobytes())
        digest.update(np.asarray(faces, dtype='<i8').tobytes())
        digest.update(
            np.asarray(np.round(squared_edge_lengths, decimals=12), dtype='<f8').tobytes()
        )
        return digest.hexdigest()

    def _spectral_cache_file(self, fingerprint):
        return self.cache_dir / f'{fingerprint}_spectral.npz'

    def _dist_cache_file(self, fingerprint):
        suffix = 'geodesic_area_norm' if self.normalize_dist_by_area else 'geodesic_raw'
        return self.cache_dir / f'{fingerprint}_{suffix}.npz'

    @staticmethod
    @contextmanager
    def _cache_lock(cache_file):
        lock_file = cache_file.with_suffix(cache_file.suffix + '.lock')
        with lock_file.open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_savez(cache_file, **arrays):
        temporary_file = cache_file.with_name(
            f'.{cache_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
        )
        try:
            with temporary_file.open('wb') as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_file, cache_file)
        finally:
            temporary_file.unlink(missing_ok=True)

    def _load_spectral_cache(self, cache_file, num_verts):
        required_fields = {
            'k_eig',
            'mass',
            'evals',
            'evecs',
            'L_data',
            'L_indices',
            'L_indptr',
            'L_shape',
            'gradX_data',
            'gradX_indices',
            'gradX_indptr',
            'gradX_shape',
            'gradY_data',
            'gradY_indices',
            'gradY_indptr',
            'gradY_shape',
        }

        try:
            with np.load(cache_file, allow_pickle=False) as npz:
                if not required_fields.issubset(npz.files):
                    return None
                if int(npz['k_eig']) < self.num_evecs:
                    return None

                mass_np = npz['mass']
                evals_np = npz['evals']
                evecs_np = npz['evecs']
                if mass_np.shape != (num_verts,):
                    return None
                if evals_np.ndim != 1 or evals_np.shape[0] < self.num_evecs:
                    return None
                if evecs_np.ndim != 2 or evecs_np.shape[0] != num_verts:
                    return None
                if evecs_np.shape[1] < self.num_evecs:
                    return None

                item = {'mass': torch.from_numpy(mass_np).float()}
                if self.return_L:
                    item['L'] = sparse_np_to_torch(read_sp_mat(npz, 'L'))
                if self.return_evecs:
                    evecs = torch.from_numpy(
                        evecs_np[:, :self.num_evecs]
                    ).float()
                    item['evecs'] = evecs
                    item['evecs_trans'] = evecs.T * item['mass'][None]
                    item['evals'] = torch.from_numpy(
                        evals_np[:self.num_evecs]
                    ).float()
                if self.return_grad:
                    item['gradX'] = sparse_np_to_torch(read_sp_mat(npz, 'gradX'))
                    item['gradY'] = sparse_np_to_torch(read_sp_mat(npz, 'gradY'))

                return item
        except (EOFError, KeyError, OSError, ValueError):
            return None

    def _compute_and_cache_spectral(self, verts, faces, cache_file):
        _, mass, L, evals, evecs, gradX, gradY = compute_operators(
            verts.float(),
            faces.long(),
            k=self.num_evecs,
        )

        L_np = sparse_torch_to_np(L).astype(np.float32)
        gradX_np = sparse_torch_to_np(gradX).astype(np.float32)
        gradY_np = sparse_torch_to_np(gradY).astype(np.float32)

        self._atomic_savez(
            cache_file,
            k_eig=np.int32(self.num_evecs),
            mass=torch2np(mass).astype(np.float32),
            evals=torch2np(evals).astype(np.float32),
            evecs=torch2np(evecs).astype(np.float32),
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

        item = {'mass': mass.float()}
        if self.return_L:
            item['L'] = L.float()
        if self.return_evecs:
            evecs = evecs[:, :self.num_evecs].float()
            item['evecs'] = evecs
            item['evecs_trans'] = evecs.T * item['mass'][None]
            item['evals'] = evals[:self.num_evecs].float()
        if self.return_grad:
            item['gradX'] = gradX.float()
            item['gradY'] = gradY.float()
        return item

    def _get_spectral(self, verts, faces, fingerprint):
        cache_file = self._spectral_cache_file(fingerprint)
        if cache_file.is_file():
            item = self._load_spectral_cache(cache_file, verts.shape[0])
            if item is not None:
                return item

        if not self.compute_cache:
            raise FileNotFoundError(
                f'missing usable spectral cache {cache_file}; set compute_cache=True to compute it'
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(cache_file):
            if cache_file.is_file():
                item = self._load_spectral_cache(cache_file, verts.shape[0])
                if item is not None:
                    return item
            return self._compute_and_cache_spectral(verts, faces, cache_file)

    @staticmethod
    def _load_dist_cache(cache_file, num_verts):
        try:
            with np.load(cache_file, allow_pickle=False) as npz:
                if 'dist_mat' not in npz.files:
                    return None
                dist = npz['dist_mat']
                if dist.shape != (num_verts, num_verts):
                    return None
                return torch.from_numpy(dist).float()
        except (EOFError, KeyError, OSError, ValueError):
            return None

    def _get_dist(self, verts, faces, mass, fingerprint):
        cache_file = self._dist_cache_file(fingerprint)
        if cache_file.is_file():
            dist = self._load_dist_cache(cache_file, verts.shape[0])
            if dist is not None:
                return dist

        if not self.compute_cache:
            raise FileNotFoundError(
                f'missing geodesic cache {cache_file}; set compute_cache=True to compute it'
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_lock(cache_file):
            if cache_file.is_file():
                dist = self._load_dist_cache(cache_file, verts.shape[0])
                if dist is not None:
                    return dist

            dist = compute_geodesic_distmat(
                verts.numpy().astype(np.float64),
                faces.numpy().astype(np.int64),
            )

            if self.normalize_dist_by_area:
                dist = dist / np.sqrt(float(mass.sum()))

            dist = dist.astype(np.float32)
            self._atomic_savez(cache_file, dist_mat=dist)
            return torch.from_numpy(dist)

    def _build_shape(self, name, verts, faces, spectral):
        item = {
            'name': name,
            'verts': verts.float(),
        }

        if self.return_faces:
            item['faces'] = faces.long()

        requested_keys = []
        if self.return_L:
            requested_keys.append('L')
        if self.return_mass:
            requested_keys.append('mass')
        if self.return_evecs:
            requested_keys.extend(['evecs', 'evecs_trans', 'evals'])
        if self.return_grad:
            requested_keys.extend(['gradX', 'gradY'])

        for key in requested_keys:
            if key in spectral:
                item[key] = spectral[key]

        return item

    def __getitem__(self, index):
        pair = self.pairs[index]
        pair_dir = pair['pair_dir']

        verts_0, faces_0, fingerprint_0 = self._load_mesh(pair['mesh_0'])
        verts_1, faces_1, fingerprint_1 = self._load_mesh(pair['mesh_1'])

        spec_0 = self._get_spectral(
            verts_0,
            faces_0,
            fingerprint_0,
        )
        spec_1 = self._get_spectral(
            verts_1,
            faces_1,
            fingerprint_1,
        )

        first = self._build_shape(pair['mesh_0'].stem, verts_0, faces_0, spec_0)
        second = self._build_shape(pair['mesh_1'].stem, verts_1, faces_1, spec_1)

        if self.return_corr:
            corr_10 = torch.from_numpy(np.load(pair['corr_10']).astype(np.int64)).long()
            if corr_10.ndim != 1:
                raise ValueError(f'corres_10 must be a 1D p2p array: {pair["corr_10"]}')
            if corr_10.shape[0] != verts_1.shape[0]:
                raise ValueError(
                    f'corres_10 length must match shape 1 vertex count: '
                    f'{corr_10.shape[0]} vs {verts_1.shape[0]} at {pair_dir}'
                )

            first['corr'] = corr_10
            second['corr'] = torch.arange(verts_1.shape[0], dtype=torch.long)

        if self.return_dist:
            first['dist'] = self._get_dist(
                verts_0,
                faces_0,
                spec_0['mass'],
                fingerprint_0,
            )

        return {
            'first': first,
            'second': second,
            'pair_id': pair_dir.name,
        }

    def __len__(self):
        return len(self.pairs)
