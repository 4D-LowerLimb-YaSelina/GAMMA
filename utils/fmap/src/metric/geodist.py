# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from src.metric import BaseMetric
from src.infra.registry import METRIC_REGISTRY
from src.utils.fmap import fmap2pointmap, fmap2pointmap_pb
from src.utils.tensor import to_numpy


@METRIC_REGISTRY.register()
class GeodesicDist(BaseMetric):
    """"The original implementation of geodesic error. Doesn't support padded batched computation, as it doesn't consider the vertices mask. It's also much slower than the vectorized version."""
    def __init__(self, pck_threshold=0.10, pck_steps=40):
        super().__init__()
        self.pck_threshold = pck_threshold
        self.pck_steps = pck_steps

    def calculate_geodesic_error(self, dist_x, corr_x, corr_y, p2p, return_mean=True):
        """
        Calculate the geodesic error between predicted correspondence and gt correspondence

        Args:
            dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
            corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
            corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
            p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
            return_mean (bool, optional): Average the geodesic error. Default True.
        Returns:
            avg_geodesic_error (np.ndarray): Average geodesic error.
        """
        ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
        ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
        geo_err = np.take(dist_x, ind21)
        if return_mean:
            return geo_err.mean()
        else:
            return geo_err

    def pck_and_auc(self, dists):
        # compute pck
        dists = np.ravel(dists)
        thresholds = np.linspace(0., self.pck_threshold, self.pck_steps)
        pck_ls = []

        for i in range(thresholds.shape[0]):
            thres = thresholds[i]
            pck = np.mean((dists <= thres).astype(float))
            pck_ls.append(pck)

        pck_arr = np.array(pck_ls)

        # compute auc
        auc = np.trapz(pck_arr, np.linspace(0., 1., self.pck_steps))
        return pck_arr, auc

    def cal(self, Cxy, evecs_x, evecs_y, dist_x, corr_x, corr_y):
        num = Cxy.shape[0]
        dist_sum = 0.0
        auc_sum = 0.0
        pck_arr_sum = np.zeros(self.pck_steps)

        for it in range(num):
            p2p = fmap2pointmap(Cxy[it], evecs_x[it], evecs_y[it])
            dists = self.calculate_geodesic_error(
                to_numpy(dist_x[it]),
                to_numpy(corr_x[it]),
                to_numpy(corr_y[it]),
                to_numpy(p2p),
                return_mean=False,
            )
            pck, auc = self.pck_and_auc(dists)

            dist_sum += dists.mean()
            auc_sum += auc
            pck_arr_sum += pck

        return dist_sum, auc_sum, pck_arr_sum

    def forward(self, infer, data):
        dist_sum, auc_sum, pck_arr_sum = self.cal(
            Cxy=infer['Cxy'],
            evecs_x=data['first']['evecs'],
            evecs_y=data['second']['evecs'],
            dist_x=data['first']['dist'],
            corr_x=data['first']['corr'],
            corr_y=data['second']['corr'],
        )
        
        return dist_sum, pck_arr_sum, auc_sum

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])

    def start_test_feed(self, script, name, rank=None, log_tag: str = 'test'):
        """Method to start feeding the metric method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
            log_tag: The tag for logging the loss. Default is 'test', since it's typically used in the testing epoch. If used in training, it can be left unset. If used in benchmarking, it can be set to 'benchmark/<bench_name>'.
        """
        self.script = script
        self.name = name
        self.rank = rank
        self.log_tag = log_tag

        # init state for feed
        self.metric_total = 0.0
        self.metric_avg = 0.0 # used by script to determine best model
        self.auc_total = 0.0
        self.pck_arr_total = np.zeros(self.pck_steps)
        self.sample_total = 0

    def test_feed(self, infer, data):
        dist_sum, auc_sum, pck_arr_sum = self(infer, data)
        B = self.get_batch_size(infer, data)
        
        self.metric_total += dist_sum
        self.auc_total += auc_sum
        self.pck_arr_total += pck_arr_sum
        self.sample_total += B

        return dist_sum / B

    def end_test_feed(self):
        if self.rank is None:
            # if not using distributed training, simply log the average loss
            self.metric_avg = self.metric_total / self.sample_total
            self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}', self.metric_avg, self.script.global_step)

            self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

            fig = plt.figure()
            plt.plot(
                np.linspace(0., self.pck_threshold, self.pck_steps),
                self.pck_arr_total / self.sample_total,
            )
            plt.xlabel('geodist')
            plt.ylabel('ratio')
            plt.grid(linestyle='--')
            self.script.writer.add_figure(f'metric/{self.log_tag}/{self.name}/pck', fig, self.script.global_step)

        else:
            # if using distributed training, gather the total loss and sample count across all ranks
            self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
            dist.all_reduce(self.metric_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.auc_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.pck_arr_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                # only log in rank 0
                self.metric_avg = self.metric_total / self.sample_total
                self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}', self.metric_avg, self.script.global_step)

                self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

                fig = plt.figure()
                plt.plot(
                    np.linspace(0., self.pck_threshold, self.pck_steps),
                    (self.pck_arr_total / self.sample_total).cpu().numpy(),
                )
                plt.xlabel('geodist')
                plt.ylabel('ratio')
                plt.grid(linestyle='--')
                self.script.writer.add_figure(f'metric/{self.log_tag}/{self.name}/pck', fig, self.script.global_step)


@METRIC_REGISTRY.register()
class GeodesicDist_PB(BaseMetric):
    """Vectorized implementation of geodesic error. Supports padded batched computation and is much faster than the original implementation."""
    def __init__(self, pck_threshold=0.10, pck_steps=40, plot_pck=True):
        super().__init__()
        self.pck_threshold = pck_threshold
        self.pck_steps = pck_steps
        self.plot_pck = plot_pck

    def geodesic_error(self, dist_x, corr_x, corr_y, p2p):
        """
        Batched geodesic error calculation.

        Args:
            dist_x (torch.Tensor): [B, Vx, Vx]
            corr_x (torch.Tensor): [B, V] temple -> shape x
            corr_y (torch.Tensor): [B, V] temple -> shape y
            p2p (torch.Tensor):    [B, Vy]

        Returns:
            torch.Tensor: geodesic error [B, V]
        """
        assert corr_x.shape == corr_y.shape, "corr_x and corr_y must have the same shape"

        # template --(corr_y)--> shape y --(p2p)--> shape x
        # i.e. generate a corr_x* predicted from corr_y and p2p
        B, V = corr_y.shape
        batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V)  # [B, V]
        pred_corr_x = p2p[batch_idx, corr_y]  # [B, V]

        # gather the geodesic distances: dist_x[batch, corr_x, pred_corr_x]
        return dist_x[batch_idx, corr_x, pred_corr_x]  # [B, V]

    def pck_and_auc(self, geo_err):
        thresholds = torch.linspace(0, self.pck_threshold, self.pck_steps, device=geo_err.device)  # [pck_steps]

        # [B, V, 1] vs [1, 1, pck_steps] -> [B, V, pck_steps]
        geo_err = geo_err.unsqueeze(-1)  # [B, V, 1]
        thresholds = thresholds.view(1, 1, -1)  # [1, 1, pck_steps]
        correct = (geo_err <= thresholds).float()  # [B, V, pck_steps]
        pck = correct.mean(dim=1).squeeze(dim=1)  # [B, pck_steps]

        # compute AUC using trapezoidal rule
        auc = torch.trapz(pck, thresholds.squeeze()) / self.pck_threshold  # [B]normalize by max threshold

        return pck, auc

    def cal(self, Cxy, evecs_x, evecs_y, dist_x, corr_x, corr_y, verts_mask_x, verts_mask_y):
        p2p = fmap2pointmap_pb(Cxy, evecs_x, evecs_y, verts_mask_x, verts_mask_y)
        geo_error = self.geodesic_error(dist_x, corr_x, corr_y, p2p)
        pck, auc = self.pck_and_auc(geo_error)

        return geo_error.mean(axis=1), pck, auc

    def forward(self, infer, data):
        geo_error, pck, auc = self.cal(
            Cxy=infer['Cxy'],
            evecs_x=data['first']['evecs'],
            evecs_y=data['second']['evecs'],
            dist_x=data['first']['dist'],
            corr_x=data['first']['corr'],
            corr_y=data['second']['corr'],
            verts_mask_x=data['first']['verts_mask'],
            verts_mask_y=data['second']['verts_mask'],
        )

        return geo_error, pck, auc

    def get_batch_size(self, infer, data):
        return len(infer['Cxy'])

    def start_test_feed(self, script, name, rank=None, log_tag: str = 'test'):
        """Method to start feeding the metric method. Typically called at the beginning of the testing/benchmark epoch
        
        Args:
            script: The traning/benchmark script object
            name: The name of the metric method
            rank: The rank of the process (if using distributed training)
            log_tag: The tag for logging the loss. Default is 'test', since it's typically used in the testing epoch. If used in training, it can be left unset. If used in benchmarking, it can be set to 'benchmark/<bench_name>'.
        """
        self.script = script
        self.name = name
        self.rank = rank
        self.log_tag = log_tag

        # init state for feed
        self.metric_total = 0.0
        self.metric_avg = 0.0 # used by optimizer to determine best model
        self.pck_total = 0.0
        self.auc_total = 0.0
        self.sample_total = 0

    def test_feed(self, infer, data):
        geo_error, pck, auc = self(infer, data)
        B = self.get_batch_size(infer, data)

        self.metric_total += geo_error.sum()
        self.pck_total += pck.sum(axis=0)
        self.auc_total += auc.sum()
        self.sample_total += B

        return geo_error.sum() / B

    def end_test_feed(self):
        if self.rank is None:
            # if not using distributed training, simply log the average loss
            self.metric_avg = self.metric_total / self.sample_total
            self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}', self.metric_avg, self.script.global_step)

            self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

            fig = plt.figure()
            plt.plot(
                np.linspace(0., self.pck_threshold, self.pck_steps),
                (self.pck_total / self.sample_total).cpu().numpy(),
            )
            plt.xlabel('geodist')
            plt.ylabel('ratio')
            plt.grid(linestyle='--')
            self.script.writer.add_figure(f'metric/{self.log_tag}/{self.name}/pck', fig, self.script.global_step)
    
        else:
            # if using distributed training, gather the total loss and sample count across all ranks
            self.sample_total = torch.tensor(self.sample_total).to(device=self.rank)
            dist.all_reduce(self.metric_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.auc_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.pck_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sample_total, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                # only log in rank 0
                self.metric_avg = self.metric_total / self.sample_total
                self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}', self.metric_avg, self.script.global_step)

                self.script.writer.add_scalar(f'metric/{self.log_tag}/{self.name}/auc', self.auc_total / self.sample_total, self.script.global_step)

                fig = plt.figure()
                plt.plot(
                    np.linspace(0., self.pck_threshold, self.pck_steps),
                    (self.pck_total / self.sample_total).cpu().numpy(),
                )
                plt.xlabel('geodist')
                plt.ylabel('ratio')
                plt.grid(linestyle='--')
                self.script.writer.add_figure(f'metric/{self.log_tag}/{self.name}/pck', fig, self.script.global_step)

@METRIC_REGISTRY.register()
class GeodesicDist_s2t_PB(GeodesicDist_PB):
    def geodesic_error(self, dist_x, corr_x, corr_y, p2p):
        """
        Batched geodesic error calculation.

        Args:
            dist_x (torch.Tensor): [B, V_t, V_t]
            corr_x (torch.Tensor): [B, V_x] shape x -> temple
            corr_y (torch.Tensor): [B, V_y] shape y -> temple
            p2p (torch.Tensor):    [B, V_y]

        Returns:
            torch.Tensor: geodesic error [B, V_y]
        """
        # shape y --(p2p)--> shape x --(corr_x)--> template
        # i.e. generate a corr_y* predicted from p2p and corr_x
        B, V_y = corr_y.shape
        batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V_y)  # [B, V_y]
        pred_corr_y = corr_x[batch_idx, p2p]  # [B, V_y]

        # gather the geodesic distances: dist_x[batch, corr_y, pred_corr_y]
        return dist_x[batch_idx, corr_y, pred_corr_y]  # [B, V_y]


@METRIC_REGISTRY.register()
class GeodesicDist_s2t_PB_ZoomOut(GeodesicDist_PB):
    """S2T geodesic error for ZoomOut using the sparse Pyx returned by the network."""

    def pyx_to_pointmap(self, Pyx, verts_mask_y):
        """
        Convert ZoomOut's sparse Pyx list to a padded point map.

        Pyx[i] has shape [V_x, V_y], with nonzeros at [x_idx, y_idx], so it
        directly encodes the point map y -> x.
        """
        B, V_y = verts_mask_y.shape
        p2p = torch.full((B, V_y), -1, dtype=torch.long, device=verts_mask_y.device)

        if len(Pyx) != B:
            raise ValueError(f"Expected {B} sparse Pyx tensors, got {len(Pyx)}")

        for batch_idx, P in enumerate(Pyx):
            P = P.coalesce()
            rows, cols = P.indices()
            rows = rows.to(device=verts_mask_y.device, dtype=torch.long)
            cols = cols.to(device=verts_mask_y.device, dtype=torch.long)

            valid_cols = (cols >= 0) & (cols < V_y)
            cols = cols[valid_cols]
            rows = rows[valid_cols]
            p2p[batch_idx, cols] = rows

        return torch.where(verts_mask_y.bool(), p2p, -1)

    def geodesic_error(self, dist_x, corr_x, corr_y, p2p, verts_mask_y):
        """
        Args:
            dist_x: [B, V_t, V_t] template geodesic distance matrix
            corr_x: [B, V_x] shape x -> template
            corr_y: [B, V_y] shape y -> template
            p2p:    [B, V_y] predicted shape y -> shape x map

        Returns:
            tuple[torch.Tensor, torch.Tensor]: geodesic error [B, V_y], valid mask [B, V_y]
        """
        B, V_y = corr_y.shape
        batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V_y)

        valid = verts_mask_y.bool() & (p2p >= 0) & (corr_y >= 0)
        valid = valid & (p2p < corr_x.shape[1]) & (corr_y < dist_x.shape[1])

        safe_p2p = p2p.clamp(min=0, max=corr_x.shape[1] - 1)
        pred_corr_y = corr_x[batch_idx, safe_p2p]
        valid = valid & (pred_corr_y >= 0) & (pred_corr_y < dist_x.shape[2])

        safe_corr_y = corr_y.clamp(min=0, max=dist_x.shape[1] - 1)
        safe_pred_corr_y = pred_corr_y.clamp(min=0, max=dist_x.shape[2] - 1)
        geo_error = dist_x[batch_idx, safe_corr_y, safe_pred_corr_y]
        geo_error = geo_error.masked_fill(~valid, 0)
        return geo_error, valid

    def pck_and_auc(self, geo_err, valid):
        thresholds = torch.linspace(0, self.pck_threshold, self.pck_steps, device=geo_err.device)
        valid_count = valid.sum(dim=1).clamp_min(1).float()

        correct = (geo_err.unsqueeze(-1) <= thresholds.view(1, 1, -1)) & valid.unsqueeze(-1)
        pck = correct.float().sum(dim=1) / valid_count.unsqueeze(-1)
        auc = torch.trapz(pck, thresholds) / self.pck_threshold
        return pck, auc

    def cal(self, Pyx, dist_x, corr_x, corr_y, verts_mask_y):
        p2p = self.pyx_to_pointmap(Pyx, verts_mask_y)
        geo_error, valid = self.geodesic_error(dist_x, corr_x, corr_y, p2p, verts_mask_y)
        pck, auc = self.pck_and_auc(geo_error, valid)

        valid_count = valid.sum(dim=1).clamp_min(1).float()
        return geo_error.sum(dim=1) / valid_count, pck, auc

    def forward(self, infer, data):
        geo_error, pck, auc = self.cal(
            Pyx=infer['Pyx'],
            dist_x=data['first']['dist'],
            corr_x=data['first']['corr'],
            corr_y=data['second']['corr'],
            verts_mask_y=data['second']['verts_mask'],
        )
        return geo_error, pck, auc


@METRIC_REGISTRY.register()
class BeCoSGeodesicDist_PB(GeodesicDist_PB):
    """BeCoS geodesic error for padded shape-1 -> shape-0 correspondences."""

    def geodesic_error(self, dist_x, corr_x, corr_y, p2p):
        """
        Args:
            dist_x (torch.Tensor): [B, Vx, Vx]
            corr_x (torch.Tensor): [B, V] shape 1 -> shape 0 ground truth
            corr_y (torch.Tensor): [B, V] valid shape 1 vertex ids, padded with -1
            p2p (torch.Tensor):    [B, Vy] predicted shape 1 -> shape 0 map

        Returns:
            tuple[torch.Tensor, torch.Tensor]: geodesic error [B, V], valid mask [B, V]
        """
        assert corr_x.shape == corr_y.shape, "corr_x and corr_y must have the same shape"

        valid = (corr_x >= 0) & (corr_y >= 0)
        safe_corr_x = corr_x.clamp_min(0)
        safe_corr_y = corr_y.clamp_min(0)

        B, V = safe_corr_y.shape
        batch_idx = torch.arange(B, device=corr_y.device).unsqueeze(1).expand(B, V)
        pred_corr_x = p2p[batch_idx, safe_corr_y]

        geo_error = dist_x[batch_idx, safe_corr_x, pred_corr_x]
        geo_error = geo_error.masked_fill(~valid, 0)
        return geo_error, valid

    def pck_and_auc(self, geo_err, valid):
        thresholds = torch.linspace(0, self.pck_threshold, self.pck_steps, device=geo_err.device)
        valid_count = valid.sum(dim=1).clamp_min(1).float()

        correct = (geo_err.unsqueeze(-1) <= thresholds.view(1, 1, -1)) & valid.unsqueeze(-1)
        pck = correct.float().sum(dim=1) / valid_count.unsqueeze(-1)
        auc = torch.trapz(pck, thresholds) / self.pck_threshold

        return pck, auc

    def cal(self, Cxy, evecs_x, evecs_y, dist_x, corr_x, corr_y, verts_mask_x, verts_mask_y):
        p2p = fmap2pointmap_pb(Cxy, evecs_x, evecs_y, verts_mask_x, verts_mask_y)
        geo_error, valid = self.geodesic_error(dist_x, corr_x, corr_y, p2p)
        pck, auc = self.pck_and_auc(geo_error, valid)

        valid_count = valid.sum(dim=1).clamp_min(1).float()
        return geo_error.sum(dim=1) / valid_count, pck, auc
