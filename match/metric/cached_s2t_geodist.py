import torch

from match.metric.s2t_geodist import S2TGeodistMetric


class CachedS2TGeodistMetric(S2TGeodistMetric):
    """S2T geodesic metric that reuses one device copy of template distances."""

    def __init__(
        self,
        *args,
        cache_template_dist_on_device: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cache_template_dist_on_device = bool(cache_template_dist_on_device)
        self._template_dist_cache_key = None
        self._template_dist_device = None

    @staticmethod
    def _cache_key(tensor, device):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.dtype,
            str(tensor.device),
            str(device),
        )

    def _prepare_template_dist(self, dist_template, device):
        if not torch.is_tensor(dist_template):
            raise TypeError(
                "Template geodesic distances must be a torch.Tensor, got "
                f"{type(dist_template)}"
            )
        if not self.cache_template_dist_on_device:
            return dist_template.to(device)

        cache_key = self._cache_key(dist_template, device)
        if cache_key != self._template_dist_cache_key:
            self._template_dist_device = dist_template.to(device)
            self._template_dist_cache_key = cache_key
        return self._template_dist_device
