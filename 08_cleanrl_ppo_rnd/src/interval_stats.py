"""Accumulate the training metrics over a logging interval instead of sampling the last one.

With a sparse log, a row that reports the last iteration's intrinsic reward and the last minibatch's
losses is a single noisy sample taken every quarter of an hour. Averaging over the whole interval
instead costs nothing and turns each row into a mean over thousands of samples.

The scalars being averaged are GPU tensors, so they are accumulated as GPU tensors and read to the
host once per row. Reading each one as it arrives would put a synchronisation into the innermost
loop, which is exactly what the throughput work removed.
"""

import torch


class IntervalStatistics:
    """Sum named GPU scalars across a logging interval, and read them once at the end."""

    def __init__(self, device, mean_fields, max_fields=()):
        """Prepare accumulators for fields reported as interval means and as interval maxima."""
        self.device = device
        self.mean_fields = list(mean_fields)
        self.max_fields = list(max_fields)
        self._index = {name: i for i, name in enumerate(self.mean_fields)}
        self._sums = torch.zeros(len(self.mean_fields), dtype=torch.float64, device=device)
        self._maxes = {name: torch.full((), float("-inf"), dtype=torch.float64, device=device)
                       for name in self.max_fields}
        self._count = 0

    def add(self, values):
        """Add one sample. `values` maps field name to a scalar tensor (or a float)."""
        # One stack and one add rather than one add per field, so a sample costs two kernels.
        packed = torch.stack([
            torch.as_tensor(values[name], dtype=torch.float64, device=self.device).reshape(())
            for name in self.mean_fields
        ])
        self._sums += packed
        for name in self.max_fields:
            v = torch.as_tensor(values[name], dtype=torch.float64, device=self.device).reshape(())
            self._maxes[name] = torch.maximum(self._maxes[name], v)
        self._count += 1

    def read_and_reset(self, prefix=""):
        """Transfer the interval's means and maxima to the host once, then zero the accumulators."""
        if self._count == 0:
            return {}
        # before: 12 separate .item() calls, each a synchronisation
        # after:  one stack, one .cpu(), one synchronisation for the whole row
        names = list(self.mean_fields) + [f"max_{n}" for n in self.max_fields]
        packed = torch.cat([self._sums / self._count,
                            torch.stack([self._maxes[n] for n in self.max_fields])
                            if self.max_fields else torch.empty(0, dtype=torch.float64,
                                                                device=self.device)]).cpu().tolist()
        out = {prefix + n: v for n, v in zip(names, packed)}
        out[prefix + "samples_averaged"] = self._count
        self._sums.zero_()
        for name in self.max_fields:
            self._maxes[name].fill_(float("-inf"))
        self._count = 0
        return out
