from __future__ import annotations

from typing import Callable, Dict, Iterator, Optional

import torch

from .rollout import Rollout, default_to_rgb


class DynamicalSystem(torch.nn.Module):
    """Minimal batched dynamical-system interface."""

    discrete_state: Optional[bool] = None
    num_states_per_channel: Optional[int | tuple[int, ...]] = None

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)

    @staticmethod
    def sample_params(
        B: int, device: str | torch.device = "cpu", **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def iter_params(
        *, device: str | torch.device = "cpu", batch_size: int = 64, **kwargs
    ) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def seed(
        self,
        B: int = 1,
        generator: Optional[Callable[..., torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def rollout(
        self,
        x0: torch.Tensor,
        params: Dict[str, torch.Tensor],
        steps: int,
        every: int = 1,
        skip: int = 0,
    ) -> Rollout:
        x = x0
        outs = []
        for t in range(int(steps)):
            x = self(x, params)
            if (t + 1) % int(every) == 0 and t >= int(skip):
                outs.append(x)
        tensor = torch.stack(outs, dim=1) if outs else x.unsqueeze(1)
        is_discrete = self.discrete_state
        if is_discrete is None:
            is_discrete = bool(getattr(self, "discrete", not tensor.is_floating_point()))
        num_states = self.num_states_per_channel
        if num_states is None and hasattr(self, "num_states"):
            num_states = int(getattr(self, "num_states"))
        elif num_states is None and is_discrete:
            num_states = 2
        return Rollout(
            tensor=tensor,
            steps=int(steps),
            every=int(every),
            skip=int(skip),
            is_discrete=bool(is_discrete),
            num_states=num_states,
            spatial_dim=max(0, x0.ndim - 2),
            to_rgb_fn=self.to_rgb,
            system_name=type(self).__name__,
        )

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        return default_to_rgb(x)
