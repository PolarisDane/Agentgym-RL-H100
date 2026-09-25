"""Separate action-forecast (AF) LoRA: the decoupled control for plan_forecast.

Ablation question: does the plan-forecast gain come from the forecast gradient
reshaping the POLICY's own parameters (shared backbone), or would a forecaster
that merely reads the policy's features do just as well?

Here the forecast head is a LoRA adapter that
  * sees the CURRENT policy weights (so its features track the policy), but
  * receives ALL of the forecast gradient itself -- the backbone gets none,
so the only difference from the shared-backbone method is the gradient path.

The adapter is active ONLY inside the plan-forecast forward/backward. Rollout,
PG update, ref / old-logprob passes and the FSDP->vLLM weight sync never see it,
so with af_lora on, the policy update is exactly GRPO.

B is zero-initialised, so at step 0 the forecast distribution is identical to the
shared-backbone variant's.

Implementation note: the adapter hangs off forward hooks on the inner nn.Linear
modules of the FSDP-wrapped actor. FSDP1 (use_orig_params=False) keeps the module
tree intact and rebinds `weight` to a view of the flat parameter during forward,
so the hooks fire normally, including on the recompute pass under gradient
checkpointing. The backward still populates the flat params' grads (FSDP cannot
freeze part of a flat parameter); update_plan_forecast discards them and steps
only the LoRA optimizer, and asserts the policy weights did not move.
"""
from contextlib import contextmanager
from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class AFLoRA(nn.Module):
    """LoRA adapters over the actor's linear projections, enabled on demand."""

    def __init__(self,
                 actor_module: nn.Module,
                 rank: int = 64,
                 alpha: float = 128.0,
                 targets: Tuple[str, ...] = DEFAULT_TARGETS,
                 dtype: torch.dtype = torch.float32,
                 device=None):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.targets = tuple(targets)
        self._enabled = False
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        device = device if device is not None else torch.cuda.current_device()

        self._modules_by_key: Dict[str, nn.Module] = {}
        for name, mod in actor_module.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(name.endswith(t) for t in self.targets):
                continue
            key = name.replace('.', '__')
            a = nn.Parameter(torch.empty(self.rank, mod.in_features, dtype=dtype, device=device))
            b = nn.Parameter(torch.zeros(mod.out_features, self.rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))     # B stays 0 -> delta == 0 at init
            self.A[key] = a
            self.B[key] = b
            self._modules_by_key[key] = mod

        if not self._modules_by_key:
            raise ValueError(f"AFLoRA matched no Linear module with suffixes {self.targets}")

    # ---- wiring ---------------------------------------------------------

    def _make_hook(self, key: str):
        def hook(module, args, output):
            if not self._enabled:
                return output
            x = args[0]
            a, b = self.A[key], self.B[key]
            delta = F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype))
            return output + self.scaling * delta

        return hook

    def attach(self) -> None:
        """Register the hooks once; they are inert while `_enabled` is False."""
        if self._handles:
            return
        for key, mod in self._modules_by_key.items():
            self._handles.append(mod.register_forward_hook(self._make_hook(key)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def enabled(self):
        prev, self._enabled = self._enabled, True
        try:
            yield
        finally:
            self._enabled = prev

    # ---- bookkeeping ----------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def all_reduce_grads(self) -> None:
        """LoRA params are replicated per rank; average their grads like DDP."""
        if not torch.distributed.is_initialized():
            return
        world = torch.distributed.get_world_size()
        if world == 1:
            return
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(flat)
        flat.div_(world)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to('cpu', torch.bfloat16) for k, v in self.state_dict().items()}


def policy_fingerprint(actor_module: nn.Module, n_params: int = 8, n_elem: int = 64) -> torch.Tensor:
    """A cheap slice of the policy's own parameters, for the no-leak assertion.

    Reads the local shards (flat params under FSDP), so it is rank-local and free
    of collectives -- safe to call inside the forecast update.
    """
    out = []
    for i, p in enumerate(actor_module.parameters()):
        if i >= n_params:
            break
        flat = p.detach().reshape(-1)
        out.append(flat[:min(n_elem, flat.numel())].float().clone())
    return torch.cat(out) if out else torch.zeros(1)
