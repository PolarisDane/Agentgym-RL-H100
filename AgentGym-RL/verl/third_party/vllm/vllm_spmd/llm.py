# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Legacy hybrid-engine LLM interface on top of vLLM>=0.7 SPMD mode.

The agent rollout and FSDPVLLMShardingManager were written against verl's
patched vLLM 0.6.3 wrapper (third_party/vllm/vllm_v_0_6_3/llm.py). vLLM 0.6.3
only ships kernels up to sm_90, so on Blackwell (sm_100 / sm_120) we have to
run a newer vLLM. This class keeps the old surface
    LLM(actor_module, tokenizer, model_hf_config, ...)
    sync_model_weights / offload_model_weights
    init_cache_engine / free_cache_engine
    generate(prompts=None, prompt_token_ids=..., ...) -> (padded token ids, logprobs)
and maps it onto the public vllm.LLM API with
distributed_executor_backend="external_launcher" + sleep mode, so the rollout
code stays unchanged.
"""

import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed._tensor import DTensor
from torch.nn.utils.rnn import pad_sequence

_EXPANDABLE = "expandable_segments:True"


def _expandable_segments_requested() -> bool:
    return _EXPANDABLE in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def _set_expandable_segments(enable: bool) -> None:
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


@contextmanager
def _without_expandable_segments():
    """vLLM's CuMemAllocator (sleep mode) refuses to start when
    PYTORCH_CUDA_ALLOC_CONF has expandable_segments:True
    (pytorch/pytorch#147851). Hide it while the engine allocates its pool and
    turn expandable segments back on afterwards for FSDP."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if conf is None or _EXPANDABLE not in conf:
        yield
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        kv for kv in conf.split(",") if kv.strip() and kv.strip() != _EXPANDABLE)
    _set_expandable_segments(False)
    try:
        yield
    finally:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
        _set_expandable_segments(True)


class LLM:

    def __init__(
        self,
        model,  # actor module; unused, weights arrive through sync_model_weights
        tokenizer,
        model_hf_config,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.9,
        skip_tokenizer_init: bool = False,
        load_format: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        from vllm import LLM as _LLM

        # external_launcher reuses the torch.distributed world set up by the FSDP
        # workers. Each Ray worker sees exactly one GPU, so the local device is 0.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("LOCAL_RANK", "0")

        model_path = model_hf_config._name_or_path
        tokenizer_path = getattr(tokenizer, "name_or_path", None) or model_path
        # verl load formats (dummy_dtensor / dummy_hf / dtensor / hf) -> vLLM ones.
        # Real weights always come from the FSDP actor via sync_model_weights.
        vllm_load_format = "dummy" if load_format.startswith("dummy") else "auto"

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        # V1 cascade attention corrupts decoding on sm_120 (RTX PRO 6000 Blackwell).
        # Qwen3-8B, 32 greedy requests sharing the SciWorld instruction prefix: 21/32
        # outputs degenerate with cascade vs 0/32 without. The heuristic only picks
        # cascade for some (model, batch) shapes -- Qwen3-8B at 32 seqs/GPU, not 16 --
        # so it shows up as silent quality loss when the GPU count changes.
        from vllm.engine.arg_utils import EngineArgs
        if hasattr(EngineArgs, "disable_cascade_attn"):
            kwargs.setdefault("disable_cascade_attn", True)

        with _without_expandable_segments():
            self.llm = _LLM(
                model=model_path,
                tokenizer=tokenizer_path,
                skip_tokenizer_init=skip_tokenizer_init,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                dtype=dtype,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization,
                load_format=vllm_load_format,
                enable_sleep_mode=True,
                seed=seed,
                **kwargs,
            )

        self.pad_token_id = (tokenizer.pad_token_id
                             if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        self._asleep = set()  # subset of {"weights", "kv_cache"}
        self._restore_expandable = _expandable_segments_requested()

    # ---- memory management -------------------------------------------------

    def _wake(self, tags: List[str]) -> None:
        tags = [t for t in tags if t in self._asleep]
        if not tags:
            return
        if self._restore_expandable:
            _set_expandable_segments(False)
        self.llm.wake_up(tags=tags)
        self._asleep.difference_update(tags)

    def offload_model_weights(self) -> None:
        if len(self._asleep) == 2:
            return
        # level 1: weights backed up on CPU, kv cache discarded, prefix cache reset
        self.llm.sleep(level=1)
        self._asleep = {"weights", "kv_cache"}
        if self._restore_expandable:
            _set_expandable_segments(True)

    def init_cache_engine(self) -> None:
        self._wake(["kv_cache"])

    def free_cache_engine(self) -> None:
        # vLLM cannot drop the kv cache without also sleeping the weights; the
        # memory is released by offload_model_weights when the sharding manager
        # exits, right after the rollout returns.
        pass

    @property
    def model(self) -> torch.nn.Module:
        return self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self._wake(["weights"])

        def _full(t):
            # FSDP sharded state dict -> DTensor. full_tensor() is a collective;
            # every rank iterates the state dict in the same order.
            return t.full_tensor() if isinstance(t, DTensor) else t

        self.model.load_weights((name, _full(param)) for name, param in actor_weights.items())

    # ---- generation --------------------------------------------------------

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def generate(self,
                 prompts=None,
                 sampling_params=None,
                 prompt_token_ids: Optional[List[List[int]]] = None,
                 use_tqdm: bool = False,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        from vllm.inputs import TokensPrompt

        assert prompts is None, "the agent rollout feeds token ids only"
        if not prompt_token_ids:
            return torch.empty((0, 0), dtype=torch.long), []
        self._wake(["weights", "kv_cache"])
        inputs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids]
        outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=use_tqdm, **kwargs)
        return self._post_process_outputs(outputs)

    def _post_process_outputs(self, request_outputs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Same contract as vllm_v_0_6_3.LLM._post_process_outputs.
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:
            for output in request_output.outputs:
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprobs.append(
                        torch.tensor([lp[tid].logprob for lp, tid in zip(logprobs_dicts, output.token_ids)]))

        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=self.pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=self.pad_token_id)
        return output_token_ids, logprobs
