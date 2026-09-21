#!/usr/bin/env python3
"""Evaluate a local vLLM model on SciWorld.

This script is dedicated to local testing using checkpoints. It bypasses all
OpenAI API logic and uses the vLLM library directly for inference.

Features:
  * Uses vLLM for local model inference.
  * Supports SciWorld environment parallelism via multiple env servers.
  * Supports resuming from partially completed runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Setup path for internal packages
REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTENV_PKG = REPO_ROOT / "AgentGym" / "agentenv"
if AGENTENV_PKG.exists() and str(AGENTENV_PKG) not in sys.path:
    sys.path.insert(0, str(AGENTENV_PKG))

from agentenv.envs import SciworldEnvClient  # noqa: E402

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vllm is not installed. Please install it with 'pip install vllm'")
    sys.exit(1)

# Defaults
# Official AgentGym-RL-Data-ID eval split (200 items, identical to AgentEval's
# sciworld_test.json). The legacy in-repo path is kept as a fallback.
DEFAULT_TEST_FILE = Path(
    os.environ.get(
        "SCIWORLD_TEST_FILE",
        "/data1/datasets/AgentGym-RL-Data-ID/eval/sciworld_test.json",
    )
)
LEGACY_TEST_FILE = REPO_ROOT / "AgentItemId" / "test" / "sciworld_test.json"
if not DEFAULT_TEST_FILE.exists() and LEGACY_TEST_FILE.exists():
    DEFAULT_TEST_FILE = LEGACY_TEST_FILE
DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 2400


# ---------------------------------------------------------------------------
# 2026-09-20: thinking 模型（Qwen3）的多轮 prompt 必须与训练 rollout 逐 token 一致，
# 统一由 verl/workers/rollout/token_io.py 构造（训练侧 RolloutHandler 用的也是它）。
# 按文件路径加载，避免 import verl 包带来的副作用。非 thinking 模型（Qwen2.5）不经过这里。
import importlib.util as _ilu
_TIO_PATH = Path(__file__).resolve().parent.parent / "AgentGym-RL" / "verl" / "workers" / "rollout" / "token_io.py"
_spec = _ilu.spec_from_file_location("_token_io", _TIO_PATH)
token_io = _ilu.module_from_spec(_spec); _spec.loader.exec_module(token_io)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Qwen3 等混合推理模型默认先生成 <think>...</think>，会把每轮的生成预算耗光。
# 与训练 rollout（verl/workers/rollout/schemas.py::_thinking_kwargs）同一逻辑：
# 探测模板是否认 enable_thinking —— 认就关掉，不认（Qwen2.5 等）返回 {}、行为不变。
_NOTHINK_CACHE: dict = {}
def _nothink(tok) -> dict:
    key = getattr(tok, "name_or_path", None) or id(tok)
    if key not in _NOTHINK_CACHE:
        p = [{"role": "user", "content": "x"}]
        try:
            a = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            b = tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False)
            _NOTHINK_CACHE[key] = {"enable_thinking": False} if a != b else {}
        except Exception:
            _NOTHINK_CACHE[key] = {}
    return _NOTHINK_CACHE[key]
# ---------------------------------------------------------------------------


class LocalModel:
    """Wrapper for vLLM model to handle thread-safe inference."""
    def __init__(self, model_path: str, tp: int = 1, gpu_util: float = 0.9):
        self._ensure_weights(model_path)
        print(f"[vllm] Loading model from {model_path} (tp={tp}, util={gpu_util})...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._lock = threading.Lock()

    def _ensure_weights(self, model_path: str):
        """Check for weights and try to merge shards if missing."""
        path = Path(model_path)
        weight_files = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]
        if any((path / f).exists() for f in weight_files):
            return

        # Weights missing, check for shards in parent directory
        actor_dir = path.parent
        shard_files = list(actor_dir.glob("model_world_size_*_rank_0.pt"))
        if not shard_files:
            print(f"ERROR: No weights found in {model_path} and no shards found in {actor_dir}")
            return

        print(f"[merge] Weights missing in {model_path}, attempting to merge shards from {actor_dir}...")
        import subprocess
        # Correct path to model_merger.py based on repository structure
        merger_script = REPO_ROOT / "AgentGym-RL" / "scripts" / "model_merger.py"
        if not merger_script.exists():
            print(f"ERROR: Merger script not found at {merger_script}")
            return
            
        try:
            # Force the working directory to the training code dir where the script expects to run
            train_code_dir = REPO_ROOT / "AgentGym-RL"
            subprocess.check_call([
                sys.executable, str(merger_script),
                "--local_dir", str(actor_dir)
            ], cwd=str(train_code_dir))
            print("[merge] Successfully merged weights.")
        except Exception as e:
            print(f"ERROR: Failed to merge weights: {e}")
            return # Exit if merging failed to avoid vLLM error later

        # Re-verify weight existence after merge attempt
        if not any((path / f).exists() for f in weight_files):
            print(f"ERROR: Weights still missing in {model_path} after merge attempt.")
            return

    def generate_ids(self, prompt_ids, sampling_params) -> list:
        """token 进 token 出（thinking 模型路径用）：返回 vLLM 原始生成 token。"""
        from vllm.inputs import TokensPrompt
        with self._lock:
            outputs = self.llm.generate(TokensPrompt(prompt_token_ids=list(prompt_ids)),
                                        sampling_params, use_tqdm=False)
        return list(outputs[0].outputs[0].token_ids)

    def generate(self, messages: list[dict[str, str]], sampling_params: SamplingParams) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **_nothink(self.tokenizer)
        )
        with self._lock:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

# ---------------------------------------------------------------------------
def _is_success(score: float, done: bool) -> int:
    """Same criterion as the training reward (vllm_rollout._shape_task_reward)."""
    return 1 if done and score >= 100.0 else 0


def _record_success(rec: dict[str, Any]) -> int:
    # Recomputed from the score rather than read from ``success``: records written
    # before this fix stored ``score >= 1`` there. They also lack ``done``, which
    # ``terminated_by == "env_done"`` recovers.
    done = rec.get("done", rec.get("terminated_by") == "env_done")
    return _is_success(float(rec["reward"]), bool(done))


def run_trajectory(
    env_client: SciworldEnvClient,
    model: LocalModel,
    sampling_params: SamplingParams,
    item_idx: int,
    max_rounds: int,
    obs_source: str = "step",
) -> dict[str, Any]:
    """Run one SciWorld trajectory."""
    reset_info = env_client.reset(item_idx)
    state = env_client.observe()
    
    # Initialize conversation from environment starting point
    convo_start = env_client.conversation_start 
    conversation = [
        {"role": "user", "content": convo_start[0]["value"]},
        {"role": "assistant", "content": convo_start[1]["value"]},
        {"role": "user", "content": state},
    ]

    reward = 0.0
    done = False
    rounds = 0
    terminated_by = "max_rounds"
    
    while not done and rounds < max_rounds:
        try:
            generated = model.generate(conversation, sampling_params)
            generated = generated.strip()
        except Exception as exc:
            print(f"[item {item_idx}] Inference failed: {exc}")
            terminated_by = "model_error"
            break

        conversation.append({"role": "assistant", "content": generated})
        
        step = env_client.step(generated)
        reward, done = step.reward, step.done

        # 与训练 rollout 对齐：vllm_rollout.py 第 0 轮用 observe()，之后每轮用
        # step_output.state。SciWorld 里两者通常恒等（step() 成功时把
        # self.info["observation"] 设成同一个串），唯一分叉是动作解析失败时
        # step.state = "Invalid Action.\n\n" + 上一次观察，而 observe() 丢掉了
        # 这个前缀 —— 等于评测比训练少给模型一条纠错反馈。
        # --obs-source observe 可复现 2026-09-19 之前的旧口径。
        nxt = env_client.observe() if obs_source == "observe" else step.state
        conversation.append({"role": "user", "content": nxt})
        
        rounds += 1
        if done:
            terminated_by = "env_done"

    return {
        "item_id": f"sciworld_{item_idx}",
        # ``reward`` is SciWorld's raw score (the client returns ``score``): 0-100, and
        # -100 when the episode is lost. Success matches the training reward: the
        # episode ended with the full score.
        "reward": float(reward),
        "done": bool(done),
        "success": _is_success(float(reward), bool(done)),
        "rounds": rounds,
        "terminated_by": terminated_by,
        "conversations": conversation,
        "task_description": reset_info.get("task_description", ""),
    }


def run_trajectory_tokenio(env_client, model, sampling_params, item_idx: int, max_rounds: int,
                           obs_source: str = "step") -> dict:
    """thinking 模型（Qwen3）用：prompt 逐 token 复现训练 rollout，见 token_io.run_episode。"""
    reset_info = env_client.reset(item_idx)
    first = env_client.observe()
    cs = env_client.conversation_start
    def step_fn(content):
        st = env_client.step(content)
        return (env_client.observe() if obs_source == "observe" else st.state), st.reward, st.done
    ep = token_io.run_episode(model.tokenizer, lambda ids: model.generate_ids(ids, sampling_params),
                              step_fn, cs[0]["value"], cs[1]["value"], first, max_rounds)
    return {"item_id": f"sciworld_{item_idx}", "reward": ep["reward"], "done": ep["done"],
            "success": _is_success(ep["reward"], ep["done"]), "rounds": ep["rounds"],
            "terminated_by": ep["terminated_by"], "conversations": ep["conversation"],
            "task_description": reset_info.get("task_description", ""), "prompt_mode": "token_io"}

# ---------------------------------------------------------------------------
class EnvPool:
    """Thread-local SciworldEnvClients round-robining across servers."""
    def __init__(self, env_addrs: list[str], timeout: int):
        self._addrs = env_addrs
        self._timeout = timeout
        self._local = threading.local()
        self._counter = 0
        self._lock = threading.Lock()

    def get(self) -> SciworldEnvClient:
        client = getattr(self._local, "client", None)
        if client is None:
            with self._lock:
                addr = self._addrs[self._counter % len(self._addrs)]
                self._counter += 1
            client = SciworldEnvClient(env_server_base=addr, data_len=1, timeout=self._timeout)
            self._local.client = client
        return client

# ---------------------------------------------------------------------------
def load_test_ids(test_file: Path) -> list[int]:
    if not test_file.exists():
        print(f"ERROR: test id file {test_file} not found. Pass --test-file.")
        sys.exit(1)
    with test_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    
    ids = []
    for r in rows:
        if "official_goal_idx" in r:
            ids.append(int(r["official_goal_idx"]))
        else:
            ids.append(int(r["item_id"].split("_")[-1]))
    return ids

def aggregate(results: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    all_recs = list(results.values())
    if all_recs:
        all_succ = sum(_record_success(r) for r in all_recs) / len(all_recs)
        # raw mean (lost episodes count -100) and the mean with losses floored at 0
        all_score = sum(r["reward"] for r in all_recs) / len(all_recs)
        all_score_clip0 = sum(max(0.0, r["reward"]) for r in all_recs) / len(all_recs)
    else:
        all_succ = all_score = all_score_clip0 = float("nan")

    summary = {
        "All": {"success": all_succ, "score": all_score, "score_clip0": all_score_clip0,
                "count": len(all_recs)}
    }
    return summary

def format_report(summary: dict[str, dict[str, float]]) -> str:
    cols = ["All"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    succ_row = "| " + " | ".join(f"{summary[c]['success']*100:.2f}" if summary[c]['count'] else "-" for c in cols) + " |"
    score_row = "| " + " | ".join(f"{summary[c]['score']:.4f}" if summary[c]['count'] else "-" for c in cols) + " |"
    clip_row = "| " + " | ".join(f"{summary[c]['score_clip0']:.4f}" if summary[c]['count'] else "-" for c in cols) + " |"
    return (f"SciWorld Evaluation Results (success = done and score >= 100):\n{header}\n{sep}\n"
            f"Success Rate (%): {succ_row}\nAverage Score (raw, -100 on loss): {score_row}\n"
            f"Average Score (losses floored at 0): {clip_row}")

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SciWorld with local vLLM.")
    parser.add_argument("--model-path", required=True, help="Path to local checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-addrs", default="http://127.0.0.1:36101", help="Comma-separated env URLs")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-util", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--obs-source", default="step", choices=["step", "observe"],
                        help="每轮(非第0轮)喂给模型的观察来源。step=step_output.state，"
                             "与训练 rollout 一致（默认）；observe=每轮重新 observe()，"
                             "2026-09-19 之前的旧口径，会丢掉 'Invalid Action.' 前缀。")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help=f"JSON list of test item ids (default: {DEFAULT_TEST_FILE})",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_addrs = [s.strip() for s in args.env_addrs.split(",") if s.strip()]
    
    test_ids = load_test_ids(args.test_file)
    print(f"[data] test ids: {len(test_ids)} from {args.test_file}")
    if args.limit > 0:
        test_ids = test_ids[:args.limit]

    # Resolve absolute path to model
    abs_model_path = str(Path(args.model_path).resolve())
    model = LocalModel(abs_model_path, tp=args.tp, gpu_util=args.gpu_util)
    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=args.max_tokens)
    pool = EnvPool(env_addrs, timeout=DEFAULT_TIMEOUT)

    results = {}
    pending = []
    for idx in test_ids:
        out_path = args.output_dir / f"sciworld_{idx}.json"
        if out_path.exists() and not args.overwrite:
            with out_path.open("r") as f:
                results[idx] = json.load(f)
        else:
            pending.append(idx)

    print(f"Total: {len(test_ids)}, Cached: {len(results)}, Todo: {len(pending)}")

    def worker(idx):
        try:
            env = pool.get()
            if token_io.uses_token_io(model.tokenizer):   # Qwen3 等 thinking 模型
                payload = run_trajectory_tokenio(env, model, sampling_params, idx, args.max_rounds,
                                                 obs_source=args.obs_source)
            else:                                          # Qwen2.5 等：原路径不变
                payload = run_trajectory(env, model, sampling_params, idx, args.max_rounds,
                                     obs_source=args.obs_source)
            out_path = args.output_dir / f"sciworld_{idx}.json"
            with out_path.open("w") as f:
                json.dump(payload, f, indent=2)
            return idx, payload, None
        except Exception as e:
            return idx, None, f"{e}\n{traceback.format_exc()}"

    start_time = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(worker, idx) for idx in pending]
            for i, fut in enumerate(as_completed(futures), 1):
                idx, payload, err = fut.result()
                if err:
                    print(f"[{i}/{len(pending)}] Item {idx} FAILED: {err}")
                else:
                    results[idx] = payload
                    print(f"[{i}/{len(pending)}] Item {idx} reward={payload['reward']} rounds={payload['rounds']}")

    elapsed = time.time() - start_time
    summary = aggregate(results)
    print(f"\n==== EVAL COMPLETE ({elapsed:.1f}s) ====")
    print(format_report(summary))
    
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump({"summary": summary, "elapsed": elapsed, "model": args.model_path,
                   "setting": {"max_rounds": args.max_rounds, "max_tokens": args.max_tokens,
                               "temperature": args.temp, "top_p": args.top_p,
                               "obs_source": args.obs_source}}, f, indent=2)

if __name__ == "__main__":
    main()
