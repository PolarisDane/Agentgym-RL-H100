#!/usr/bin/env python
"""AppWorld checkpoint 评测。

复刻训练时的 rollout 逻辑（verl agent_vllm_rollout），保证评测与训练同口径：
  - 直接 import agentenv 的 AppWorldEnvClient，system prompt / 代码抽取 / 环境协议
    全部与训练一致，不重新实现
  - 与训练一样按轮同步批量生成：每轮把所有还活着的轨迹的 prompt 一起喂给 vLLM
  - 每条轨迹独占一个 env server 进程（AppWorld 的 supervisor 活跃任务是进程级全局
    状态，同进程多 episode 会互相判 done —— 见 appworld_concurrency_audit.py A 项）

用法:
  python eval_appworld.py --model-path <hf_dir> --split test_normal \
      --env-addrs "http://127.0.0.1:36301,..." --output-dir runs/eval/step50_normal
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/data1/repos/Agentgym-RL/AgentGym/agentenv")

DEFAULT_MAX_ROUNDS = 30
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.4   # G2PO 的 validation 温度
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_MODEL_LEN = 32768   # Qwen2.5-14B 的 max_position_embeddings 上限；训练时 verl 强制覆盖到 34816，评测不越界


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--split", required=True,
                   choices=["train", "dev", "test_normal", "test_challenge"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--env-addrs", required=True,
                   help="逗号分隔的 env server 地址；数量必须 >= 并发数")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-util", type=float, default=0.85)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"{summary_path} 已存在，用 --overwrite 覆盖")
        return 0

    from agentenv.envs.appworld import AppWorldEnvClient, _APPWORLD_SYSTEM
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    addrs = [a.strip() for a in args.env_addrs.split(",") if a.strip()]
    n_tasks = _split_size(args.split)
    if args.limit:
        n_tasks = min(n_tasks, args.limit)
    conc = min(args.concurrency, len(addrs), n_tasks)
    assert conc >= 1, "并发数为 0"
    print(f"[eval] split={args.split} 任务数={n_tasks} 并发={conc} "
          f"(env server {len(addrs)} 个, 每条轨迹独占 1 个)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              dtype="bfloat16", enforce_eager=True, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens)

    results = []
    t0 = time.time()
    # 分批：每批 conc 条，批内按轮同步生成（与训练 rollout 一致）
    for base in range(0, n_tasks, conc):
        ids = list(range(base, min(base + conc, n_tasks)))
        results += _run_batch(ids, addrs, llm, tok, sp, args,
                              AppWorldEnvClient, _APPWORLD_SYSTEM)
        done = len(results)
        succ = sum(1 for r in results if r["reward"] > 0)
        el = time.time() - t0
        print(f"[eval] {done}/{n_tasks}  成功 {succ} ({succ/done:.1%})  "
              f"用时 {el/60:.1f} 分  预计剩余 {(el/done)*(n_tasks-done)/60:.1f} 分",
              flush=True)

    succ = sum(1 for r in results if r["reward"] > 0)
    summary = {
        "model_path": str(args.model_path),
        "split": args.split,
        "num_tasks": len(results),
        "num_success": succ,
        "success_rate": succ / len(results) if results else 0.0,
        "temperature": args.temp,
        "max_rounds": args.max_rounds,
        "max_tokens_per_turn": args.max_tokens,
        "mean_rounds": sum(r["rounds"] for r in results) / len(results) if results else 0,
        "hit_round_cap": sum(1 for r in results if r["rounds"] >= args.max_rounds),
        "env_errors": sum(1 for r in results if r.get("error")),
        "elapsed_min": (time.time() - t0) / 60,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.output_dir / "trajectories.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("[eval] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _split_size(split):
    """从 AppWorld 数据集读该 split 的任务数（不依赖 eval json，避免口径不一致）。"""
    import subprocess
    code = (
        "import os;os.environ.setdefault('APPWORLD_ROOT','/data1/datasets/appworld');"
        "from appworld import load_task_ids;print(len(load_task_ids('%s')))" % split
    )
    out = subprocess.run(
        ["/usr/local/miniconda3/envs/agentenv-appworld/bin/python3", "-c", code],
        capture_output=True, text=True, cwd="/data1/datasets/appworld")
    return int(out.stdout.strip().splitlines()[-1])


def _run_batch(ids, addrs, llm, tok, sp, args, ClientCls, system_prompt):
    """一批轨迹按轮同步推进，复刻 verl 的 rollout 循环。"""
    n = len(ids)
    clients, convs, done, reward, err = [], [], [], [], []
    for k, tid in enumerate(ids):
        c = ClientCls(env_server_base=addrs[k % len(addrs)], data_len=1, timeout=600)
        clients.append(c)
        try:
            instr = c.reset(tid)
        except Exception as e:
            instr = ""
            err.append(str(e)[:200])
        convs.append([
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": instr},
        ])
        done.append(False)
        reward.append(0.0)
    err += [None] * (n - len(err))

    rounds_used = [0] * n
    for _ in range(args.max_rounds):
        active = [i for i in range(n) if not done[i]]
        if not active:
            break
        prompts = [tok.apply_chat_template(convs[i], tokenize=False,
                                           add_generation_prompt=True) for i in active]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]

        def one(j):
            i = active[j]
            convs[i].append({"role": "assistant", "content": texts[j]})
            rounds_used[i] += 1
            try:
                so = clients[i].step(texts[j])
                convs[i].append({"role": "user", "content": so.state})
                reward[i] = so.reward
                return so.done
            except Exception as e:
                err[i] = str(e)[:200]
                return True

        with ThreadPoolExecutor(max_workers=len(active)) as ex:
            flags = list(ex.map(one, range(len(active))))
        for j, f in enumerate(flags):
            if f:
                done[active[j]] = True

    for c in clients:
        try:
            c.close()
        except Exception:
            pass

    return [{"task_index": ids[i], "reward": reward[i], "rounds": rounds_used[i],
             "error": err[i], "conversations": convs[i]} for i in range(n)]


if __name__ == "__main__":
    sys.exit(main())
