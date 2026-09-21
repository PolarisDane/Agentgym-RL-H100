#!/usr/bin/env python3
"""SciWorld + Reflexion 评测。

与 eval_sciworld.py 共用 LocalModel / EnvPool / 指标口径，只把"单条轨迹"换成
"最多 N 次 trial + 累积反思"：

    for trial in 1..N:
        env.reset(item) -> 完整跑一条轨迹（同样 max_rounds 上限）
        满分则停
        否则让**同一个模型**基于 (任务, 本次轨迹, 得分, 终止原因) 写一段反思
        反思累积进 memory（上限 MEM_MAX 条），注入下一 trial 的 system prompt

指标（与 eval_sciworld.py 完全同口径，便于直接对比）：
    Succ  = 满分率，done and reward >= 100.0
    Score = 平均 reward，范围 -100..100
    主报告用**最后一次 trial** 的结果；另附 best（历史最好）作参考。

设计说明：SciWorld 的 reward 是连续的 -100..100 且中途随子目标增长，不是
Reflexion 原论文的二元成败信号。所以反思 prompt 里明确告诉模型拿了多少分、
满分多少、以及终止原因（env_done / max_rounds / 负分），否则它无法区分
"差一点"和"完全跑偏"。
"""
from __future__ import annotations
import argparse, json, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sciworld import (  # noqa: E402
    LocalModel, EnvPool, load_test_ids, aggregate, format_report,
    _is_success, _record_success, DEFAULT_TEST_FILE, DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TIMEOUT,
)
from vllm import SamplingParams  # noqa: E402

MEM_MAX = 3
REFLECT_MAX_TOKENS = 256

REFLECT_TMPL = """You just attempted the following task in ScienceWorld and did not solve it.

Task: {task}

Your attempt (each round shows your action and the resulting observation):
{trace}

Outcome: you scored {score:.1f} out of 100. {why}

Write a short reflection (at most 4 sentences) diagnosing what went wrong and
what you will do differently next time. Be concrete: name the specific actions
or objects involved. Do not restate the task. Do not write an action."""

WHY = {
    "env_done": "The episode ended but without the full score.",
    "max_rounds": "You ran out of rounds ({r}) before finishing.",
    "model_error": "Generation failed partway through.",
    "negative": "You took an action that lost the episode (score -100).",
}

MEM_HDR = ("\n\nYou have attempted this task before and failed. "
           "Here are your own reflections from those attempts — use them:\n")


def _trace(conversation: list[dict], max_rounds_shown: int = 30) -> str:
    """把会话压成 '第N轮 动作 -> 观察' 的紧凑文本（跳过开场的 3 条）。"""
    out, n = [], 0
    for i in range(3, len(conversation) - 1, 2):
        if conversation[i].get("role") != "assistant":
            continue
        n += 1
        act = (conversation[i].get("content") or "").strip().replace("\n", " ")
        obs = (conversation[i + 1].get("content") or "").strip().replace("\n", " ")
        out.append(f"  {n}. action: {act[:160]}\n     obs: {obs[:200]}")
        if n >= max_rounds_shown:
            break
    return "\n".join(out) if out else "  (no actions taken)"


def run_one_trial(env_client, model, sp, item_idx, max_rounds, memory):
    """跑一条完整轨迹；memory 非空时把累积反思追加到 system prompt。"""
    reset_info = env_client.reset(item_idx)
    state = env_client.observe()
    convo_start = env_client.conversation_start
    sys_prompt = convo_start[0]["value"]
    if memory:
        sys_prompt += MEM_HDR + "\n".join(f"- {m}" for m in memory)
    conversation = [
        {"role": "user", "content": sys_prompt},
        {"role": "assistant", "content": convo_start[1]["value"]},
        {"role": "user", "content": state},
    ]
    reward, done, rounds, term = 0.0, False, 0, "max_rounds"
    while not done and rounds < max_rounds:
        try:
            generated = model.generate(conversation, sp).strip()
        except Exception as exc:
            print(f"[item {item_idx}] inference failed: {exc}")
            term = "model_error"
            break
        conversation.append({"role": "assistant", "content": generated})
        step = env_client.step(generated)
        reward, done = step.reward, step.done
        conversation.append({"role": "user", "content": env_client.observe()})
        rounds += 1
        if done:
            term = "env_done"
    if reward <= -100.0:
        term = "negative"
    return {
        "reward": float(reward), "done": bool(done),
        "success": _is_success(float(reward), bool(done)),
        "rounds": rounds, "terminated_by": term,
        "conversations": conversation,
        "task_description": reset_info.get("task_description", ""),
    }


def run_reflexion(env_client, model, sp, reflect_sp, item_idx, max_rounds, n_trials):
    memory: list[str] = []
    trials: list[dict[str, Any]] = []
    for t in range(n_trials):
        tr = run_one_trial(env_client, model, sp, item_idx, max_rounds, memory)
        tr["trial"] = t + 1
        tr["memory_used"] = list(memory)
        trials.append(tr)
        if tr["success"]:
            break
        if t == n_trials - 1:
            break                      # 最后一次不必再反思
        why = WHY.get(tr["terminated_by"], "").format(r=tr["rounds"])
        prompt = REFLECT_TMPL.format(
            task=tr["task_description"] or "(unknown)",
            trace=_trace(tr["conversations"]),
            score=tr["reward"], why=why)
        try:
            refl = model.generate([{"role": "user", "content": prompt}], reflect_sp).strip()
        except Exception as exc:
            print(f"[item {item_idx}] reflection failed: {exc}")
            refl = ""
        if refl:
            memory.append(refl)
            memory[:] = memory[-MEM_MAX:]
        tr["reflection"] = refl
    last, best = trials[-1], max(trials, key=lambda x: x["reward"])
    return {
        "item_id": f"sciworld_{item_idx}",
        # 主指标 = 最后一次 trial（与 eval_sciworld.py 的字段名一致，便于共用 aggregate）
        "reward": last["reward"], "done": last["done"],
        "success": last["success"], "rounds": last["rounds"],
        "terminated_by": last["terminated_by"],
        "task_description": last["task_description"],
        "conversations": last["conversations"],
        # Reflexion 特有
        "n_trials": len(trials),
        "best_reward": best["reward"], "best_success": best["success"],
        "trial_rewards": [t["reward"] for t in trials],
        "reflections": [t.get("reflection", "") for t in trials[:-1]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--env-addrs", default="http://127.0.0.1:36101")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-util", type=float, default=0.8)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = a.output_dir / "results.json"
    if out_json.exists() and not a.overwrite:
        print(f"{out_json} 已存在，用 --overwrite 覆盖"); sys.exit(0)

    ids = load_test_ids(a.test_file)
    if a.limit: ids = ids[: a.limit]
    print(f"[reflexion] {len(ids)} 题 × 最多 {a.n_trials} trial, max_rounds={a.max_rounds}")

    model = LocalModel(a.model_path, tp=a.tp, gpu_util=a.gpu_util)
    sp = SamplingParams(temperature=a.temp, top_p=a.top_p, max_tokens=a.max_tokens)
    reflect_sp = SamplingParams(temperature=a.temp, top_p=a.top_p, max_tokens=REFLECT_MAX_TOKENS)
    pool = EnvPool([x.strip() for x in a.env_addrs.split(",") if x.strip()], timeout=DEFAULT_TIMEOUT)

    results: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()
    def work(i):
        return i, run_reflexion(pool.get(), model, sp, reflect_sp, i, a.max_rounds, a.n_trials)
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(work, i): i for i in ids}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                i, r = f.result()
                with lock: results[i] = r
            except Exception as exc:
                print(f"[item {futs[f]}] failed: {exc}")
            if k % 20 == 0: print(f"  {k}/{len(ids)}")

    summary = aggregate(results)
    recs = list(results.values())
    if recs:
        summary["All"]["best_success"] = sum(r["best_success"] for r in recs) / len(recs)
        summary["All"]["best_score"] = sum(r["best_reward"] for r in recs) / len(recs)
        summary["All"]["mean_trials"] = sum(r["n_trials"] for r in recs) / len(recs)
    out_json.write_text(json.dumps({"summary": summary, "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(summary))
    s = summary["All"]
    print(f"\n[Reflexion] last-trial  Succ {s['success']*100:.2f}%  Score {s['score']:.4f}")
    print(f"[Reflexion] best-of-{a.n_trials}   Succ {s.get('best_success',0)*100:.2f}%  "
          f"Score {s.get('best_score',0):.4f}   平均 trial 数 {s.get('mean_trials',0):.2f}")

if __name__ == "__main__":
    main()
