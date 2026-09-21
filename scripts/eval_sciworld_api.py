#!/usr/bin/env python3
"""SciWorld 评测（通过 OpenRouter 调用远端模型，如 Gemini 2.5 Flash-Lite）。

与 eval_sciworld.py 共用 EnvPool / 指标口径 / 测试集，只把本地 vLLM 换成 HTTP API。
纯 CPU，不占 GPU，可与训练并跑。

完整性优先的设计（这是本脚本的首要目标）：
  1. 断点续跑：每题结果单独落盘 sciworld_<id>.json，重启自动跳过已完成的；
     只有 --overwrite 才重来。中途挂掉不丢已完成的题。
  2. 分级重试：429/5xx/超时/连接错误 -> 指数退避重试，最多 API_RETRY 次；
     单轮彻底失败才放弃该轮，并在记录里标 terminated_by=api_error。
  3. 空回复防护：Gemini 偶尔返回空串或只有 finish_reason=length。空回复按
     "无效动作"送进环境（环境会给 invalid 提示），而不是直接中断整条轨迹 ——
     否则一次抖动就废掉一题。
  4. 全局限流：令牌桶 + 并发上限，避免打爆 rate limit 反而拖慢整体。
  5. 成本核算：累计 usage.cost（OpenRouter 直接返回真实计费），跑完给总账。
  6. 收尾校验：跑完检查 len(results)==len(ids)，缺哪题明确列出来，绝不静默少跑。
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_sciworld import (  # noqa: E402
    EnvPool, load_test_ids, aggregate, format_report, _is_success,
    DEFAULT_TEST_FILE, DEFAULT_MAX_ROUNDS, DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TIMEOUT,
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_RETRY = 6
RETRY_BASE = 2.0


class RateLimiter:
    """简单令牌桶：每秒最多 rps 次调用。"""
    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            time.sleep(wait)


class APIModel:
    """OpenRouter 客户端，带重试与用量统计。"""
    def __init__(self, model: str, key: str, limiter: RateLimiter, timeout: int = 180):
        self.model, self.key, self.limiter, self.timeout = model, key, limiter, timeout
        self.sess = requests.Session()
        self._lock = threading.Lock()
        self.tok_in = self.tok_out = 0
        self.cost = 0.0
        self.n_calls = self.n_retry = self.n_fail = self.n_empty = 0
        self.n_empty_retry = 0       # 空 content 触发的重试次数
        self.tok_reason = 0          # 隐藏思考 token，用来核对预算够不够
        self.n_truncated = 0         # finish_reason == "length"
        self.reasoning_max = 0
        self.reasoning_effort = ""

    def generate(self, messages, max_tokens, temperature, top_p):
        # reasoning 模型的隐藏思考 token 也算在 max_tokens 里。若按可见预算
        # (200，与训练 rollout 同口径) 设置，模型会把预算全花在思考上、返回 ""，
        # 环境判为非法动作 —— 一个配置错误看起来像 baseline 很差。
        # 所以把两个预算拆开：可见预算保持 max_tokens，额外给 reasoning_max
        # 的思考余量，总上限 = max_tokens + reasoning_max。
        # 这样 Gemini 与本地模型的**可见输出**预算仍然一致，可比。
        total = max_tokens + (self.reasoning_max if self.reasoning_max > 0 else 0)
        body = {"model": self.model, "messages": messages,
                "max_tokens": total, "temperature": temperature, "top_p": top_p}
        if self.reasoning_max > 0:
            body["reasoning"] = {"max_tokens": self.reasoning_max}
        elif self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        hdr = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        last = ""
        for attempt in range(API_RETRY):
            self.limiter.acquire()
            try:
                r = self.sess.post(API_URL, headers=hdr, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    d = r.json()
                    if "error" in d:
                        last = f"api-error: {d['error']}"
                    else:
                        ch = (d.get("choices") or [{}])[0]
                        txt = (ch.get("message") or {}).get("content") or ""
                        fin = ch.get("finish_reason")
                        # 用量统计必须在重试判断**之前**：被重试掉的那次调用一样
                        # 产生了 token 和费用，跳过就会少算成本。
                        u = d.get("usage") or {}
                        det = (u.get("completion_tokens_details") or {})
                        with self._lock:
                            self.n_calls += 1
                            self.tok_in += u.get("prompt_tokens", 0)
                            self.tok_out += u.get("completion_tokens", 0)
                            self.tok_reason += int(det.get("reasoning_tokens", 0) or 0)
                            self.cost += float(u.get("cost", 0.0) or 0.0)
                            if fin == "length":
                                self.n_truncated += 1
                        # 空 content 属于瞬时故障，重试而不是直接返回空串
                        # （返回空串会被环境算成一次非法动作）。截断导致的空
                        # 内容也重试 —— 温度 1.0 下换一次采样往往就不再超预算。
                        if not txt.strip() and attempt < API_RETRY - 1:
                            last = f"empty-content(finish={fin})"
                            with self._lock:
                                self.n_empty_retry += 1
                            time.sleep(RETRY_BASE ** attempt * (1 + random.random() * 0.3))
                            continue
                        if not txt.strip():
                            with self._lock:
                                self.n_empty += 1
                        return txt
                elif r.status_code in (429, 500, 502, 503, 504, 520, 524):
                    last = f"http-{r.status_code}"
                else:
                    last = f"http-{r.status_code}: {r.text[:160]}"
                    if r.status_code in (400, 401, 403, 404):
                        break            # 不可重试的错误，直接放弃
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
            with self._lock:
                self.n_retry += 1
            time.sleep(RETRY_BASE ** attempt * (1 + random.random() * 0.3))
        with self._lock:
            self.n_fail += 1
        print(f"    [api] 放弃: {last}", flush=True)
        return ""


def run_trajectory(env_client, model: APIModel, item_idx: int, max_rounds: int,
                   max_tokens: int, temp: float, top_p: float,
                   obs_source: str = "step") -> dict[str, Any]:
    reset_info = env_client.reset(item_idx)
    state = env_client.observe()
    cs = env_client.conversation_start
    conv = [{"role": "user", "content": cs[0]["value"]},
            {"role": "assistant", "content": cs[1]["value"]},
            {"role": "user", "content": state}]
    reward, done, rounds, term, n_empty = 0.0, False, 0, "max_rounds", 0
    n_empty_run = 0              # 连续空回复计数，收到正常回复就清零
    while not done and rounds < max_rounds:
        gen = model.generate(conv, max_tokens, temp, top_p).strip()
        if not gen:
            # 空回复不中断整条轨迹：送一个显式无效动作让环境给出提示，
            # **连续** 3 次才判定为 api 故障退出。
            # 2026-09-19: 原先 n_empty 从不重置，是累计计数 —— 与注释不符，
            # 30 轮里零散 3 次空回复就会把一条正常推进的轨迹判死。
            n_empty += 1
            n_empty_run += 1
            gen = "Thought:\n(no response)\n\nAction:\nlook around"
            if n_empty_run >= 3:
                term = "api_error"
                break
        else:
            n_empty_run = 0
        conv.append({"role": "assistant", "content": gen})
        step = env_client.step(gen)
        reward, done = step.reward, step.done
        # 与训练 rollout 对齐：第 0 轮 observe()，之后用 step_output.state。
        # 两者在 SciWorld 里通常恒等；分叉只在解析失败时 —— step.state 带
        # "Invalid Action." 前缀而 observe() 没有。见 --obs-source。
        nxt = env_client.observe() if obs_source == "observe" else step.state
        conv.append({"role": "user", "content": nxt})
        rounds += 1
        if done:
            term = "env_done"
    return {"item_id": f"sciworld_{item_idx}", "reward": float(reward), "done": bool(done),
            "success": _is_success(float(reward), bool(done)), "rounds": rounds,
            "terminated_by": term, "n_empty_replies": n_empty,
            "conversations": conv,
            "task_description": reset_info.get("task_description", "")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemini-2.5-flash-lite")
    p.add_argument("--key-file", default="/data1/secrets/openrouter.key")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--env-addrs", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--rps", type=float, default=8.0, help="全局每秒请求上限")
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--obs-source", default="step", choices=["step", "observe"],
                   help="每轮(非第0轮)观察来源。step=step_output.state，与训练 rollout "
                        "一致（默认）；observe=旧口径。")
    p.add_argument("--max-tokens-api", type=int, default=8192,
                   help="API 模型每轮的生成上限，覆盖 --max-tokens。默认 8192。"
                        "实测(Gemini2.5-flash-lite, 43k 字符上下文, 8 次采样)："
                        "不开思考时可见输出中位 1704、最大 4112 token —— "
                        "本地模型评测用的 200 对这个模型远远不够，会把 Action 截掉判非法。"
                        "cap 只在被撞到时才有成本，给宽是免费的。")
    p.add_argument("--reasoning-max-tokens", type=int, default=0,
                   help="隐藏思考预算，**额外**加在生成上限之上。默认 0=不发 reasoning 字段。"
                        "实测该模型默认就不思考；而且给多少花多少 —— "
                        "给 4096 就思考 4092 并把可见预算挤没，截断 7/8。")
    p.add_argument("--reasoning-effort", default="",
                   help="reasoning-max-tokens 为 0 时的 effort 档位；默认空=不发。"
                        "注意实测 effort=low 的思考量随 cap 按比例伸缩(约 cap/5)，"
                        "不是绝对预算 —— 调大 cap 避截断会同时推高思考量和成本。")
    a = p.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)
    key = Path(a.key_file).read_text().strip()
    ids = load_test_ids(a.test_file)
    if a.limit:
        ids = ids[: a.limit]

    # 断点续跑
    results: dict[int, dict] = {}
    if not a.overwrite:
        for i in ids:
            f = a.output_dir / f"sciworld_{i}.json"
            if f.exists() and f.stat().st_size > 0:
                try:
                    results[i] = json.loads(f.read_text())
                except Exception:
                    pass
    todo = [i for i in ids if i not in results]
    print(f"[api-eval] 模型={a.model}  总题数={len(ids)}  已完成={len(results)}  待跑={len(todo)}")
    print(f"[api-eval] 并发={a.concurrency} 限流={a.rps}/s max_rounds={a.max_rounds} temp={a.temp}")
    _tot = a.max_tokens_api + max(a.reasoning_max_tokens, 0)
    print(f"[api-eval] obs_source={a.obs_source}  生成上限={a.max_tokens_api} "
          f"+ 思考预算={a.reasoning_max_tokens} => max_tokens={_tot}")
    if a.max_tokens_api != a.max_tokens:
        print(f"[api-eval] 注意：本地模型评测用 max_tokens={a.max_tokens}，"
              f"本次 API 用 {a.max_tokens_api} —— 两者可见预算不同，比较时须知。")

    limiter = RateLimiter(a.rps)
    model = APIModel(a.model, key, limiter)
    model.reasoning_max = a.reasoning_max_tokens
    model.reasoning_effort = a.reasoning_effort
    pool = EnvPool([x.strip() for x in a.env_addrs.split(",") if x.strip()], timeout=DEFAULT_TIMEOUT)
    lock = threading.Lock()
    t0 = time.time()

    def work(i):
        r = run_trajectory(pool.get(), model, i, a.max_rounds, a.max_tokens_api, a.temp, a.top_p,
                           obs_source=a.obs_source)
        (a.output_dir / f"sciworld_{i}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        return i, r

    if todo:
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = {ex.submit(work, i): i for i in todo}
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    i, r = f.result()
                    with lock:
                        results[i] = r
                except Exception as e:
                    print(f"  [item {futs[f]}] 失败: {type(e).__name__}: {e}", flush=True)
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)}  用时{el/60:.1f}min  "
                          f"已花费${model.cost:.4f}  重试{model.n_retry} 放弃{model.n_fail} 空回复{model.n_empty}",
                          flush=True)

    # 收尾校验：绝不静默少跑
    missing = [i for i in ids if i not in results]
    print(f"\n==== 完整性校验 ====")
    print(f"  应完成 {len(ids)}，实际 {len(results)}，缺失 {len(missing)}")
    if missing:
        print(f"  !! 缺失题号: {missing[:30]}{' ...' if len(missing)>30 else ''}")
        print(f"  !! 重跑同一命令即可断点续跑（不要加 --overwrite）")
    n_api_err = sum(1 for r in results.values() if r.get("terminated_by") == "api_error")
    if n_api_err:
        print(f"  !! {n_api_err} 题因 API 连续空回复中断，建议删掉对应 json 后重跑")

    summary = aggregate(results)
    print(format_report(summary))
    s = summary["All"]
    print(f"\n[{a.model}]  Succ {s['success']*100:.2f}%   Score {s['score']:.4f}   "
          f"score_clip0 {s['score_clip0']:.4f}   n={s['count']}")
    print(f"[用量] 输入 {model.tok_in:,} tok  输出 {model.tok_out:,} tok  "
          f"(其中隐藏思考 {model.tok_reason:,})  调用 {model.n_calls:,} 次  "
          f"实际花费 ${model.cost:.4f}")
    print(f"[健康] 空回复 {model.n_empty}  空回复重试 {model.n_empty_retry}  "
          f"截断(finish_reason=length) {model.n_truncated}  "
          f"重试 {model.n_retry}  放弃 {model.n_fail}")
    if model.n_truncated and a.reasoning_max_tokens:
        print("  !! 仍有截断：思考预算可能不够，调大 --reasoning-max-tokens")
    (a.output_dir / "summary.json").write_text(json.dumps({
        "summary": summary, "model": a.model, "elapsed_min": (time.time()-t0)/60,
        "tokens_in": model.tok_in, "tokens_out": model.tok_out,
        "api_calls": model.n_calls, "cost_usd": model.cost,
        "n_retry": model.n_retry, "n_fail": model.n_fail, "n_empty": model.n_empty,
        "n_missing": len(missing), "missing_ids": missing,
        "tokens_reasoning": model.tok_reason, "n_truncated": model.n_truncated,
        "n_empty_retry": model.n_empty_retry,
        "setting": {"max_rounds": a.max_rounds, "max_tokens": a.max_tokens_api,
                    "max_tokens_local_eval_default": a.max_tokens,
                    "reasoning_max_tokens": a.reasoning_max_tokens,
                    "reasoning_effort": a.reasoning_effort,
                    "temperature": a.temp, "top_p": a.top_p,
                    "obs_source": a.obs_source},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
