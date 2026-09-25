#!/usr/bin/env python3
"""离线计算 checkpoint 的 TE（时序自洽）指标 —— 不需要重新 rollout。

用评测时保存的完整轨迹 + 一次前向，复现训练中观测到的 te/gain_tok 与 te/kl。

两个口径（都算，对比本身有信息）：
  own   每个 ckpt 用**自己产生**的轨迹 —— 与训练中的 te/kl 同口径，但混淆了
        "模型更自洽" 和 "轨迹本身变规整"（好模型的轨迹更短、无效动作更少）
  fixed 所有 ckpt 用**同一批**轨迹（默认 base model 的）—— 控制住轨迹分布，
        差异纯粹来自模型
  若 own 降幅 >> fixed 降幅，说明 TE 下降主要来自轨迹变简单而非模型变自洽。

指标（与训练中 te/* 完全同定义）：
  gain_tok = (log qF - log π⁰) / n_tok，qF = 成员(k>=1)预测分布的平均
  te_kl    = KL(π ‖ q)，q = [(1-η)p0 + η qF]/Z，全词表逐 token
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AgentGym-RL"))
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from verl.agent_trainer.ppo.temporal_ensemble import (  # noqa: E402
    build_te_scoring_samples, slot_logprobs_from_logits,
    slot_topk_from_logits, fullvocab_te_kl_rows, TE_TOPM,
)
from verl.workers.rollout.schemas import _action_token_mask  # noqa: E402


def load_af_lora(model, path: str):
    """交叉 TE 用：把 AF-LoRA 适配器挂到模型上（默认不启用，只有 enabled() 块里生效）。

    训练时 LoRA 建在 FSDP 包装后的模块上，保存的 key 带 _fsdp_wrapped_module 前缀；
    这里挂到普通 HF 模型上，需要把前缀去掉后再对齐。
    """
    import importlib.util as _ilu
    from pathlib import Path as _P
    _p = _P(__file__).resolve().parent.parent / "AgentGym-RL" / "verl" / "agent_trainer" / "ppo" / "af_lora.py"
    _sp = _ilu.spec_from_file_location("_af_lora", _p)
    _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
    d = torch.load(path, map_location="cpu", weights_only=False)
    lora = _m.AFLoRA(model, rank=int(d["rank"]), alpha=float(d["alpha"]),
                     targets=tuple(d["targets"]), dtype=torch.float32,
                     device=next(model.parameters()).device)
    def clean(k):
        return k.replace("_fsdp_wrapped_module__", "").replace("__fsdp_wrapped_module__", "")
    sd = {clean(k): v.float() for k, v in d["lora"].items()}
    missing = [k for k in lora.state_dict() if k not in sd]
    if missing:
        raise KeyError(f"af_lora key mismatch, e.g. {missing[:2]} not in {list(sd)[:2]}")
    lora.load_state_dict(sd)
    lora.attach()
    print(f"[te_offline] af_lora loaded from {path}: {len(lora.A)} adapters, step {d.get('global_step')}")
    return lora


def load_trajs(d: str, limit: int = 0):
    out = []
    import re as _re
    _fs = glob.glob(os.path.join(d, "sciworld_*.json"))
    # 按 item_id 数值排序。原先是字典序 —— sciworld_1002 排在 sciworld_536 之前，
    # 取前 N 条会落进一段连续 id 区间（实测 25 条全是"测熔点"一类任务）。
    _fs.sort(key=lambda x: int(_re.search(r"sciworld_(\d+)", x).group(1)))
    for f in _fs[: limit or None]:
        try:
            j = json.load(open(f))
            cv = j.get("conversations") or []
            if len(cv) >= 5:
                out.append(cv)
        except Exception:
            pass
    return out


@torch.no_grad()
def te_for(model, tok, trajs, k, eta, device, max_traj=0, skip_invalid=True, af_lora=None):
    """返回 (gain_tok_mean, te_kl_mean, n_targets, n_pos)."""
    gains, npos = [], 0
    kls = {}
    qfs, p0s = [], []          # 分解用：成员侧 log qF / rollout 侧 log π⁰（都已按 ntok 归一）
    per_traj = []              # 逐轨迹明细：bootstrap 置信区间 + 按来源族拆分都要用它
    _n0 = 0
    for ti, cv in enumerate(trajs[: max_traj or None]):
        _n0 = len(gains)
        try:
            samples = build_te_scoring_samples(cv, tok, k=k, skip_invalid=skip_invalid,
                                               env="sciworld")
        except Exception:
            continue
        if not samples:
            continue
        # rollout 侧：每个动作轮的裸动作 token 与其 log-prob
        assistants = [m for m in cv if (m.get("role") or m.get("from")) in ("assistant", "gpt")]
        if len(assistants) < 2:
            continue
        roll = {}
        for ai, m in enumerate(assistants[1:]):          # 跳过开场 ack
            c = m.get("content") or m.get("value") or ""
            ids = tok(c, add_special_tokens=False)["input_ids"]
            mk = _action_token_mask(tok, c, ids, "sciworld")
            roll[ai] = [x for x, keep in zip(ids, mk) if keep]
        # 用整条会话的一次前向拿 rollout 侧 log-prob。
        # ⚠️ 不能用 "在全序列里搜动作 token 串" 定位 —— 退化轨迹会把同一动作
        # (go to kitchen 等) 复读几十次，取第一个匹配会落在模型已复读多次之后的
        # 位置，条件概率接近 1（实测 pi0 sum = -0.001），gain 因此被算成正数。
        # 正确做法：按"第 t 个 assistant 消息"的字符偏移定位，再映射到 token。
        full = tok.apply_chat_template(cv, tokenize=False, add_generation_prompt=False)
        enc = tok(full, add_special_tokens=False, return_tensors="pt",
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k2: v2.to(device) for k2, v2 in enc.items()}
        # 每个 assistant 消息在 full 里的字符区间（按出现顺序 rindex 往后扫）
        msg_spans = {}
        _cur = 0
        _ai = -1
        for m in cv:
            c = m.get("content") or m.get("value") or ""
            if not c:
                continue
            try:
                st = full.index(c, _cur)
            except ValueError:
                continue
            _cur = st + len(c)
            if (m.get("role") or m.get("from")) in ("assistant", "gpt"):
                _ai += 1
                if _ai >= 1:                 # 跳过开场 ack，与 roll 的编号对齐
                    msg_spans[_ai - 1] = (st, st + len(c))
        if enc["input_ids"].shape[1] > 8192:
            continue
        lg = model(**enc).logits[0]
        lp_all = torch.log_softmax(lg[:-1].float(), -1)
        tokids = enc["input_ids"][0, 1:]
        gathered = lp_all.gather(1, tokids.unsqueeze(-1)).squeeze(-1)
        # 逐样本打分（成员侧）
        mem_topm = {}      # target_turn -> [(ids[T,M], prs[T,M]), ...] 每个成员一项
        pos_of = {}        # target_turn -> (rollout 动作在 tokids 里的起始下标, ntok)
        for s in samples:
            ids = s["input_ids"].unsqueeze(0).to(device)
            am = s["attention_mask"].unsqueeze(0).to(device)
            if af_lora is not None:
                with af_lora.enabled():
                    out = model(input_ids=ids, attention_mask=am).logits
            else:
                out = model(input_ids=ids, attention_mask=am).logits
            slp = slot_logprobs_from_logits(out, ids, [s["slot_spans"]])[0]
            tk_ids, tk_prs = slot_topk_from_logits(out, ids, [s["slot_spans"]])
            src = s["src_turn"]
            for j, (t, v) in enumerate(zip(s["slot_turns"], slp)):
                if t <= src or v != v:
                    continue
                ro = roll.get(int(t))
                if not ro:
                    continue
                # rollout 侧同一动作的 log-prob 和：在 full 编码里定位该动作 token
                # 简化：用成员 span 的 token 数做归一，π⁰ 用同一前向的 gathered 近似
                ntok = len(ro)
                if ntok == 0:
                    continue
                # 在"第 t 个 assistant 消息"的字符区间内定位动作 token
                sp = msg_spans.get(int(t))
                if sp is None:
                    continue
                cand = [i for i, (a, b) in enumerate(offs)
                        if a >= sp[0] and b <= sp[1]]
                if not cand:
                    continue
                seq = tokids.tolist()
                lo, hi = cand[0], cand[-1] + 1
                pos = None
                for st in range(max(lo - 1, 0), max(hi - ntok, lo)):
                    if seq[st:st + ntok] == ro:
                        pos = st
                        break
                if pos is None:
                    continue
                p0 = float(gathered[pos:pos + ntok].sum())
                # 成员 span 与 rollout 动作的 token 数必须一致，否则是不可比的两个量
                # （训练侧 assemble_te_tensors_token 有同样的丢弃逻辑）
                if s["slot_spans"][j][1] - s["slot_spans"][j][0] != ntok:
                    continue
                gains.append((v - p0) / ntok)
                qfs.append(float(v) / ntok)
                p0s.append(float(p0) / ntok)
                npos += ntok
                # KL 用：登记该成员在这个 target 上的 top-M 分布
                nt_cap = min(ntok, tk_ids.shape[2])
                mem_topm.setdefault(int(t), []).append(
                    (tk_ids[0, j, :nt_cap].clone(), tk_prs[0, j, :nt_cap].clone()))
                pos_of[int(t)] = (pos, ntok)
        # ---- D_KL(pi0 || q) 逐位置计算，q = [(1-eta)pi0 + eta qF]/Z ----
        z_rows, id_rows, pr_rows = [], [], []
        for t, mems in mem_topm.items():
            if t not in pos_of:
                continue
            pos, ntok = pos_of[t]
            T = min(ntok, min(m[0].shape[0] for m in mems))
            for u in range(T):
                # 成员 top-M 按 token id 算术平均（与 assemble_te_topm 同语义）
                acc = {}
                for mid, mpr in mems:
                    for tid, pr in zip(mid[u].tolist(), mpr[u].tolist()):
                        if tid >= 0 and pr > 0:
                            acc[tid] = acc.get(tid, 0.0) + pr
                if not acc:
                    continue
                n_m = len(mems)
                top = sorted(acc.items(), key=lambda x: -x[1])[:TE_TOPM]
                iv = [k2 for k2, _ in top] + [-1] * (TE_TOPM - len(top))
                pv = [v2 / n_m for _, v2 in top] + [0.0] * (TE_TOPM - len(top))
                r = pos + u
                if r >= lg.shape[0] - 1:
                    continue
                z_rows.append(lg[r])
                id_rows.append(iv)
                pr_rows.append(pv)
        if z_rows:
            zz = torch.stack(z_rows).float()
            ii = torch.tensor(id_rows, dtype=torch.long, device=zz.device)
            pp = torch.tensor(pr_rows, dtype=torch.float32, device=zz.device)
            for _eta, _key in ((eta, "kl"), (0.9, "kl90")):
                _v, _n = fullvocab_te_kl_rows(zz, ii, pp, _eta)
                kls.setdefault(_key, []).append((float(_v), _n))
            del zz, ii, pp
        del z_rows, id_rows, pr_rows

        if len(gains) > _n0:
            sl = slice(_n0, len(gains))
            _rec = {"traj_idx": ti, "n": len(gains) - _n0,
                    "gain": sum(gains[sl]) / (len(gains) - _n0),
                    "logqF": sum(qfs[sl]) / (len(gains) - _n0),
                    "logp0": sum(p0s[sl]) / (len(gains) - _n0)}
            for _key in ("kl", "kl90"):
                if kls.get(_key):
                    _v, _n = kls[_key][-1]
                    _rec[_key] = _v
            per_traj.append(_rec)
    gm = sum(gains) / len(gains) if gains else float("nan")
    qm = sum(qfs) / len(qfs) if qfs else float("nan")
    pm = sum(p0s) / len(p0s) if p0s else float("nan")
    def _wmean(key):
        rs = kls.get(key) or []
        tot = sum(n for _, n in rs)
        return (sum(v * n for v, n in rs) / tot) if tot else float("nan")
    return gm, len(gains), npos, qm, pm, per_traj, _wmean("kl"), _wmean("kl90")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="name=path,name=path")
    p.add_argument("--trajs", required=True, help="name=dir,name=dir（own 口径）")
    p.add_argument("--fixed-traj", default="", help="fixed 口径统一用的轨迹目录")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--max-traj", type=int, default=40)
    p.add_argument("--skip-own", action="store_true",
                   help="只算 fixed 口径，跳过 own（own 已证明不可用，省一半时间）")
    p.add_argument("--af-lora", default="", help="name=path,...：该模型的成员前向启用此 LoRA（交叉 TE）")
    p.add_argument("--out", default="/data1/logs/te_offline_result.json")
    a = p.parse_args()

    models = dict(x.split("=", 1) for x in a.models.split(","))
    trajs_d = dict(x.split("=", 1) for x in a.trajs.split(","))
    fixed = load_trajs(a.fixed_traj, a.max_traj) if a.fixed_traj else None
    print(f"[te_offline] k={a.k} eta={a.eta} 每模型最多 {a.max_traj} 条轨迹")
    if fixed:
        print(f"[te_offline] fixed 口径轨迹: {a.fixed_traj} ({len(fixed)} 条)")

    res = {}
    for name, mp in models.items():
        print(f"\n=== {name} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
        af_lora = None
        if a.af_lora:
            _map = dict(x.split("=", 1) for x in a.af_lora.split(","))
            if name in _map:
                af_lora = load_af_lora(model, _map[name])
        r = {}
        own = [] if a.skip_own else load_trajs(trajs_d[name], a.max_traj)
        if a.skip_own:
            r["own"] = {"gain_tok": float("nan"), "n_targets": 0, "n_pos": 0,
                        "n_traj": 0, "logqF_tok": float("nan"),
                        "logp0_tok": float("nan"), "per_traj": []}
        if not a.skip_own:
            g, n, npos, qm, pm, pt, kl, kl90 = te_for(model, tok, own, a.k, a.eta, "cuda", a.max_traj)
            r["own"] = {"gain_tok": g, "n_targets": n, "n_pos": npos, "n_traj": len(own),
                        "logqF_tok": qm, "logp0_tok": pm, "kl": kl, "kl90": kl90,
                        "per_traj": pt}
            print(f"  own   gain={g:+.4f} kl={kl:.4f} kl90={kl90:.4f}  targets={n}")
        if fixed:
            g2, n2, npos2, qm2, pm2, pt2, kl2, kl902 = te_for(model, tok, fixed, a.k, a.eta, "cuda", a.max_traj, af_lora=af_lora)
            r["fixed"] = {"gain_tok": g2, "n_targets": n2, "n_pos": npos2, "n_traj": len(fixed),
                          "logqF_tok": qm2, "logp0_tok": pm2, "kl": kl2, "kl90": kl902,
                          "per_traj": pt2}
            print(f"  fixed gain={g2:+.4f} = logqF {qm2:+.4f} - logp0 {pm2:+.4f} | "
                  f"KL(eta=.5)={kl2:.4f}  KL(eta=.9)={kl902:.4f}   targets={n2}")
        res[name] = r
        del model
        torch.cuda.empty_cache()

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n结果 -> {a.out}")
    if "base" in res:
        print(f"\n{'模型':<10}{'own gain':>12}{'vs base':>12}{'fixed gain':>13}{'vs base':>12}")
        b_own = res["base"].get("own", {}).get("gain_tok", float("nan"))
        b_fix = res["base"].get("fixed", {}).get("gain_tok", float("nan"))
        for k2, v in res.items():
            o = v.get("own", {}).get("gain_tok", float("nan")); f = v.get("fixed", {}).get("gain_tok", float("nan"))
            print(f"{k2:<10}{o:>12.4f}{o-b_own:>+12.4f}{f:>13.4f}{f-b_fix:>+12.4f}")

if __name__ == "__main__":
    main()
