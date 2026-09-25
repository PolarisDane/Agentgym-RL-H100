"""Behavioural foresight: free-running plan hit-rate on the fixed trajectory set.

At every action turn s of a held-out successful trajectory, the model is given the
SAME synthetic plan prompt used by the plan-forecast objective and GENERATES its next-k
action plan (greedy, no teacher forcing). Each generated line j is compared by exact
match against the realized action a_{s+j} of that trajectory.

Complements the TE metric: TE is a teacher-forced likelihood, this is free-running and
discrete, so a system cannot score well merely by being trained on a likelihood of the
same form. j=0 is the current action (imitation); j>=1 is foresight, matching TE's
k>=1 member rule.

  python plan_hit.py --model NAME=PATH[,...] [--af-lora NAME=PATH] --trajs DIR --out X.json
"""
import argparse, glob, json, re, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve()
AG = "/data/home/xingyangl/workspace/Agentgym-RL-H100/AgentGym-RL"
sys.path.insert(0, AG)
from verl.agent_trainer.ppo.plan_forecast import build_plan_targets, DEFAULT_PLAN_PROMPT, _to_chat_list


STOP = {"the", "a", "an"}


def norm(s: str, drop_articles: bool = False) -> str:
    s = s.strip().lower()
    s = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", s)     # "1." / "2)" / "-" / "*"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if drop_articles:
        s = " ".join(w for w in s.split() if w not in STOP)
    return s


PROSE = re.compile(r"^(thought|plan|reasoning|note|explanation|step\s*\d*)\s*:", re.I)


def extract_actions(text: str, k: int):
    """Pull action candidates out of a free-form generation, so the metric measures
    foresight rather than whether the system was trained on the bare-list format.

    Handles: bare lists (CoPE / the AF adapter), rollout style 'Thought: ... Action: x'
    (GRPO-trained policies and the base model), and numbered lists.
    """
    out = []
    lines = [ln.strip() for ln in text.split("\n")]
    for ln in lines:
        if not ln:
            continue
        if re.match(r"^action\s*:", ln, re.I):                 # 'Action: go to kitchen'
            rest = re.sub(r"^action\s*:\s*", "", ln, flags=re.I).strip()
            if rest:
                out.append(rest)
            continue
        if PROSE.match(ln):                                    # prose header -> skip
            continue
        if len(ln.split()) > 10:                               # prose sentence -> skip
            continue
        out.append(ln)
        if len(out) >= k:
            break
    # 'Action:' on its own line puts the action on the NEXT line; recover that case
    if not out:
        for i, ln in enumerate(lines[:-1]):
            if re.match(r"^action\s*:\s*$", ln, re.I) and lines[i + 1].strip():
                out.append(lines[i + 1].strip())
                break
    return out[:k]


def load_af_lora(model, path):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("_af_lora", f"{AG}/verl/agent_trainer/ppo/af_lora.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    d = torch.load(path, map_location="cpu", weights_only=False)
    lora = m.AFLoRA(model, rank=int(d["rank"]), alpha=float(d["alpha"]),
                    targets=tuple(d["targets"]), dtype=torch.float32,
                    device=next(model.parameters()).device)
    clean = lambda k: k.replace("_fsdp_wrapped_module__", "").replace("__fsdp_wrapped_module__", "")
    sd = {clean(k): v.float() for k, v in d["lora"].items()}
    miss = [k for k in lora.state_dict() if k not in sd]
    if miss:
        raise KeyError(f"af_lora key mismatch: {miss[:2]}")
    lora.load_state_dict(sd); lora.attach()
    print(f"  [af_lora] {len(lora.A)} adapters from step {d.get('global_step')}", flush=True)
    return lora


def build_prompts(trajs, tok, k, skip_invalid):
    """One prompt per action turn; identical construction to the training samples."""
    out = []
    for ti, tr in enumerate(trajs):
        convo = _to_chat_list(tr["conversations"])
        for tgt in build_plan_targets(tr["conversations"], k=k, skip_invalid=skip_invalid, env="sciworld"):
            items = tgt.get("actions") or []
            if not items:
                continue
            prefix = list(convo[:tgt["prefix_end"] + 1])
            prefix.append({"role": "user", "content": DEFAULT_PLAN_PROMPT.format(k=len(items))})
            text = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
            out.append({"traj": ti, "src": tgt.get("src_turn"), "prompt": text, "gold": items})
    return out


@torch.no_grad()
def run_model(path, prompts_of, k, lora_path, bs, max_new, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True).eval()
    lora = load_af_lora(model, lora_path) if lora_path else None
    items = prompts_of(tok)
    hits = [[0, 0] for _ in range(k)]           # [hit, total] per slot (exact)
    hits_loose = [[0, 0] for _ in range(k)]     # articles ignored
    unparsed = 0
    complied = 0                                # produced >= k action lines
    hits_cond = [[0, 0] for _ in range(k)]      # only over prompts that produced a full plan
    samples_out = []
    records = []
    from contextlib import nullcontext
    # NOTE: lora.enabled() is a @contextmanager -- single use. Build a fresh one per
    # batch, otherwise the second batch raises and the adapter silently stops applying.
    mk_ctx = (lambda: lora.enabled()) if lora is not None else (lambda: nullcontext())
    for b0 in range(0, len(items), bs):
        batch = items[b0:b0 + bs]
        enc = tok([x["prompt"] for x in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with mk_ctx():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        txts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for x, t in zip(batch, txts):
            acts = extract_actions(t, k)
            if not acts:
                unparsed += 1
            full_plan = len(acts) >= min(k, len(x["gold"]))
            if full_plan:
                complied += 1
            for j, gold in enumerate(x["gold"][:k]):
                hits[j][1] += 1
                if j < len(acts):
                    if norm(acts[j]) == norm(gold):
                        hits[j][0] += 1
                    if norm(acts[j], True) == norm(gold, True):
                        hits_loose[j][0] += 1
                hits_loose[j][1] += 1
                if full_plan:
                    hits_cond[j][1] += 1
                    if j < len(acts) and norm(acts[j], True) == norm(gold, True):
                        hits_cond[j][0] += 1
            rec = {"traj": x["traj"], "src": x["src"], "full_plan": bool(full_plan),
                   "gold": x["gold"][:k], "parsed": acts,
                   "hit": [int(j < len(acts) and norm(acts[j], True) == norm(gold, True))
                           for j, gold in enumerate(x["gold"][:k])],
                   "n_slots": len(x["gold"][:k])}
            records.append(rec)
            if len(samples_out) < 40:
                samples_out.append({"traj": x["traj"], "src": x["src"], "gold": x["gold"],
                                    "gen": t, "parsed": acts})
        if b0 % (bs * 20) == 0:
            print(f"    {b0+len(batch)}/{len(items)}", flush=True)
    del model
    torch.cuda.empty_cache()
    def pack(hh):
        return {"per_slot": [{"slot": j, "hit": h, "n": n, "rate": (h / n if n else float("nan"))}
                             for j, (h, n) in enumerate(hh)],
                "hit0": hh[0][0] / max(1, hh[0][1]),
                "future": sum(h for h, _ in hh[1:]) / max(1, sum(n for _, n in hh[1:]))}
    return {"exact": pack(hits), "loose": pack(hits_loose), "cond": pack(hits_cond),
            "plan_rate": complied / max(1, len(items)),
            "n_prompts": len(items), "unparsed": unparsed, "samples": samples_out,
            "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--af-lora", default="")
    p.add_argument("--trajs", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--skip-invalid", type=int, default=1)
    p.add_argument("--max-traj", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    files = sorted(glob.glob(f"{a.trajs}/sciworld_*.json"))[:a.max_traj]
    trajs = [json.load(open(f)) for f in files]
    print(f"[plan_hit] {len(trajs)} trajectories, k={a.k}, skip_invalid={bool(a.skip_invalid)}")
    lora_map = dict(x.split("=", 1) for x in a.af_lora.split(",")) if a.af_lora else {}

    res = {}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        print(f"\n=== {name} ===", flush=True)
        res[name] = run_model(path, lambda tok: build_prompts(trajs, tok, a.k, bool(a.skip_invalid)),
                              a.k, lora_map.get(name), a.batch_size, a.max_new)
        r = res[name]
        print(f"  exact: hit0 {r['exact']['hit0']*100:.1f}% future {r['exact']['future']*100:.1f}% | "
              f"loose: hit0 {r['loose']['hit0']*100:.1f}% future {r['loose']['future']*100:.1f}% | "
              f"cond_future {r['cond']['future']*100:.1f}% | "
              f"plan_rate {r['plan_rate']*100:.1f}%  prompts {r['n_prompts']}  unparsed {r['unparsed']}", flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
