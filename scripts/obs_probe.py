"""Observation-prediction probe: did the action-forecast objective leave environment
dynamics in the parameters?

CoPE is never trained on observations, so any gain over a matched GRPO policy is
transfer, which pure planning distillation does not predict.

Two measurements per decision point (context = conversation up to and including the
action a_t, then the WM-SFT prompt verbatim):

  discrimination (primary) -- per-token NLL of the TRUE o_{t+1} vs a distractor
      observation drawn from the same trajectory. Immune to the global calibration
      drift that makes raw NLL levels incomparable across RL-trained models.

  NLL decomposition (secondary) -- mean NLL over four token classes:
      delta   : the word is NOT already in o_t   (what the action changed)
      carry   : the word is already in o_t       (copied context)
      crossed with
      ref     : the word occurs in a future action a_{t+1..t+K}
      noref   : it does not
      'policy-space world model' predicts the gain to concentrate in delta&ref.

Usage:
  python obs_probe.py --models name=path[,...] [--af-lora name=path] \
      --trajs DIR --max-traj 60 --out X.json
"""
import argparse, glob, json, random, re, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AG = str(Path(__file__).resolve().parent.parent / "AgentGym-RL")
sys.path.insert(0, AG)
from verl.agent_trainer.ppo.world_model_loss import DEFAULT_WORLD_MODEL_PROMPT
from verl.agent_trainer.ppo.plan_forecast import _to_chat_list, extract_action

WORD = re.compile(r"[a-z0-9]+")


def words(s: str):
    return set(WORD.findall(s.lower()))


def build_points(traj, k):
    """One entry per decision point: context messages, the true next observation,
    the o_t it follows, and the next-k realized actions (for the 'referenced' split)."""
    convo = _to_chat_list(traj["conversations"])
    ai = [i for i, m in enumerate(convo) if m["role"] == "assistant"]
    ai = ai[1:] if ai else []                      # drop the opening ack
    pts = []
    for n, i in enumerate(ai):
        if i + 1 >= len(convo) or convo[i + 1]["role"] != "user":
            continue
        prev_obs = convo[i - 1]["content"] if i >= 1 and convo[i - 1]["role"] == "user" else ""
        future = [extract_action(convo[j]["content"]) for j in ai[n + 1:n + 1 + k]]
        pts.append({"ctx": convo[: i + 1], "next_obs": convo[i + 1]["content"],
                    "prev_obs": prev_obs, "future": [f for f in future if f]})
    return pts


def token_classes(tok, target_text, prev_obs, future_actions):
    """Per-target-token class ids: 0 delta&ref, 1 delta&noref, 2 carry&ref, 3 carry&noref."""
    enc = tok(target_text, add_special_tokens=False, return_offsets_mapping=True)
    prev_w = words(prev_obs)
    fut_w = words(" ".join(future_actions))
    cls = []
    for a, b in enc["offset_mapping"]:
        # the word containing this token's start char
        s = target_text.rfind(" ", 0, a) + 1
        e = target_text.find(" ", b)
        w = target_text[s: e if e > 0 else len(target_text)].lower()
        w = "".join(WORD.findall(w))
        delta = w not in prev_w
        ref = w in fut_w
        cls.append(0 if (delta and ref) else 1 if delta else 2 if ref else 3)
    return enc["input_ids"], cls



VERBS = {"go", "to", "open", "close", "focus", "on", "move", "pick", "up", "put", "in",
         "activate", "deactivate", "look", "around", "at", "read", "wait", "wait1",
         "examine", "use", "mix", "pour", "connect", "teleport", "dunk", "eat", "drink",
         "flush", "the", "a", "an", "and", "with", "into", "from", "inventory", "task"}


def entity_vocab(traj):
    """Candidate entities of one trajectory: content words appearing in its ACTIONS.

    Actions in SciWorld are 'verb + object [+ prep + object]', so dropping the verb /
    preposition vocabulary leaves objects, receptacles and locations -- exactly the
    slots whose value is determined by the environment's response to the action.
    """
    convo = _to_chat_list(traj["conversations"])
    ai = [i for i, m in enumerate(convo) if m["role"] == "assistant"][1:]
    out = set()
    for i in ai:
        for w in WORD.findall(extract_action(convo[i]["content"]).lower()):
            if len(w) >= 3 and w not in VERBS:
                out.add(w)
    return sorted(out)


def entity_spans(text, vocab):
    """Char spans of vocabulary entities occurring in text (longest match, no overlap)."""
    low = text.lower()
    spans = []
    for w in sorted(vocab, key=len, reverse=True):
        start = 0
        while True:
            i = low.find(w, start)
            if i < 0:
                break
            start = i + len(w)
            if (i and low[i - 1].isalnum()) or (start < len(low) and low[start].isalnum()):
                continue                       # not a whole word
            if any(not (start <= a or i >= b) for a, b in spans):
                continue                       # overlaps an existing (longer) match
            spans.append((i, start))
    return sorted(spans)


@torch.no_grad()
def cloze_at(model, tok, ctx_msgs, obs, span, true_word, alts, device, max_len=6144):
    """Forced choice at one entity slot: is the TRUE entity more likely than the
    alternatives, given identical left context (conversation + o_{t+1} up to the slot)?

    Chance level is 1/(1+len(alts)), so unlike whole-observation NLL this has headroom,
    and every sample sits exactly on a token whose value the action determined.
    """
    full = tok.apply_chat_template(list(ctx_msgs) + [{"role": "user", "content": obs}],
                                   tokenize=False, add_generation_prompt=False)
    i = full.rfind(obs)
    if i < 0:
        return None
    left = full[: i + span[0]]
    lids = tok(left, add_special_tokens=False)["input_ids"]
    cands = [obs[span[0]:span[1]]] + list(alts)
    seqs, lens = [], []
    for c in cands:
        cids = tok(c, add_special_tokens=False)["input_ids"]
        if not cids:
            return None
        ids = lids + cids
        if len(ids) > max_len:
            ids = ids[len(ids) - max_len:]
        seqs.append(ids); lens.append(len(cids))
    width = max(len(x) for x in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batch = torch.full((len(seqs), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(batch)
    for r, ids in enumerate(seqs):                      # left pad, keeps the slot at the end
        batch[r, width - len(ids):] = torch.tensor(ids, device=device)
        mask[r, width - len(ids):] = 1
    logits = model(input_ids=batch, attention_mask=mask).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1)
    tgt = batch[:, 1:]
    tok_lp = lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    scores = []
    for r, n in enumerate(lens):
        scores.append(-tok_lp[r, -n:].mean().item())    # length-normalised NLL
    return int(min(range(len(scores)), key=lambda j: scores[j]) == 0), scores


@torch.no_grad()
def nll_of(model, tok, ctx_msgs, target_text, device, mode="natural", max_len=6144):
    """Teacher-forced per-token NLL of target_text.

    mode='natural'  : the observation is scored where it actually occurs, as the next
        user turn. No synthetic instruction, so the conditioning is byte-identical to
        what every system saw in training / rollout -- the prompt-familiarity confound
        disappears, and this is the quantity the repo's own env-token NLL uses.
    mode='prompted' : the WM-SFT instruction is inserted first. Kept as a robustness
        check and for fairness to any future WM-SFT reference, which trains in this form.
    """
    if mode == "prompted":
        prefix = list(ctx_msgs) + [{"role": "user", "content": DEFAULT_WORLD_MODEL_PROMPT}]
        ptext = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
    else:
        full = tok.apply_chat_template(list(ctx_msgs) + [{"role": "user", "content": target_text}],
                                       tokenize=False, add_generation_prompt=False)
        i = full.rfind(target_text)
        if i < 0:
            return None
        ptext = full[:i]
    pids = tok(ptext, add_special_tokens=False)["input_ids"]
    tids = tok(target_text, add_special_tokens=False)["input_ids"]
    if not tids:
        return None
    ids = pids + tids
    if len(ids) > max_len:                      # left-truncate the context only
        ids = ids[len(ids) - max_len:]
        if len(ids) <= len(tids):
            return None
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)
    tgt = x[0, 1:]
    tok_lp = lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
    return (-tok_lp[-len(tids):]).cpu()          # per-token NLL of the target


@torch.no_grad()
def run(path, trajs, k, lora_path, device="cuda", seed=0, mode="natural"):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map=device, trust_remote_code=True).eval()
    lora = None
    if lora_path:
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("_af_lora", f"{AG}/verl/agent_trainer/ppo/af_lora.py")
        m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
        d = torch.load(lora_path, map_location="cpu", weights_only=False)
        lora = m.AFLoRA(model, rank=int(d["rank"]), alpha=float(d["alpha"]),
                        targets=tuple(d["targets"]), dtype=torch.float32, device=device)
        clean = lambda kk: kk.replace("_fsdp_wrapped_module__", "").replace("__fsdp_wrapped_module__", "")
        sd = {clean(kk): v.float() for kk, v in d["lora"].items()}
        miss = [kk for kk in lora.state_dict() if kk not in sd]
        if miss:
            raise KeyError(f"af_lora key mismatch: {miss[:2]}")
        lora.load_state_dict(sd); lora.attach()
        print(f"  [af_lora] {len(lora.A)} adapters, step {d.get('global_step')}", flush=True)

    from contextlib import nullcontext
    mk = (lambda: lora.enabled()) if lora is not None else (lambda: nullcontext())
    rng = random.Random(seed)
    cls_sum = [0.0] * 4
    cls_n = [0] * 4
    wins = tot = 0
    per_traj = {}
    cloze = {"new_ref": [0, 0], "new_noref": [0, 0], "echo": [0, 0], "carry": [0, 0]}
    abl, abl_traj = {}, {}
    cloze_traj = {}
    for ti, tr in enumerate(trajs):
        pts = build_points(tr, k)
        obs_pool = [p["next_obs"] for p in pts]
        vocab = entity_vocab(tr)
        if mode == "ablate":
            # Intervention probe: score the SAME entity slot twice, changing only the
            # action in the context. A model that carries environment dynamics must lose
            # accuracy when the action is swapped; a model that copies or relies on a text
            # prior is unaffected. Same slot / same candidates / same model, so the global
            # drift that made copy-bucket normalisation ambiguous cancels by construction.
            acts = [extract_action(tr["conversations"][i]["content"])
                    for i, m in enumerate(_to_chat_list(tr["conversations"]))
                    if m["role"] == "assistant"]
            acts = [a for a in acts[1:] if a]
            for p in pts:
                prev_w, fut_w = words(p["prev_obs"]), words(" ".join(p["future"]))
                present = words(p["next_obs"])
                pool = [w for w in vocab if w not in present]
                true_act = extract_action(p["ctx"][-1]["content"])
                others = [a for a in acts if a and a != true_act]
                if len(pool) < 3 or not others or not true_act:
                    continue
                # both conditions use an identically formatted turn: only the action differs
                ctx_true = list(p["ctx"][:-1]) + [{"role": "assistant", "content": f"Action:\n{true_act}"}]
                ctx_fake = list(p["ctx"][:-1]) + [{"role": "assistant",
                                                   "content": f"Action:\n{rng.choice(others)}"}]
                for sp in entity_spans(p["next_obs"], vocab):
                    w = p["next_obs"][sp[0]:sp[1]].lower()
                    if w in prev_w:
                        bucket = "carry"
                    elif w in words(true_act):
                        bucket = "echo"
                    else:
                        bucket = "new_ref" if w in fut_w else "new_noref"
                    alts = rng.sample(pool, 3)
                    with mk():
                        r1 = cloze_at(model, tok, ctx_true, p["next_obs"], sp, w, alts, device)
                        r2 = cloze_at(model, tok, ctx_fake, p["next_obs"], sp, w, alts, device)
                    if r1 is None or r2 is None:
                        continue
                    b = abl.setdefault(bucket, [0, 0, 0])
                    b[0] += r1[0]; b[1] += r2[0]; b[2] += 1
                    t = abl_traj.setdefault(ti, {}).setdefault(bucket, [0, 0, 0])
                    t[0] += r1[0]; t[1] += r2[0]; t[2] += 1
            if ti % 10 == 0:
                b = abl.get("new_ref", [0, 0, 0])
                print(f"    traj {ti}/{len(trajs)} new_ref 真实 {b[0]}/{b[2]} 替换 {b[1]}/{b[2]}", flush=True)
            continue
        if mode == "cloze":
            for p in pts:
                prev_w, fut_w = words(p["prev_obs"]), words(" ".join(p["future"]))
                # The observation is largely an echo of a_t in SciWorld, so an entity that
                # is simply the object of the action just taken needs no dynamics knowledge
                # to fill in -- it is copied from the context. Split it out.
                last = p["ctx"][-1]["content"]
                cur_w = words(extract_action(last))
                present = words(p["next_obs"])
                pool = [w for w in vocab if w not in present]
                if len(pool) < 3:
                    continue
                for sp in entity_spans(p["next_obs"], vocab):
                    w = p["next_obs"][sp[0]:sp[1]].lower()
                    if w in prev_w:
                        bucket = "carry"
                    elif w in cur_w:
                        bucket = "echo"                      # object of a_t: copy, not dynamics
                    else:
                        bucket = "new_ref" if w in fut_w else "new_noref"
                    alts = rng.sample(pool, 3)
                    with mk():
                        r = cloze_at(model, tok, p["ctx"], p["next_obs"], sp, w, alts, device)
                    if r is None:
                        continue
                    hit, _ = r
                    cloze[bucket][0] += hit; cloze[bucket][1] += 1
                    t = cloze_traj.setdefault(ti, {}).setdefault(bucket, [0, 0])
                    t[0] += hit; t[1] += 1
            if ti % 10 == 0:
                d = cloze["new_ref"]; e = cloze["new_noref"]
                print(f"    traj {ti}/{len(trajs)} new&ref {d[0]}/{d[1]} new&noref {e[0]}/{e[1]}", flush=True)
            continue
        for pi, p in enumerate(pts):
            with mk():
                nll = nll_of(model, tok, p["ctx"], p["next_obs"], device, mode)
            if nll is None:
                continue
            ids, cls = token_classes(tok, p["next_obs"], p["prev_obs"], p["future"])
            n = min(len(cls), nll.numel())
            for c, v in zip(cls[:n], nll[:n].tolist()):
                cls_sum[c] += v; cls_n[c] += 1
            # discrimination against a distractor observation from the same trajectory
            cands = [o for j, o in enumerate(obs_pool) if j != pi and o.strip() != p["next_obs"].strip()]
            if cands:
                dist = rng.choice(cands)
                with mk():
                    dnll = nll_of(model, tok, p["ctx"], dist, device, mode)
                if dnll is not None:
                    a, b = nll.mean().item(), dnll.mean().item()   # length-normalised
                    tot += 1
                    if a < b:
                        wins += 1
                    t = per_traj.setdefault(ti, [0, 0]); t[1] += 1; t[0] += int(a < b)
        if ti % 10 == 0:
            print(f"    traj {ti}/{len(trajs)} acc so far {wins/max(1,tot)*100:.1f}%", flush=True)
    del model
    torch.cuda.empty_cache()
    if mode == "ablate":
        return {"mode": mode,
                "ablate": {k2: {"true_hit": v[0], "fake_hit": v[1], "n": v[2],
                                "acc_true": v[0] / v[2] if v[2] else float("nan"),
                                "acc_fake": v[1] / v[2] if v[2] else float("nan")}
                           for k2, v in abl.items()},
                "per_traj_ablate": {str(k2): v for k2, v in abl_traj.items()}}
    if mode == "cloze":
        return {"mode": mode,
                "cloze": {k2: {"hit": v[0], "n": v[1], "acc": (v[0] / v[1] if v[1] else float("nan"))}
                          for k2, v in cloze.items()},
                "per_traj_cloze": {str(k2): v for k2, v in cloze_traj.items()}}
    names = ["delta&ref", "delta&noref", "carry&ref", "carry&noref"]
    return {"mode": mode, "discrim_acc": wins / max(1, tot), "n_pairs": tot,
            "nll": {nm: (cls_sum[i] / cls_n[i] if cls_n[i] else float("nan")) for i, nm in enumerate(names)},
            "n_tokens": {nm: cls_n[i] for i, nm in enumerate(names)},
            "per_traj": {str(kk): v for kk, v in per_traj.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--af-lora", default="")
    p.add_argument("--trajs", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-traj", type=int, default=60)
    p.add_argument("--mode", default="natural", choices=["natural", "prompted", "cloze", "ablate"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    trajs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.trajs}/sciworld_*.json"))[:a.max_traj]]
    lora_map = dict(x.split("=", 1) for x in a.af_lora.split(",")) if a.af_lora else {}
    res = {}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        print(f"\n=== {name} ===", flush=True)
        res[name] = run(path, trajs, a.k, lora_map.get(name), mode=a.mode)
        r = res[name]
        if a.mode == "ablate":
            print("  " + "  ".join(f"{kk} 真实{vv['acc_true']*100:.1f}% 替换{vv['acc_fake']*100:.1f}% "
                                   f"敏感度{(vv['acc_true']-vv['acc_fake'])*100:+.1f} (n={vv['n']})"
                                   for kk, vv in r["ablate"].items()), flush=True)
        elif a.mode == "cloze":
            print("  cloze " + "  ".join(f"{kk} {vv['acc']*100:.1f}% (n={vv['n']})"
                                         for kk, vv in r["cloze"].items()), flush=True)
        else:
            print(f"  discrim {r['discrim_acc']*100:.1f}% (n={r['n_pairs']})  NLL " +
                  "  ".join(f"{kk} {vv:.3f}" for kk, vv in r["nll"].items()), flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
