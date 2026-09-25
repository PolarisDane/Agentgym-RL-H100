"""Frozen-representation linear probe: what does the hidden state encode about the
observation the environment is about to return?

At the last token of the action turn a_t -- after the action, before o_{t+1} -- extract
the hidden state and fit a linear multi-label classifier for "which entities appear in
o_{t+1}". Every system gets the same probe capacity, the same samples and the same
splits, so differences come only from what the representation carries (the Othello-GPT
style of evidence). Unlike output-space metrics this is unaffected by response format or
by the global drift in text likelihood that RL introduces.

Entities are bucketed exactly as in obs_probe.py:
  echo      : the entity is the object of a_t          (copy, no dynamics needed)
  carry     : it already appears in o_t                (copy)
  new_ref   : new, and referenced by a_{t+1..t+K}      (dynamics AND decision-relevant)
  new_noref : new, not referenced later                (dynamics only)

Writes hidden states + labels; fitting is done by --fit (CPU, sklearn).
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AG = str(Path(__file__).resolve().parent.parent / "AgentGym-RL")
sys.path.insert(0, AG)
from verl.agent_trainer.ppo.plan_forecast import _to_chat_list, extract_action

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obs_probe import build_points, entity_vocab, words, WORD


def collect(trajs, k, top_n):
    """Decision points + the global entity label space (most frequent entities)."""
    from collections import Counter
    pts, cnt = [], Counter()
    for ti, tr in enumerate(trajs):
        vocab = set(entity_vocab(tr))
        for p in build_points(tr, k):
            p["traj"] = ti
            p["vocab"] = vocab
            p["obs_w"] = words(p["next_obs"]) & vocab
            cnt.update(p["obs_w"])
            pts.append(p)
    labels = [w for w, _ in cnt.most_common(top_n)]
    return pts, labels


@torch.no_grad()
def extract(path, pts, labels, lora_path, device="cuda", layer=-1, bs=8):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
        lora.load_state_dict({clean(kk): v.float() for kk, v in d["lora"].items()})
        lora.attach()
        print(f"  [af_lora] {len(lora.A)} adapters", flush=True)
    from contextlib import nullcontext
    mk = (lambda: lora.enabled()) if lora is not None else (lambda: nullcontext())

    texts = [tok.apply_chat_template(p["ctx"], tokenize=False, add_generation_prompt=False)
             for p in pts]
    feats = np.zeros((len(pts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False, truncation=True, max_length=6144).to(device)
        with mk():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer][:, -1, :]          # last token = end of a_t
        feats[i:i + h.shape[0]] = h.float().cpu().numpy()
        if i % (bs * 20) == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return feats


def build_labels(pts, labels, k):
    """Y[i, j] = entity j occurs in o_{t+1}; B[i, j] = its bucket for that decision point."""
    idx = {w: j for j, w in enumerate(labels)}
    Y = np.zeros((len(pts), len(labels)), dtype=np.int8)
    B = np.empty((len(pts), len(labels)), dtype=object)
    groups = np.array([p["traj"] for p in pts])
    for i, p in enumerate(pts):
        prev_w = words(p["prev_obs"])
        fut_w = words(" ".join(p["future"]))
        act_w = words(extract_action(p["ctx"][-1]["content"]))
        for w, j in idx.items():
            Y[i, j] = int(w in p["obs_w"])
            B[i, j] = ("carry" if w in prev_w else "echo" if w in act_w
                       else "new_ref" if w in fut_w else "new_noref")
    return Y, B, groups


def fit_and_eval(feats, Y, B, groups, C=0.01, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats)
    X = scaler.transform(feats)
    gkf = GroupKFold(n_splits=folds)
    probs = np.full(Y.shape, np.nan)
    for tr_i, te_i in gkf.split(X, groups=groups):
        for j in range(Y.shape[1]):
            y = Y[tr_i, j]
            if y.sum() < 3 or (1 - y).sum() < 3:
                continue
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(X[tr_i], y)
            probs[te_i, j] = clf.predict_proba(X[te_i])[:, 1]
    out = {}
    for bucket in ["all", "new_ref", "new_noref", "echo", "carry"]:
        m = ~np.isnan(probs)
        if bucket != "all":
            m &= (B == bucket)
        y, p = Y[m], probs[m]
        out[bucket] = {"auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
                       "n": int(m.sum()), "pos": int(y.sum())}
    return out, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--af-lora", default="")
    ap.add_argument("--trajs", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-traj", type=int, default=60)
    ap.add_argument("--top-entities", type=int, default=50)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trajs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.trajs}/sciworld_*.json"))[:a.max_traj]]
    pts, labels = collect(trajs, a.k, a.top_entities)
    Y, B, groups = build_labels(pts, labels, a.k)
    print(f"[repr_probe] {len(pts)} decision points, {len(labels)} entity labels, "
          f"positives {int(Y.sum())}", flush=True)
    lora_map = dict(x.split("=", 1) for x in a.af_lora.split(",")) if a.af_lora else {}
    res = {}
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        print(f"\n=== {name} ===", flush=True)
        feats = extract(path, pts, labels, lora_map.get(name), layer=a.layer)
        res[name], probs = fit_and_eval(feats, Y, B, groups, C=a.C)
        # keep features and out-of-fold predictions: CIs, other probe capacities and
        # other bucketings can then be recomputed without touching a GPU again
        np.savez_compressed(Path(a.out).with_suffix("") .as_posix() + f".{name}.npz",
                            feats=feats, probs=probs, Y=Y, groups=groups,
                            B=np.array([[str(x) for x in row] for row in B]))
        print("  " + "  ".join(f"{kk} AUC {vv['auc']:.3f} (n={vv['n']},pos={vv['pos']})"
                               for kk, vv in res[name].items()), flush=True)
        Path(a.out).write_text(json.dumps({"labels": labels, "res": res}, ensure_ascii=False, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
