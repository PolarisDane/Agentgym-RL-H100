#!/usr/bin/env python3
"""Pull the curves the figures are drawn from out of wandb into figures/data/*.csv.

Talks to wandb's GraphQL API directly with the key from ~/.netrc. The wandb SDK
is not used on purpose: on this host its login goes through a local service
process that times out behind the proxy.

The machine reaches the internet through a tunnel on 127.0.0.1:7890, so run with

    http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
    no_proxy=localhost,127.0.0.1,::1 python scripts/fetch_wandb.py [dataset ...]

Every run is addressed by its wandb run id (the short code in the run URL), so
renaming a run in the UI cannot silently repoint a figure. README.md carries the
id -> experiment mapping in readable form.
"""
from __future__ import annotations

import csv
import json
import netrc
import sys
import time
from pathlib import Path

import requests

ENTITY, PROJECT = "2488721971-sjtu", "agentgym-sciworld"   # default location of a run
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# metric -> column name in the CSV
COMMON = {"critic/task_score/mean": "task_score", "critic/score/mean": "score",
          "critic/rewards/mean": "rewards"}
DIAG = {"critic/task_score/mean": "task_score", "critic/task_round/mean": "task_round",
        "plan_forecast/sft_loss": "sft_loss", "plan_forecast/n_samples": "n_samples",
        "plan_forecast/grad_norm": "pf_grad_norm", "plan_forecast/loss_weight_mean": "lw_mean",
        "plan_forecast/loss_weight_std": "lw_std", "plan_forecast/group_n_distilled": "n_groups",
        "plan_forecast/group_unique_frac": "uniq_frac", "plan_forecast/n_traj_used": "n_traj",
        "actor/entropy_loss": "entropy", "actor/kl_loss": "kl_loss",
        "actor/grad_norm": "actor_grad_norm", "response_length/mean": "resp_len",
        "timing_s/step": "step_time"}
TIMING = {"critic/task_score/mean": "task_score", "timing_s/step": "t_step",
          "timing_s/gen": "t_gen", "timing_s/update_actor": "t_update_actor",
          "timing_s/update_plan_forecast": "t_pf", "timing_s/old_log_prob": "t_logp",
          "timing_s/ref": "t_ref", "critic/task_round/mean": "task_round",
          "response_length/mean": "resp_len", "plan_forecast/n_samples": "n_samples",
          "actor/entropy_loss": "entropy", "actor/wm_sft_loss": "wm_sft_loss",
          "actor/world_model_coeff": "wm_coeff"}

# dataset -> (csv name, metrics, [(label, run id, extra columns), ...])
# Runs whose segments share a label are stitched by step (a resumed run).
# A run outside the default entity/project carries "entity"/"project" in its extra
# columns (they are dropped before the row is written, and kept in the CSV as
# plain columns so the source is visible there too).
DATASETS = {
    "horizon": ("sciworld_horizon_curves.csv", COMMON, [
        ("K1", "4ns6zsyw", {"K": 1, "cluster": "inspire-4gpu"}),
        ("K1_noskip", "1rnmj68v", {"K": 1, "cluster": "inspire-4gpu"}),
        ("K2", "dyk546f9", {"K": 2, "cluster": "ours-8gpu"}),
        ("K3_inspire", "wxahf6yd", {"K": 3, "cluster": "inspire-4gpu"}),
        ("K3_xy_a", "c1f9il4w", {"K": 3, "cluster": "clusterB-8gpu"}),
        ("K3_xy_b", "w2az2lg6", {"K": 3, "cluster": "clusterB-8gpu"}),
        ("K3_ours", "b4jtuf8m", {"K": 3, "cluster": "ours-8gpu"}),
        ("K4", "3kvxihjo", {"K": 4, "cluster": "ours-8gpu"}),
        ("K5", "47y1k0x0", {"K": 5, "cluster": "clusterB-8gpu"}),   # segment 1
        ("K5", "la0p7i43", {"K": 5, "cluster": "clusterB-8gpu"}),   # resumed segment
        ("BASELINE", "9tqxr5o7", {"K": 0, "cluster": "ours-8gpu"}),
    ]),
    "k3_pair": ("sciworld_k3_pair_curves.csv", DIAG, [
        ("A_gnorm_on", "w2az2lg6", {}),
        ("B_gnorm_off", "yfgxqpnu", {}),
    ]),
    "groupnorm": ("sciworld_groupnorm_curves.csv", DIAG, [
        ("gnorm_K3", "b4jtuf8m", {"group_norm": True}),
        ("nognorm_K3_v1", "cs7pem30", {"group_norm": False}),
        ("nognorm_K3_v2", "c3opfb2c", {"group_norm": False}),
        ("nognorm_K3_v3", "xvmlprq3", {"group_norm": False}),
        ("nognorm_K3_v4", "yfgxqpnu", {"group_norm": False}),
    ]),
    "skipinvalid": ("sciworld_skipinvalid_curves.csv", DIAG, [
        ("K1_skip", "4ns6zsyw", {"K": 1, "skip": True}),
        ("K1_noskip", "1rnmj68v", {"K": 1, "skip": False}),
        ("K2_skip", "dyk546f9", {"K": 2, "skip": True}),
        ("K2_noskip", "xwrvan9c", {"K": 2, "skip": False}),
    ]),
    "efficiency": ("sciworld_efficiency_curves.csv", TIMING, [
        ("ours_K2", "dyk546f9", {}),
        ("grpo", "9tqxr5o7", {}),
    ]),
    "echo": ("sciworld_echo_baseline_curves.csv", TIMING, [
        ("echo_wmsft", "6ip8v3j2", {}),
        ("grpo_inspire", "g5n2964d", {}),
    ]),
    # action-forecast coefficient sweep, run in the co-evolve-neurips project
    "af_coef": ("sciworld_af_coef_curves.csv", DIAG, [
        ("coef_0.0001", "tndk3zog", {"coef": 0.0001, "max_model_len": 32768, "max_tokens": 200,
                                     "entity": "co-evolve-neurips"}),
        ("coef_0.001", "h9op8mn0", {"coef": 0.001, "max_model_len": 8192, "max_tokens": 512,
                                    "entity": "co-evolve-neurips"}),
        ("coef_0.01", "wqqny0ib", {"coef": 0.01, "max_model_len": 32768, "max_tokens": 200,
                                   "entity": "co-evolve-neurips"}),
        ("coef_0.1", "321njph9", {"coef": 0.1, "max_model_len": 32768, "max_tokens": 200,
                                  "entity": "co-evolve-neurips"}),
        ("base_grpo", "cyn5yomo", {"coef": 0.0, "max_model_len": 32768, "max_tokens": 200,
                                   "entity": "co-evolve-neurips"}),
    ]),
}


def session():
    key = netrc.netrc().authenticators("api.wandb.ai")[2]
    s = requests.Session()
    s.auth = ("api", key)
    return s


def gql(s, query, variables):
    err = None
    for _ in range(5):
        try:
            r = s.post("https://api.wandb.ai/graphql",
                       json={"query": query, "variables": variables}, timeout=180)
            r.raise_for_status()
            d = r.json()
            if "errors" in d:
                raise RuntimeError(d["errors"])
            return d["data"]
        except Exception as e:                      # the tunnel drops out now and then
            err, _ = e, time.sleep(5)
    raise err


Q_KEYS = ("query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e)"
          "{run(name:$r){historyKeys displayName}}}")
Q_HIST = ("query($e:String!,$p:String!,$r:String!,$s:[JSONString!]!){project(name:$p,"
          "entityName:$e){run(name:$r){sampledHistory(specs:$s)}}}")


def fetch_run(s, run_id, metrics, entity=ENTITY, project=PROJECT):
    """{step: {column: value}} for one run.

    One request per metric on purpose: sampledHistory returns an EMPTY table if
    any requested key is missing from that run, and the older runs do not log
    every key.
    """
    info = gql(s, Q_KEYS, {"e": entity, "p": project, "r": run_id})["project"]["run"]
    have = set(info["historyKeys"]["keys"])
    rows: dict[int, dict] = {}
    for metric, column in metrics.items():
        if metric not in have:
            continue
        spec = json.dumps({"keys": ["_step", metric], "samples": 100000})
        hist = gql(s, Q_HIST, {"e": entity, "p": project, "r": run_id, "s": [spec]})
        for row in hist["project"]["run"]["sampledHistory"][0]:
            if row.get("_step") is not None:
                rows.setdefault(row["_step"], {})[column] = row.get(metric)
    return info["displayName"], rows


def build(name):
    csv_name, metrics, runs = DATASETS[name]
    s = session()
    columns = list(dict.fromkeys(metrics.values()))
    extra_keys = list(dict.fromkeys(k for _, _, extra in runs for k in extra))
    merged: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for label, run_id, extra in runs:
        display, rows = fetch_run(s, run_id, metrics,
                                  extra.get("entity", ENTITY), extra.get("project", PROJECT))
        merged.setdefault(label, {}).update(rows)          # later segment wins on overlap
        meta[label] = {"display": display, "run_id": run_id, **extra}
        print(f"  {label:<14} {run_id}  {display[:46]:<46} {len(rows):4d} steps")
    out = []
    for label, rows in merged.items():
        for step in sorted(rows):
            out.append({"label": label, "step": step, **meta[label],
                        **{c: rows[step].get(c) for c in columns}})
    path = DATA_DIR / csv_name
    fields = ["label", "step", "display", "run_id"] + extra_keys + columns
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(DATASETS)):
        print(f"== {name}")
        build(name)
