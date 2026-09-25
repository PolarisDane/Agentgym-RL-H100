"""Fold an AF-LoRA adapter into the policy weights, producing a standard HF checkpoint.

LoRA is linear, so  y = xW^T + s * (xA^T)B^T = x (W + s B A)^T  exactly. Merging lets the
adapter-on model be evaluated through the same vLLM path (and the same protocol) as the
adapter-off policy, instead of teaching the serving stack about a hook-based adapter.

  python merge_af_lora.py --hf CKPT/actor/huggingface --lora CKPT/actor/af_lora.pt --out DIR
"""
import argparse, json, shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def clean(k: str) -> str:
    """AFLoRA stores module paths with '.' replaced by '__' and FSDP wrappers inlined;
    map back to the HF parameter path."""
    k = k.replace("_fsdp_wrapped_module__", "").replace("__fsdp_wrapped_module__", "")
    return k.replace("__", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.hf), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = torch.load(a.lora, map_location="cpu", weights_only=False)
    scale = float(d["alpha"]) / int(d["rank"])
    deltas = {}
    for k, v in d["lora"].items():
        kind, name = k.split(".", 1)
        deltas.setdefault(clean(name), {})[kind] = v.float()
    print(f"[merge] {len(deltas)} adapters, rank {d['rank']}, alpha {d['alpha']}, scale {scale}")

    applied = 0
    for shard in sorted(src.glob("*.safetensors")):
        sd = load_file(str(shard))
        for pname in list(sd.keys()):
            if not pname.endswith(".weight"):
                continue
            base = pname[: -len(".weight")]
            if base not in deltas:
                continue
            A, B = deltas[base]["A"], deltas[base]["B"]          # [r,in], [out,r]
            W = sd[pname]
            if B.shape[0] != W.shape[0] or A.shape[1] != W.shape[1]:
                raise ValueError(f"shape mismatch at {pname}: W{tuple(W.shape)} "
                                 f"A{tuple(A.shape)} B{tuple(B.shape)}")
            sd[pname] = (W.float() + scale * (B @ A)).to(W.dtype)
            applied += 1
        save_file(sd, str(out / shard.name), metadata={"format": "pt"})
        print(f"  {shard.name} -> {out / shard.name}")
    if applied != len(deltas):
        raise RuntimeError(f"only merged {applied} of {len(deltas)} adapters -- name mismatch")

    for extra in src.iterdir():
        if extra.suffix != ".safetensors":
            shutil.copy2(extra, out / extra.name)
    print(f"[merge] merged {applied} adapters -> {out}")


if __name__ == "__main__":
    main()
