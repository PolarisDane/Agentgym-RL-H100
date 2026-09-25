"""Success-filtered SFT (RFT) on this step's own rollouts — the behavior-cloning
control for plan-forecast.

After the GRPO update, take the trajectories that SUCCEEDED this step and run one
extra SFT pass that imitates what the agent actually did in them: every action turn
becomes a (prefix, target) pair, the prefix being the conversation up to that turn
and the target the turn itself. No synthetic prompt, no future horizon, no plan —
just "do again what worked". That makes it the natural control for plan-forecast:
same win gate, same optimizer path (``update_plan_forecast``), same encoder
(``encode_sft_sample``), so the only thing that differs is WHAT is predicted.

``target='turn'`` (default) clones the whole assistant turn, Thought included —
standard rejection-sampling fine-tuning. ``target='action'`` keeps only the bare
action command (``extract_action``), which is exactly plan-forecast's target
vocabulary at horizon 1, and isolates "imitate the action" from "imitate the
reasoning that led to it".

Pure assembly logic: stdlib + tokenizer only, CPU-testable. The CE loss, the
padding and the optimizer step are all reused from the plan-forecast path.
"""
from __future__ import annotations

from typing import Dict, List, Optional

SFT_ABLATION_TARGETS = ("turn", "action")


def _pick(idxs: List[int], cap: Optional[int]) -> List[int]:
    """At most ``cap`` of ``idxs``, evenly spaced so the whole trajectory is covered.

    Taking the first ``cap`` would train only on openings, which in sciworld are the
    near-deterministic look-around turns; an even stride keeps early, middle and late
    turns of a win in proportion.
    """
    if not cap or cap <= 0 or len(idxs) <= cap:
        return idxs
    step = len(idxs) / float(cap)
    return [idxs[min(len(idxs) - 1, int(i * step))] for i in range(cap)]


def build_sft_ablation_batch(
    messages_list,
    tokenizer,
    rewards: Optional[List[float]] = None,
    gate: str = "wins",
    success_threshold: float = 0.5,
    max_length: int = 4096,
    max_samples_per_trajectory: Optional[int] = None,
    target: str = "turn",
    env: str = "",
):
    """Padded behavior-cloning batch over this step's winning trajectories.

    ``gate='wins'`` keeps trajectories whose reward > ``success_threshold`` (needs
    ``rewards`` aligned to ``messages_list``); ``gate='all'`` keeps everything, which
    turns this into plain imitation of the current policy and is only meaningful as a
    placebo. Returns ``(batch_dict | None, metrics)``; the dict carries
    ``input_ids`` / ``attention_mask`` / ``position_ids`` / ``loss_mask`` with the loss
    covering the target tokens only.
    """
    from verl.agent_trainer.ppo.plan_forecast import (
        _action_turn_indices,
        _to_chat_list,
        encode_sft_sample,
        extract_action,
    )
    from verl.agent_trainer.ppo.world_model_loss import collate_world_model_samples

    if gate not in ("wins", "all"):
        raise ValueError(f"sft_ablation gate must be 'wins' or 'all', got {gate!r}")
    if target not in SFT_ABLATION_TARGETS:
        raise ValueError(f"sft_ablation target must be one of {SFT_ABLATION_TARGETS}, got {target!r}")

    meta: Dict[str, float] = {
        "sft_ablation/gate_wins": 1.0 if gate == "wins" else 0.0,
        "sft_ablation/target_action": 1.0 if target == "action" else 0.0,
    }
    if gate == "wins" and rewards is None:
        # Without rewards there is no win to filter on; training on everything would
        # silently turn the control into plain self-imitation.
        meta["sft_ablation/skipped_no_rewards"] = 1.0
        return None, meta

    all_samples: List[dict] = []
    n_considered = 0
    n_used = 0
    n_empty_action = 0
    target_tokens: List[int] = []

    for i, msgs in enumerate(messages_list or []):
        if msgs is None:
            continue
        if gate == "wins":
            r = rewards[i] if i < len(rewards) else None
            if r is None or float(r) <= success_threshold:
                continue
        n_considered += 1

        convo = _to_chat_list(msgs)
        idxs = _pick(_action_turn_indices(convo), max_samples_per_trajectory)
        before = len(all_samples)
        for ai in idxs:
            content = convo[ai].get("content") or ""
            if target == "action":
                content = extract_action(content, env=env)
                if not content.strip():
                    # the trailing terminal turn, or a turn with no parsable action
                    n_empty_action += 1
                    continue
            sample = encode_sft_sample(
                tokenizer,
                prefix=convo[:ai],
                target_msgs=[{"role": "assistant", "content": content}],
                max_length=max_length,
            )
            if sample is None:
                continue
            target_tokens.append(int(sample["loss_mask"].sum().item()))
            all_samples.append(sample)
        if len(all_samples) > before:
            n_used += 1

    meta.update({
        "sft_ablation/n_traj_considered": float(n_considered),
        "sft_ablation/n_traj_used": float(n_used),
        "sft_ablation/n_samples": float(len(all_samples)),
        "sft_ablation/n_empty_action": float(n_empty_action),
        "sft_ablation/target_tokens_mean": (
            float(sum(target_tokens)) / len(target_tokens) if target_tokens else 0.0),
    })
    if not all_samples:
        return None, meta

    batch = collate_world_model_samples(
        all_samples,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        max_length=max_length,
    )
    return batch, meta
