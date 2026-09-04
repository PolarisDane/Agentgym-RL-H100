"""plan_forecast group_norm 的 per-sample 权重是否真正进入 loss。

修复前 dp_actor.py 做的是 `coef * lw.mean() * pf_loss`——把 [B] 权重塌成标量后
乘到已聚合的 loss 上，样本间相对权重全部丢失。修复后权重进入 token 级聚合。
这些测试锁住修复后的语义，并显式复现旧行为以证明两者不等价。
"""
import torch, pytest
from verl.agent_trainer.ppo.world_model_loss import (
    compute_world_model_sft_loss_from_logits as ce,
)

def _fixture(seed=0, B=4, T=6, V=11):
    """每条样本 loss_mask 覆盖后 3 个位置（shift 后 target 落在末尾）。"""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g)
    loss_mask = torch.zeros(B, T)
    loss_mask[:, -3:] = 1.0
    return logits, labels, loss_mask

def _per_sample_ce(logits, labels, loss_mask):
    """逐条样本单独算 CE（token-mean），作为独立参照实现。"""
    out = []
    for i in range(logits.size(0)):
        out.append(ce(logits[i:i+1], labels[i:i+1], loss_mask[i:i+1]))
    return torch.stack(out)


def test_none_is_bitwise_identical_to_old_path():
    """sample_weight=None 必须与旧实现逐位相同（不破坏 world_model 调用方）。"""
    logits, labels, loss_mask = _fixture()
    shift_mask = loss_mask[:, 1:].bool()
    ignored = labels[:, 1:].masked_fill(~shift_mask, -100)
    fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    tok = fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), ignored.reshape(-1))
    expected = tok.sum() / shift_mask.sum().clamp(min=1.0)
    assert torch.equal(ce(logits, labels, loss_mask), expected)


def test_uniform_weight_is_noop():
    """全 1 权重 == 不加权（加权平均的分母同步加权，尺度不漂移）。"""
    logits, labels, loss_mask = _fixture()
    w = torch.ones(logits.size(0))
    assert torch.allclose(ce(logits, labels, loss_mask, w),
                          ce(logits, labels, loss_mask), atol=1e-6)
    # 全体同比放大也应不变（是均值不是和）
    assert torch.allclose(ce(logits, labels, loss_mask, w * 7.5),
                          ce(logits, labels, loss_mask), atol=1e-6)


def test_matches_hand_computed_weighted_mean():
    """等长样本下，加权 loss 必须等于逐样本 CE 的加权平均（手算参照）。"""
    logits, labels, loss_mask = _fixture()
    w = torch.tensor([0.25, 0.25, 0.25, 3.25])       # 均值 1，最后一条重 13 倍
    per = _per_sample_ce(logits, labels, loss_mask)
    expected = (per * w).sum() / w.sum()             # 每条 token 数相同 -> 退化为样本加权平均
    assert torch.allclose(ce(logits, labels, loss_mask, w), expected, atol=1e-6)


def test_weights_actually_separate_samples():
    """★核心：权重不同 -> loss 必须不同。这正是 .mean() 塌缩杀死的性质。"""
    logits, labels, loss_mask = _fixture()
    per = _per_sample_ce(logits, labels, loss_mask)
    hi = int(per.argmax())                            # 挑 loss 最大/最小的两条
    lo = int(per.argmin())
    assert per[hi] - per[lo] > 1e-3, "fixture 区分度不足"

    B = logits.size(0)
    w_hi = torch.full((B,), 0.1); w_hi[hi] = B - 0.1 * (B - 1)   # 压在高 loss 样本上
    w_lo = torch.full((B,), 0.1); w_lo[lo] = B - 0.1 * (B - 1)   # 压在低 loss 样本上
    assert abs(w_hi.mean() - 1.0) < 1e-6 and abs(w_lo.mean() - 1.0) < 1e-6

    l_hi = ce(logits, labels, loss_mask, w_hi)
    l_lo = ce(logits, labels, loss_mask, w_lo)
    assert l_hi > l_lo, "权重未改变样本相对贡献 -> group_norm 仍然失效"

    # 复现旧行为：两个权重向量的 .mean() 都是 1.0 -> 旧代码给出完全相同的 loss
    base = ce(logits, labels, loss_mask)
    old_hi = w_hi.mean() * base
    old_lo = w_lo.mean() * base
    assert torch.allclose(old_hi, old_lo), "旧行为复现失败"
    assert not torch.allclose(l_hi, l_lo, atol=1e-4), "修复后应当可区分"


def test_zero_weight_excludes_sample():
    """权重 0 的样本必须对 loss 完全无贡献（等于只算其余样本）。"""
    logits, labels, loss_mask = _fixture()
    B = logits.size(0)
    w = torch.ones(B); w[0] = 0.0
    got = ce(logits, labels, loss_mask, w)
    rest = ce(logits[1:], labels[1:], loss_mask[1:])
    assert torch.allclose(got, rest, atol=1e-6)


def test_group_norm_end_to_end_semantics():
    """按 plan_forecast.py:687-691 (dedup=False) 生成权重，验证组间贡献比。

    组A 1条轨迹、组B 3条轨迹，每轨迹 1 个样本 -> B 的每条应是 A 的 1/3。
    """
    from collections import Counter
    traj = [("A", 1, 0, ("a",)), ("B", 1, 1, ("b1",)),
            ("B", 1, 2, ("b2",)), ("B", 1, 3, ("b3",))]
    n = 4
    wts = [1.0] * n
    m_g = Counter(g for g, _, _, _ in traj)
    for gid, cnt, start, _ in traj:
        for j in range(start, start + cnt):
            wts[j] = 1.0 / max(1, m_g[gid])
    mw = sum(wts) / len(wts)
    wts = [x / mw for x in wts]
    w = torch.tensor(wts)
    assert abs(w.mean() - 1.0) < 1e-6
    assert abs(w[0] / w[1] - 3.0) < 1e-6, "组A单条应为组B单条的 3 倍"

    logits, labels, loss_mask = _fixture(seed=3)
    per = _per_sample_ce(logits, labels, loss_mask)
    expected = (per * w).sum() / w.sum()
    assert torch.allclose(ce(logits, labels, loss_mask, w), expected, atol=1e-6)
    # 组A 贡献占比应为 3/(3+1+1+1) = 50%
    share = (per[0] * w[0]) / (per * w).sum()
    contrib = w[0] / w.sum()
    assert abs(contrib - 0.5) < 1e-6, f"组A 权重占比应为 50%，实际 {contrib:.4f}"


def test_unequal_target_lengths_are_token_weighted():
    """长度不等时是 token 加权，不是样本加权——记录这一既有语义。"""
    logits, labels, loss_mask = _fixture(seed=5, B=2, T=8)
    loss_mask[:] = 0.0
    loss_mask[0, -5:] = 1.0      # 样本0: 5 个监督位
    loss_mask[1, -2:] = 1.0      # 样本1: 2 个
    w = torch.ones(2)
    got = ce(logits, labels, loss_mask, w)
    per = _per_sample_ce(logits, labels, loss_mask)
    naive_sample_mean = per.mean()
    # token 加权 != 样本平均（长样本占更大比重）
    assert not torch.allclose(got, naive_sample_mean, atol=1e-4)
    n0 = int(loss_mask[0, 1:].sum()); n1 = int(loss_mask[1, 1:].sum())
    token_weighted = (per[0] * n0 + per[1] * n1) / (n0 + n1)
    assert torch.allclose(got, token_weighted, atol=1e-6)
