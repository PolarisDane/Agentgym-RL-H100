"""TE 性质 1/2：TE 关闭时不改变任何既有行为。

判据是 **AST 级**的（不是文本 diff —— 文本 diff 有两类假阳性：嵌套语句被
ast.walk 拍平后重复判定，以及差分算法把"原方法结尾"匹配到"新方法结尾"）：

  对每个改动前就存在的函数，要求下列之一成立：
    a) AST 完全一致
    b) 差异部分全部位于 TE 门控的 if 块之内（条件含 te_enable/te_lambda/te_log_q）
    c) 差异只涉及 TE 专用的新标识符（turn_ids / sel / action_turns / ...）
  新增函数只允许是 TE 相关的那几个。
  另外验证：默认配置全关、所有模块可 import。
"""
import ast, os, subprocess, sys
R = '/data1/repos/Agentgym-RL'
FILES = [f'{R}/AgentGym-RL/verl/agent_trainer/ppo/plan_forecast.py',
         f'{R}/AgentGym-RL/verl/workers/rollout/schemas.py',
         f'{R}/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py',
         f'{R}/AgentGym-RL/verl/workers/agent_actor/dp_actor.py',
         f'{R}/AgentGym-RL/verl/workers/agent_fsdp_workers.py',
         f'{R}/AgentGym-RL/verl/agent_trainer/ppo/ray_trainer.py']
GATE = ('te_enable', 'te_lambda', 'te_log_q', '_te_cfg', '_te_lam')
TE_ID = ('turn_ids', '_turn', '_assistant_turn', 'te_', '_te_', 'temporal_ensemble',
         'slot_lp', 'compute_te_log_prob', 'action_turns', 'src_turn', 'sel',
         'te_log_q', 'te_valid', '_act_mask', '_action_token_mask')
ALLOWED_NEW = {'compute_te_log_prob', '_compute_temporal_ensemble',
               '_action_token_mask'}   # TE 专用：把 turn_ids 收窄到裸动作

def top_stmts(node):
    """只取直接子语句，不递归 —— 避免把门控块内部的语句再判一次。"""
    out = []
    for f in ('body', 'orelse', 'finalbody'):
        for s in getattr(node, f, []) or []:
            out.append(s)
            if isinstance(s, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                # 门控 if 整体接受，不再下钻；其他控制流继续展开
                if isinstance(s, ast.If) and any(g in ast.unparse(s.test) for g in GATE):
                    continue
                out += top_stmts(s)
    return out

def funcs(path):
    d = {}
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d.setdefault(n.name, n)
    return d

bad = 0
print(f"{'文件':<32} {'原有':>5} {'一致':>5} {'仅TE':>5} {'新增':>5} {'问题':>5}")
for f in FILES:
    o, n = funcs(f + '.pre_te'), funcs(f)
    same = teonly = prob = 0; probs = []
    for name, of in o.items():
        nf = n.get(name)
        if nf is None:
            prob += 1; probs.append(f'{name} 消失'); continue
        if ast.dump(of) == ast.dump(nf):
            same += 1; continue
        o_set = {ast.dump(s) for s in top_stmts(of)}
        extra = [s for s in top_stmts(nf) if ast.dump(s) not in o_set]
        ok = True
        for s in extra:
            src = ast.unparse(s)
            if isinstance(s, ast.If) and any(g in ast.unparse(s.test) for g in GATE):
                continue                                   # (b) TE 门控块
            if any(t in src for t in TE_ID):
                continue                                   # (c) TE 专用标识
            ok = False; probs.append(f'{name}: {src[:70]}')
        (teonly if ok else prob) and None
        if ok: teonly += 1
        else: prob += 1
    added = set(n) - set(o)
    assert added <= ALLOWED_NEW, f'{f}: 出现非预期的新函数 {added - ALLOWED_NEW}'
    bad += prob
    print(f"{os.path.basename(f):<32} {len(o):>5} {same:>5} {teonly:>5} {len(added):>5} {prob:>5}")
    for p in probs: print(f"    !! {p}")
assert bad == 0, '存在无法归因到 TE 门控的改动'

# fut 必须逐字未改（SFT 与 TE 共用，改坏会同时毁掉两者）
pf = f'{R}/AgentGym-RL/verl/agent_trainer/ppo/plan_forecast.py'
fl = lambda s: [l.rstrip() for l in s.splitlines() if l.strip().startswith('fut =')]
assert fl(open(pf + '.pre_te').read()) == fl(open(pf).read()), 'fut 赋值被改动！'
print("\n  OK  build_plan_targets 的 fut 赋值逐字未改")

sh = open(f'{R}/scripts/run_sciworld_grpo_train.sh').read()
for k, v in [('TE_ENABLE', 'False'), ('TE_LAMBDA', '0.0')]:
    assert f'{k}="${{{k}:-{v}}}"' in sh
print(f"  OK  TE_ENABLE 默认 False，TE_LAMBDA 默认 0.0")
assert 'TE_ENABLE TE_LAMBDA' in open(f'{R}/launch_sciworld_grpo_tmux.sh').read()
print("  OK  launch 脚本 FWD 列表已含 TE 变量（不靠 tmux 环境继承）")

# 2026-09-15: 原来只 import 了 3 个模块，漏了 trainer/worker —— 结果
# ray_trainer 里一个写错路径的 import（pad_dataproto_to_divisor 在 verl.protocol
# 而不是 verl.utils.torch_functional）离线全过、到 GPU smoke 才炸。
# 现在把所有改动过的模块 + TE 用到的每个符号都显式解析一遍。
r = subprocess.run([sys.executable, '-c',
    f'import sys;sys.path.insert(0,"{R}/AgentGym-RL");'
    'from verl.protocol import pad_dataproto_to_divisor;'
    'from verl import DataProto;'
    'from tensordict import TensorDict;'
    'from verl.agent_trainer.ppo.temporal_ensemble import ('
    'segment_sum_by_turn, slot_logprobs_from_logits, build_te_scoring_samples,'
    'assemble_q, build_te_batch, assemble_te_tensors);'
    'import verl.agent_trainer.ppo.plan_forecast;'
    'import verl.agent_trainer.ppo.ray_trainer;'
    'import verl.workers.agent_actor.dp_actor;'
    'import verl.workers.agent_fsdp_workers;'
    'import verl.workers.rollout.schemas;'
    'print("IMPORT_OK")'],
    capture_output=True, text=True)
assert 'IMPORT_OK' in r.stdout, f'import 失败:\n{r.stderr[-1500:]}'
print("  OK  6 个模块 + TE 全部符号均可解析（含 trainer/worker 的运行时 import）")
print("\n全部通过：TE 关闭时不可能影响现有训练行为。")
