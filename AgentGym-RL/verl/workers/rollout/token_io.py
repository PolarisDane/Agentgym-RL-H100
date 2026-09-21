"""Token-in / token-out 多轮序列构造 —— 仅用于 thinking 模型（Qwen3 等）。

为什么需要（2026-09-20 实测）：
  旧路径里 vLLM 每轮用 ``apply_chat_template(全部消息)`` 重新渲染 prompt（G），
  训练用的序列则是逐轮拼接出来的（T）。对 Qwen2.5 两者逐 token 相同；对 Qwen3 不同：
    1. rl_dataset 写死了 Qwen2.5 的默认 system prompt，Qwen3 模板不加它（+21 token）
    2. Qwen3 模板在重新渲染时删掉历史 assistant 轮的 <think></think>，而逐轮拼接的
       T 每轮都带（每轮 +4 token）
  训练 50 步后，同一条回复在 T 与 G 下的 log-prob 相差 35~63 nats/轮 —— 训练严重离策略，
  最终以 PPO ratio 爆炸（pg_loss 9e8、grad_norm 1e13）的形式崩溃。

本模块的做法：所有序列（初始 prompt、每轮生成提示、训练序列、评测 prompt）都由同一套
token 拼接规则产生，vLLM 直接吃训练序列，生成结果以原始 token 追加 —— G 与 T 按构造相等。

**只由 thinking 模型启用**（``uses_token_io``）。Qwen2.5 等模型完全不经过这里，
旧代码路径逐字节不变。

本文件只依赖 tokenizer，不 import verl 其他模块，评测脚本可按文件路径直接加载。
"""
from typing import Dict, List

_PROBE_CACHE: Dict = {}


def uses_token_io(tokenizer) -> bool:
    """模板认 ``enable_thinking`` 即为 thinking 模型 → 走 token-io 路径。

    与 schemas._supports_thinking 同一判据；Qwen2.5 对未知 kwarg 静默忽略、输出不变，返回 False。
    """
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if key not in _PROBE_CACHE:
        probe = [{"role": "user", "content": "x"}]
        try:
            a = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
            b = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
            _PROBE_CACHE[key] = (a != b)
        except Exception:
            _PROBE_CACHE[key] = False
    return _PROBE_CACHE[key]


# 与 schemas.RolloutHandler 的 format_config["qwen"] 逐字一致
ASSISTANT_PREFIX = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
USER_PREFIX = "\n<|im_start|>user\n"
TURN_SUFFIX = "<|im_end|>"


def initial_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """初始 prompt（指令 + 应答）的文本，用模型自己的模板渲染（不再写死 Qwen2.5 的 system prompt）。

    以 ``<|im_end|>`` 结尾（去掉模板末尾的换行），与 RolloutHandler 的拼接约定对接：
    后续每段都以 "\\n<|im_start|>..." 开头。
    """
    s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                      enable_thinking=False)
    return s.rstrip("\n")


def prefix_ids(tokenizer) -> List[int]:
    return tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)


def suffix_ids(tokenizer) -> List[int]:
    return tokenizer.encode(TURN_SUFFIX, add_special_tokens=False)


def user_turn_ids(tokenizer, content: str) -> List[int]:
    """与 RolloutHandler.add_user_message 完全相同：前缀、内容、后缀**分别**编码再拼接。
    整串一起编码的话，BPE 在分段边界上的切分可能不同，会破坏逐 token 一致。"""
    return (tokenizer.encode(USER_PREFIX, add_special_tokens=False)
            + tokenizer.encode(content, add_special_tokens=False)
            + tokenizer.encode(TURN_SUFFIX, add_special_tokens=False))


def generation_prompt(tokenizer, ids: List[int]) -> List[int]:
    """在已有序列后接上 assistant 前缀（若尚未接上）—— 这就是 vLLM 该看到的 prompt。"""
    p = prefix_ids(tokenizer)
    return list(ids) if list(ids[-len(p):]) == p else list(ids) + p


def response_body(tokenizer, gen_ids: List[int]) -> List[int]:
    """vLLM 原始生成 token 去掉末尾的结束符与 padding（都是特殊 token）。

    直接用原始 token，不再 decode→encode：实测 1.2% 的回复重新编码后切分不同
    （例 生成 '[','cher','ry',']' → 重编码 '[ch','erry',']'），训练会算到模型没生成过的 token 上。
    """
    special = set(tokenizer.all_special_ids)
    ids = list(gen_ids)
    while ids and ids[-1] in special:
        ids.pop()
    return ids


def run_episode(tokenizer, generate_ids, step_fn, instruction: str, ack: str, first_obs: str,
                max_rounds: int) -> dict:
    """评测用的多轮循环，逐 token 复现训练 rollout（vllm_rollout.agent_step + RolloutHandler）。

    generate_ids(prompt_ids) -> 生成的原始 token 列表
    step_fn(content)         -> (next_obs, reward, done)；next_obs 对应训练里的 step_output.state

    与训练一致的细节：
      - 初始 prompt 用 initial_prompt_text（模型自己的模板）
      - 每轮 prompt = 当前序列 + assistant 前缀（generation_prompt）
      - 追加的是原始 token（response_body），发给环境的是 decode(skip_special_tokens=True)
        且**不 strip**（训练里也不 strip）
    """
    msgs = [{"role": "user", "content": instruction}, {"role": "assistant", "content": ack}]
    ids = tokenizer(initial_prompt_text(tokenizer, msgs), add_special_tokens=False)["input_ids"]
    ids = ids + user_turn_ids(tokenizer, first_obs)
    conversation = msgs + [{"role": "user", "content": first_obs}]
    sfx = suffix_ids(tokenizer)
    reward, done, rounds, term = 0.0, False, 0, "max_rounds"
    while not done and rounds < max_rounds:
        prompt = generation_prompt(tokenizer, ids)
        gen = generate_ids(prompt)
        body = response_body(tokenizer, gen)
        content = tokenizer.decode(gen, skip_special_tokens=True)
        ids = prompt + body + sfx
        conversation.append({"role": "assistant", "content": content})
        state, reward, done = step_fn(content)
        ids = ids + user_turn_ids(tokenizer, state)
        conversation.append({"role": "user", "content": state})
        rounds += 1
        if done:
            term = "env_done"
    return {"conversation": conversation, "reward": float(reward), "done": bool(done),
            "rounds": rounds, "terminated_by": term, "n_tokens": len(ids)}
