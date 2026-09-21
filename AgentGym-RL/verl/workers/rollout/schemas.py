from dataclasses import dataclass
import re
from typing import Optional, List, Literal
from transformers import PreTrainedTokenizer
import torch


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def to_dict(self):
        return {'role': self.role, 'content': self.content}
    def __repr__(self):
        return str(self.to_dict())
    def __str__(self):
        return self.__repr_


# --- WM-SFT observation-target filtering (webshop) --------------------------------
# WebShop search-result pages are ~68% of all observation tokens, and ~83% of THOSE are
# product content (asin / title / price) drawn by BM25 over the catalog -- unpredictable
# from (state, action), so training the world model to predict them is pure gradient
# noise. This masks those fields OUT OF THE WM-SFT TARGET ONLY. The token ids fed to the
# policy are untouched: the agent still sees every product, it is just no longer asked
# to predict them.
#
# Verified over 368 real search-result pages: after the nav section the fields are
# strictly a repeating [asin, title, price] triple (368/368) and every asin matches
# ^B0[A-Z0-9]{8}$ (3660/3660). We still structurally re-validate per page and, on any
# mismatch, fall back to NOT filtering that page (fail-safe: never mask wrongly).
_WS_ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$")
_WS_PRICE_RE = re.compile(r"^(Price:\s*)?\$[\d.]+( to \$[\d.]+)?$")
_WS_PAGE_RE = re.compile(r"^Page \d+ \(Total results: \d+\)$")
_WS_NAV = ("Next >", "< Prev")
_WS_SEP = "[SEP]"


def _webshop_obs_target_mask(content: str, tokenizer) -> tuple:
    """Return (mask, n_dropped, n_total, failsafe) for one webshop observation.

    ``mask[i]`` is 1 if content token i should be a WM-SFT target. Only search-result
    pages are filtered; everything else keeps the current all-ones behaviour.
    """
    enc = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    if "Total results:" not in content:          # not a search-result page -> unchanged
        return [1] * n, 0, n, False

    # character spans of each [SEP]-delimited field
    spans, pos = [], 0
    for raw in content.split(_WS_SEP):
        start = pos
        pos += len(raw) + len(_WS_SEP)
        t = raw.strip()
        if t:
            off = raw.index(t)
            spans.append((start + off, start + off + len(t), t))

    # nav section ends at the last page-counter / Next> / <Prev field
    nav_end = -1
    for i, (_, _, t) in enumerate(spans):
        if _WS_PAGE_RE.match(t) or t in _WS_NAV:
            nav_end = i
    tail = spans[nav_end + 1:]

    # structural re-validation; bail out (keep everything) if the page is not the
    # expected [asin, title, price]* layout
    ok = nav_end >= 0 and len(tail) > 0 and len(tail) % 3 == 0
    if ok:
        for j in range(0, len(tail), 3):
            if not _WS_ASIN_RE.match(tail[j][2]) or not _WS_PRICE_RE.match(tail[j + 2][2]):
                ok = False
                break
    if not ok:
        return [1] * n, 0, n, True

    drop = [(a, b) for (a, b, _) in tail]
    mask, dropped = [1] * n, 0
    for i, (s0, s1) in enumerate(offsets):
        if s1 <= s0:
            continue
        for a, b in drop:
            if s0 < b and s1 > a:        # token overlaps a dropped field
                mask[i] = 0
                dropped += 1
                break
    return mask, dropped, n, False


# ---------------------------------------------------------------------------
# [thinking-model 适配] Qwen3 等混合推理模型默认会先生成 <think>...</think>。
# agent rollout 每轮只给 512 token，thinking 会把预算吃光，且干扰 Thought/Action 解析。
# 关闭方式是官方的 enable_thinking=False —— 它不是删标签，而是在生成提示末尾预置一个
# 空的 <think>\n\n</think>\n\n，让模型认为已思考完。
#
# 因此 assistat_prefix_msg 必须与 apply_chat_template 的产物**逐 token 一致**：
# add_assistant_message 靠 input_ids 的精确后缀匹配来判断该给哪段 loss mask，
# 两者不一致会直接抛 ValueError。
#
# 这里不写死，而是探测 tokenizer 本身：把 enable_thinking=False 传进去，若输出变化则
# 说明是 thinking 模型。Qwen2.5 等旧模型对未知 kwarg 静默忽略、输出不变，走原分支。
_THINKING_CACHE: dict = {}

def _supports_thinking(tokenizer) -> bool:
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if key in _THINKING_CACHE:
        return _THINKING_CACHE[key]
    probe = [{"role": "user", "content": "x"}]
    try:
        default = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
        nothink = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
        res = (default != nothink)
    except Exception:
        res = False
    _THINKING_CACHE[key] = res
    return res


def _thinking_kwargs(tokenizer) -> dict:
    return {"enable_thinking": False} if _supports_thinking(tokenizer) else {}


def _assistant_prefix(tokenizer) -> str:
    """生成提示前缀，与 apply_chat_template(**_thinking_kwargs) 的尾部保持一致。"""
    base = "\n<|im_start|>assistant\n"
    return base + "<think>\n\n</think>\n\n" if _supports_thinking(tokenizer) else base
# ---------------------------------------------------------------------------



def _action_token_mask(tokenizer, content: str, response_ids, task_name: str):
    """返回与 response_ids 等长的 bool 列表，标出**裸动作**所在的 token。

    TE 的 q_t 是动作串上的分布，所以 log π^0 必须只对动作 token 求和 ——
    带上 Thought 推理段会让它与成员侧（只覆盖裸动作）差 9 倍 token 数，
    序列级 log-prob 相差 10~28 nats，KL 项随之失去意义。

    用 return_offsets_mapping 做字符→token 映射（实测 2307/2307 个动作都是
    content 的字面子串）。任何一步失败都退回"全标"，与修改前行为一致 ——
    宁可退化成旧口径，也不要静默丢掉整轮。
    """
    n = len(response_ids)
    try:
        from verl.agent_trainer.ppo.plan_forecast import extract_action
        act = extract_action(content, env=(task_name or "").lower())
        if not act:
            return [True] * n
        cs = content.rindex(act)
        ce = cs + len(act)
        enc = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if len(offs) != n:                 # 与 encode(content) 的切分不一致，放弃
            return [True] * n
        mask = [(a < ce and b > cs) for (a, b) in offs]
        return mask if any(mask) else [True] * n
    except Exception:
        return [True] * n

class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        observation_mask: List[int],
        prompt_observation_mask: List[int],
        response_observation_mask: List[int],
        max_response_len: int = 8192,
        max_model_len: int = 32768,
        wm_obs_filter: bool = False,
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.observation_mask = observation_mask
        self.prompt_observation_mask = prompt_observation_mask
        self.response_observation_mask = response_observation_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len  
        # WM-SFT observation-target filtering (off by default; webshop only)
        self.wm_obs_filter = wm_obs_filter
        self.wm_obs_dropped = 0      # obs tokens excluded from the WM-SFT target
        self.wm_obs_total = 0        # obs tokens seen (denominator for the metric)
        self.wm_obs_failsafe = 0     # pages where the structure check bailed out
        # --- Temporal Ensembling: 每个 token 属于第几个动作轮（-1 = 非动作正文）---
        # 纯追加字段，不触碰 input_ids / loss_mask / attention_mask / position_ids /
        # observation_mask 中的任何一个。不改 __init__ 签名，所以构造点零改动。
        # 即使 TE 关闭也维护（开销是一次 list 追加）；只有 te_enable 时才放进 batch
        # （见 vllm_rollout.py 的门控），对正常训练零影响。
        self.turn_ids = [-1] * len(self.input_ids)
        self.prompt_turn_ids = [-1] * len(self.prompt_ids)
        self.response_turn_ids = []
        self._assistant_turn = -1    # 第一次 add_assistant_message 后变 0
        self.format_config: dict = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        # 2026-09-20: thinking 模型（Qwen3）走 token-in/token-out —— vLLM 直接吃训练序列，
        # 保证生成上下文 G 与训练上下文 T 逐 token 相同。原因与实测见 token_io.py 顶部。
        # 非 thinking 模型（Qwen2.5）走下面的原路径，逐字节不变。
        from verl.workers.rollout import token_io
        if token_io.uses_token_io(tokenizer):
            return token_io.generation_prompt(tokenizer, self.input_ids)
        conversations = [
            msg.to_dict() for msg in self.messages
        ]
        return tokenizer.apply_chat_template(conversations, add_generation_prompt=True,
                                             tokenize=True, **_thinking_kwargs(tokenizer))
    
    
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
        response_ids: Optional[List[int]] = None,
    ) -> None:
        msg = Message(role='assistant', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = _assistant_prefix(tokenizer)   # 随模型自动切换（thinking / 非 thinking）
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        # 2026-09-20: thinking 模型且传入了 vLLM 原始 token 时，直接用原始 token
        # （去掉末尾结束符/padding），不再 decode→encode —— 后者有 1.2% 的回复切分会变。
        # 非 thinking 模型（Qwen2.5）或未传 response_ids 时走原路径，逐字节不变。
        from verl.workers.rollout import token_io
        if response_ids is not None and token_io.uses_token_io(tokenizer):
            response = token_io.response_body(tokenizer, response_ids)
        else:
            response = tokenizer.encode(content, add_special_tokens=False)
        self._assistant_turn += 1
        _t = self._assistant_turn
        # TE 口径：turn_ids 只标**裸动作**的 token，不含 Thought 推理段。
        # 2026-09-15 修正：此前标的是整个 assistant 内容（实测平均 49 token），
        # 而 TE 成员侧的 slot span 只覆盖 extract_action 抽出的裸动作（5.3 token）。
        # 两侧 token 数差 9.2 倍 -> 序列级 log-prob 相差 10~28 nats，
        # 这个差值几乎全部来自长度而非集成质量，使 KL 项比较了两个不可比的量。
        # q_t 按定义就是**动作串**上的分布，所以两侧都必须是裸动作。
        _act_mask = _action_token_mask(tokenizer, content, response, self.task_name)
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
            _turn = [_t if m else -1 for m in _act_mask]
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
            _turn = [-1] * len(prefix_token_ids) + [_t if m else -1 for m in _act_mask]
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        # TE: suffix(<|im_end|>) 不算动作正文 —— q_t 是动作串上的分布，带上模板
        # token 会让 log π^0 与 log π^k 的口径不一致。
        _turn += [-1] * len(suffix_token_ids)
        _observation_mask = [0] * len(append_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.observation_mask += _observation_mask
        self.position_ids += _position_ids
        assert len(_turn) == len(append_token_ids), (len(_turn), len(append_token_ids))
        self.turn_ids += _turn
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask) == len(self.observation_mask), f"""Rollout Handler has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}, {len(self.observation_mask)=}"""
        
    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='user', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        # WM-SFT target mask for this observation. Default = all ones (unchanged
        # behaviour); only when wm_obs_filter is on AND this is webshop do we drop the
        # unpredictable product fields. content_token_ids is NEVER modified.
        _content_obs_mask = [1] * len(content_token_ids)
        if self.wm_obs_filter and str(self.task_name).lower() == "webshop":
            _m, _drop, _tot, _fs = _webshop_obs_target_mask(content, tokenizer)
            if len(_m) == len(content_token_ids):
                _content_obs_mask = _m
                self.wm_obs_dropped += _drop
                self.wm_obs_failsafe += int(_fs)
            self.wm_obs_total += _tot

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
            _observation_mask = list(_content_obs_mask)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
            _observation_mask = [0] * len(prefix_token_ids) + list(_content_obs_mask)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        _observation_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.observation_mask += _observation_mask
        self.position_ids += _position_ids
        self.turn_ids += [-1] * len(append_token_ids)   # TE: 观测 token 不属于任何动作轮
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask) == len(self.observation_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}, {len(self.observation_mask)=}"""
        
    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.observation_mask = self.observation_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
        self.response_observation_mask = self.observation_mask[len(self.prompt_observation_mask) :][: self.max_response_len]
        self.response_turn_ids = self.turn_ids[len(self.prompt_turn_ids) :][: self.max_response_len]
