# -*- coding: utf-8 -*-
"""
mmlu_router_4choice_5plots_vote_ema.py

- 多学科 MMLU 4 选一评测
- ORCH: 三 Agent 并行分析（≤500字）+ 合并 Agent 只出 A/B/C/D
- 基线: OPENAI / DEEPSEEK / XAI 各自直接选一次
- VOTE: 三基线直接 vote，多数表决
- EMA: 对每个 agent 维护
    * ema_quality     (近期正确率，0~1)
    * ema_latency_ms  (近期延迟，ms)
    * ema_cost        (近期调用成本，按 tokens 估算)
    * ema_stability   (近期调用是否成功，0/1 的 EMA)

- Orchestrator: 用 EMA 打分来选 panel：
    score ≈ ema_quality
            + 0.1 * ema_stability
            - 0.3 * norm_latency
            - 0.1 * norm_cost

- 评测设置：
  * 默认 10 个学科，每科 30 题（可通过命令行修改）
    --num_subjects N
    --per_subject K
  * 温度 temperature=0.0（避免随机波动）

- 输出：
  * 控制台：
      - GLOBAL ACC（ORCH + 3 基线 + VOTE）
      - PER-SUBJECT ACC
      - ORCH 的混淆矩阵
      - McNemar (ORCH vs DEEPSEEK)
      - 每个 agent 的最终 EMA 指标（quality / latency / cost / stability）
  * 图像（在 PLOT_DIR 下）：
      1) global_acc.png
      2) per_subject_acc.png
      3) orch_confusion.png
      4) mcnemar_orch_vs_deepseek.png
      5) price_latency_table.png
      6) ema_curves.png       按题目顺序画出每个 agent 的 EMA 曲线（quality & latency）
"""

from __future__ import annotations
import os, json, time, random, argparse, re, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ========== Eval deps ==========
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("需要安装 datasets：pip install datasets") from e
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("需要安装 numpy：pip install numpy") from e
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("需要安装 matplotlib：pip install matplotlib") from e
try:
    from sklearn.metrics import confusion_matrix
except Exception as e:
    raise RuntimeError("需要安装 scikit-learn：pip install scikit-learn") from e

# ============================== Providers & Keys ==============================
# 建议实际使用时改成自己的 key，或只用环境变量传入
OAI_KEY   = os.getenv("OPENAI_API_KEY",   "")
OAI_MODEL = os.getenv("OPENAI_MODEL",     "gpt-4o-mini")
OAI_BASE  = os.getenv("OPENAI_BASE",      "https://api.openai.com/v1")

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL",   "deepseek-chat")
DEEPSEEK_BASE  = os.getenv("DEEPSEEK_BASE",    "https://api.deepseek.com/v1")

GROQ_KEY   = os.getenv("GROQ_API_KEY",   "")
GROQ_MODEL = os.getenv("GROQ_MODEL",     "llama3-8b-8192")

XAI_KEY   = os.getenv("XAI_API_KEY",   "")
XAI_MODEL = os.getenv("XAI_MODEL",     "grok-2-latest")
XAI_BASE  = os.getenv("XAI_BASE",      "https://api.x.ai/v1")

# ============================== 全局配置 ==============================
SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Answer directly.\n"
    "2) Do NOT ask the user for more info.\n"
    "3) No templates or meta-commentary.\n"
    "4) Be specific, verifiable and concise.\n"
    "5) If info is missing, make the minimum reasonable assumption and state it.\n"
)

CFG: Dict[str, Any] = {
    "LLM_TEMPERATURE": 0.0,        # 固定为 0，避免随机波动
    "LLM_MAX_TOKENS": 900,         # 容纳≤500字分析
    "TIMEOUT_S": 60,
    "ANSWER_MAX_CHARS": 500,
    "SEED": 42,
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])

LABELS_4 = ["A", "B", "C", "D"]

# 10 个学科，默认会用前 num_subjects 个
SUBJECTS_10 = [
    "abstract_algebra",
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_mathematics",
    "computer_security",
    "econometrics",
    "jurisprudence",
    "machine_learning",
    "moral_scenarios",
]

PLOT_DIR = "plots_4choice_ema-1113"
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================== 价格（按 100 万 tokens） ==============================
# deepseek：CNY；gpt-4o-mini & xai：USD
PRICE_PER_M = {
    "agent-openai": {
        "currency": "USD",
        "input": 0.15,
        "input_cached": 0.075,
        "output": 0.60,
    },
    "agent-deepseek": {
        "currency": "CNY",
        "input": 0.2,        # 缓存未命中
        "input_cached": 0.2, # 缓存命中
        "output": 0.2,
    },
    "agent-xai": {
        "currency": "USD",
        "input": 2.0,
        "input_cached": None,
        "output": 10.0,
    },
}

# ============================== EMA 配置 & 工具 ==============================
EMA_STATE_PATH = "ema_state_mmlu.json"
EMA_AGENT_NAMES = ["agent-openai", "agent-deepseek", "agent-xai"]

EMA_ALPHA_QUALITY   = 0.2
EMA_ALPHA_LATENCY   = 0.2
EMA_ALPHA_COST      = 0.2
EMA_ALPHA_STABILITY = 0.2

LETTER4_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def normalize_to_letter_4(text: str) -> str:
    if not text:
        return "?"
    t = (text or "").strip()
    m = LETTER4_RE.search(t)
    if m:
        return m.group(1).upper()
    up = t.upper()
    for L in LABELS_4:
        if re.match(rf"^\s*{L}\b", up):
            return L
    return "?"

def clip_text(t: str, n: int = 1000) -> str:
    t = t or ""
    return t if len(t) <= n else t[:n] + " ...[TRUNCATED]"

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def init_empty_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    state: Dict[str, Dict[str, Optional[float]]] = {}
    for name in agent_names:
        state[name] = {
            "ema_quality": None,
            "ema_latency_ms": None,
            "ema_cost": None,
            "ema_stability": None,
        }
    return state

def load_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not os.path.exists(EMA_STATE_PATH):
        state = init_empty_ema_state(agent_names)
        with open(EMA_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[EMA] no existing file, created new: {EMA_STATE_PATH}")
        return state

    try:
        with open(EMA_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        # 补齐缺失 agent 或缺失字段
        if not isinstance(state, dict):
            raise ValueError("bad json format")
    except Exception as e:
        print(f"[EMA] load error ({e}), re-init state.")
        state = {}

    for name in agent_names:
        if name not in state or not isinstance(state[name], dict):
            state[name] = {}
        st = state[name]
        for k in ["ema_quality", "ema_latency_ms", "ema_cost", "ema_stability"]:
            if k not in st:
                st[k] = None
    # 清理多余 agent 也无所谓，可以保留
    return state

def save_ema_state(state: Dict[str, Dict[str, Optional[float]]]):
    with open(EMA_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"[EMA] state saved to {EMA_STATE_PATH}")

def ema_update(old: Optional[float], x: Optional[float], alpha: float) -> Optional[float]:
    if x is None:
        return old
    if old is None:
        return float(x)
    return float(alpha * x + (1.0 - alpha) * old)

def calc_call_cost(agent_name: str,
                   tokens_in: Optional[int],
                   tokens_out: Optional[int]) -> Optional[float]:
    if tokens_in is None and tokens_out is None:
        return None
    cfg = PRICE_PER_M.get(agent_name)
    if cfg is None:
        return None
    ti = tokens_in or 0
    to = tokens_out or 0
    inp_price  = cfg.get("input")
    out_price  = cfg.get("output")
    if inp_price is None and out_price is None:
        return None
    # 简单估算：prompt 用 input 价，completion 用 output 价（若无 output 则也用 input）
    if inp_price is None:
        inp_price = 0.0
    if out_price is None:
        out_price = inp_price
    cost = (ti * inp_price + to * out_price) / 1e6
    return float(cost)

def update_agent_ema(ema_state: Dict[str, Dict[str, Optional[float]]],
                     agent_name: str,
                     correct: Optional[bool],
                     latency_ms: Optional[float],
                     cost: Optional[float],
                     stability_ok: Optional[bool]):
    if agent_name not in ema_state:
        # 不认识的 agent 就略过
        return
    st = ema_state[agent_name]
    # quality：用 1/0
    if correct is not None:
        q_obs = 1.0 if correct else 0.0
        st["ema_quality"] = ema_update(st.get("ema_quality"), q_obs, EMA_ALPHA_QUALITY)
    # latency
    if latency_ms is not None:
        st["ema_latency_ms"] = ema_update(st.get("ema_latency_ms"), float(latency_ms), EMA_ALPHA_LATENCY)
    # cost
    if cost is not None:
        st["ema_cost"] = ema_update(st.get("ema_cost"), float(cost), EMA_ALPHA_COST)
    # stability: 成功=1 / 失败=0
    if stability_ok is not None:
        s_obs = 1.0 if stability_ok else 0.0
        st["ema_stability"] = ema_update(st.get("ema_stability"), s_obs, EMA_ALPHA_STABILITY)

# VOTE：三模型多数表决（A–D），打平优先 OPENAI > DEEPSEEK > XAI
def vote_4(open_ans: str, deep_ans: str, xai_ans: str) -> str:
    import collections
    votes = {
        "OPENAI":   open_ans,
        "DEEPSEEK": deep_ans,
        "XAI":      xai_ans,
    }
    letters = {k: v for k, v in votes.items() if v in LABELS_4}
    if not letters:
        return "?"
    counter = collections.Counter(letters.values())
    letter, cnt = counter.most_common(1)[0]
    if cnt >= 2:
        return letter
    # 三家都不一样时按优先级
    for name in ["OPENAI", "DEEPSEEK", "XAI"]:
        cand = votes.get(name, "?")
        if cand in LABELS_4:
            return cand
    return "?"

# ============================== Agent 抽象 & HTTP 封装 ==============================
@runtime_checkable
class Agent(Protocol):
    name: str
    kind: str
    def infer(self, req: "OrchestrateRequest") -> "AgentResult": ...

@dataclass
class AgentResult:
    result_text: str
    latency_ms: int
    http_status: int
    error: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None

def print_agent_result(prefix: str, res: AgentResult):
    print(
        f"{prefix} -> (latency={res.latency_ms}ms, http={res.http_status}, "
        f"tok_in={res.tokens_in}, tok_out={res.tokens_out})"
    )
    print("TEXT:")
    print(clip_text(res.result_text))
    print("-" * 80)

class BaseAgent:
    def __init__(self, name: str, kind: str = "llm"):
        self.name = name
        self.kind = kind

    def request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], int]:
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
        latency_ms = int((time.time() - t0) * 1000)
        data = resp.json() if resp.content else {}
        return resp.status_code, data, latency_ms

    def infer_fail(self, req: "OrchestrateRequest", err: str) -> AgentResult:
        text = f"[{self.name} ERROR] {err}\nEcho: {req.query[:200]}"
        return AgentResult(text, latency_ms=1200, http_status=500, error=err,
                           tokens_in=None, tokens_out=None)

class OpenAICompatAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        super().__init__(name, kind="llm")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                return AgentResult(text, latency_ms=100, http_status=200,
                                   tokens_in=None, tokens_out=None)

            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, latency_ms = self.request(url, headers, payload)
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                text = json.dumps(data)[:2000]
            usage = data.get("usage", {})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            res = AgentResult(text, latency_ms=latency_ms, http_status=code,
                              tokens_in=ti, tokens_out=to)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}")
            print_agent_result(f"CALL {self.name} (ERROR)", res)
            return res

class XaiGrokAgent(BaseAgent):
    def __init__(self, name: str, api_key: str, model: str, base_url: str):
        super().__init__(name, kind="llm")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                return AgentResult(text, latency_ms=120, http_status=200,
                                   tokens_in=None, tokens_out=None)
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, latency_ms = self.request(url, headers, payload)
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                text = json.dumps(data)[:2000]
            usage = data.get("usage", {})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            res = AgentResult(text, latency_ms=latency_ms, http_status=code,
                              tokens_in=ti, tokens_out=to)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}")
            print_agent_result(f"CALL {self.name} (ERROR)", res)
            return res

class EchoAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent-echo", kind="local")

    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        text = f"[agent-echo] Echo: {req.query[:200]}"
        res = AgentResult(text, latency_ms=50, http_status=200,
                          tokens_in=None, tokens_out=None)
        print_agent_result("CALL agent-echo", res)
        return res

# ============================== 任务与 Orchestrator ==============================
@dataclass
class OrchestrateRequest:
    query: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestrateOutcome:
    chosen_agent: str
    final_text: str
    total_latency_ms: int

class Orchestrator:
    def __init__(self, agents: List[Agent],
                 ema_state: Dict[str, Dict[str, Optional[float]]]):
        self.agents = [a for a in agents if getattr(a, "kind", "") in ("llm", "local")]
        self.ema_state = ema_state

    def _score_agent(self, name: str) -> float:
        """
        用 ema_quality / ema_latency / ema_cost / ema_stability 打一个简单分数：
          score = q + 0.1*s - 0.3*norm_lat - 0.1*norm_cost
        """
        st = self.ema_state.get(name, {})
        q = st.get("ema_quality")
        lat = st.get("ema_latency_ms")
        cost = st.get("ema_cost")
        stab = st.get("ema_stability")

        if q is None:
            q = 0.5  # 初始偏中性
        if lat is None:
            lat = 2000.0
        if cost is None:
            cost = 0.0
        if stab is None:
            stab = 0.9

        norm_lat = min(lat / 5000.0, 3.0)   # 大约 0~3
        norm_cost = min(cost / 0.002, 3.0)  # 若每题成本 ~ 0.002 则 ≈1

        score = q + 0.1 * stab - 0.3 * norm_lat - 0.1 * norm_cost
        return float(score)

    def _decompose(self, dispatcher: Agent, prompt: str) -> List[str]:
        tpl = (
            "将下面的四选一题拆分为3个互补的分析子问题，覆盖：概念判别、选项排除、推理核验，"
            "每条不超过40字。只输出三行，每行一个子问题，无其它多余内容。\n\n"
            f"{prompt}\n"
        )
        res = dispatcher.infer(OrchestrateRequest(tpl))
        lines = [ln.strip(" -•\t") for ln in res.result_text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[:3]
        return [
            "定义与概念核对：关键术语和定理含义？",
            "逐项排除：每个选项不成立或不优的理由？",
            "一致性核验：将候选答案与题干逐点对照？",
        ]

    def _one_agent_analysis(self, agent: Agent, full_prompt: str, subq: str) -> AgentResult:
        ana_tpl = (
            "请围绕以下四选一多项选择题进行结构化分析，严格控制在500字以内；"
            "按【要点】列出依据与排除逻辑，最后给出“暂定答案(仅A/B/C/D中的一个字母)”。\n"
            f"子问题：{subq}\n\n{full_prompt}\n"
        )
        res = agent.infer(OrchestrateRequest(ana_tpl))
        trimmed = truncate_chars(res.result_text, CFG["ANSWER_MAX_CHARS"])
        return AgentResult(
            trimmed,
            latency_ms=res.latency_ms,
            http_status=res.http_status,
            error=res.error,
            tokens_in=res.tokens_in,
            tokens_out=res.tokens_out,
        )

    def _merge(self, analyses: List[Tuple[str, str]], full_prompt: str) -> str:
        llms = [a for a in self.agents if getattr(a, "kind", "") == "llm"]
        merger = llms[0] if llms else self.agents[0]

        bullet = "\n\n".join([f"[{name}] {txt}" for name, txt in analyses])
        merge_tpl = (
            "你将看到三名分析员对同一四选一题的分析（均不超过500字）。"
            "请综合证据，仅输出最终选项字母（A/B/C/D），不得附加解释或其它字符。\n\n"
            f"{full_prompt}\n\n{bullet}\n\n最终答案："
        )
        res = merger.infer(OrchestrateRequest(merge_tpl))
        return normalize_to_letter_4(res.result_text)

    def orchestrator_answer(self, prompt: str) -> Tuple[str, int]:
        """
        用当前 EMA 给每个 LLM 打分，选出 top-3 作为 panel，然后并行分析+merge。
        """
        t0 = time.time()
        llms = [a for a in self.agents if getattr(a, "kind", "") == "llm"]
        if not llms:
            res = self.agents[0].infer(OrchestrateRequest(prompt))
            return normalize_to_letter_4(res.result_text), res.latency_ms

        # dispatcher 就用评分最高的那个
        scores = [(self._score_agent(a.name), a) for a in llms]
        scores.sort(key=lambda x: x[0], reverse=True)
        dispatcher = scores[0][1]

        subqs = self._decompose(dispatcher, prompt)

        # panel: 取分数最高的前3个 llm
        top_k = min(3, len(scores))
        panel = [a for _, a in scores[:top_k]]

        analyses: List[Tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=len(panel)) as ex:
            futs = []
            for ag, sq in zip(panel, subqs):
                futs.append(ex.submit(self._one_agent_analysis, ag, prompt, sq))
            future_to_name = {f: ag.name for f, ag in zip(futs, panel)}
            for fu in as_completed(futs):
                res = fu.result()
                name = future_to_name[fu]
                analyses.append((name, res.result_text))

        final_letter = self._merge(analyses, prompt)
        total_ms = int((time.time() - t0) * 1000)
        return final_letter, total_ms

# ============================== 构建 Agents ==============================
def build_agents() -> List[Agent]:
    agents: List[Agent] = []
    if OAI_KEY:
        agents.append(OpenAICompatAgent("agent-openai", OAI_BASE, OAI_KEY, OAI_MODEL))
    if DEEPSEEK_KEY:
        agents.append(OpenAICompatAgent("agent-deepseek", DEEPSEEK_BASE, DEEPSEEK_KEY, DEEPSEEK_MODEL))
    if XAI_KEY:
        agents.append(XaiGrokAgent("agent-xai", XAI_KEY, XAI_MODEL, XAI_BASE))
    if not agents:
        agents.append(EchoAgent())
    return agents

# ============================== 单模型基线 ==============================
def single_agent_baseline(
    agent_name_prefix: str,
    prompt: str,
    agents: List[Agent],
) -> Tuple[str, int, str, Optional[int], Optional[int], int]:
    """
    返回：
      letter, latency_ms, agent_name, tokens_in, tokens_out, http_status
    """
    chosen: Optional[Agent] = None
    for a in agents:
        if a.name.startswith(agent_name_prefix) and getattr(a, "kind", "") == "llm":
            chosen = a
            break
    if chosen is None:
        for a in agents:
            if getattr(a, "kind", "") == "llm":
                chosen = a
                break
    if chosen is None:
        chosen = agents[0]

    q = (
        "You are given a 4-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        f"{prompt}\n"
    )
    res = chosen.infer(OrchestrateRequest(q))
    letter = normalize_to_letter_4(res.result_text)
    return letter, res.latency_ms, chosen.name, res.tokens_in, res.tokens_out, res.http_status

# ============================== McNemar 检验 ==============================
def mcnemar_test(gold: List[str], pred_a: List[str], pred_b: List[str]) -> Tuple[int, int, float, float]:
    b = c = 0
    for y, pa, pb in zip(gold, pred_a, pred_b):
        if pa == "?" or pb == "?":
            continue
        if pa == y and pb != y:
            b += 1
        elif pa != y and pb == y:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    z = math.sqrt(chi2)
    def Phi(x):
        return 0.5 * (1.0 + math.erf(x / (2 ** 0.5)))
    p = 2 * (1 - Phi(z))
    return b, c, chi2, p

# ============================== EMA 曲线画图 ==============================
def plot_ema_curves(ema_history: Dict[str, Dict[str, List[float]]]):
    """
    ema_history[agent_name]["quality"] = [q1, q2, ...]
    ema_history[agent_name]["latency_ms"] = [l1, l2, ...]
    """
    if not ema_history:
        return
    any_agent = next(iter(ema_history.values()))
    n = len(any_agent["quality"])
    if n == 0:
        return
    x = np.arange(1, n + 1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for name, hist in ema_history.items():
        qs = np.array(hist["quality"], dtype=float)
        ls = np.array(hist["latency_ms"], dtype=float)
        axs[0].plot(x, qs, label=name)
        axs[1].plot(x, ls, label=name)

    axs[0].set_ylabel("EMA Quality")
    axs[0].set_title("EMA Quality over questions")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()

    axs[1].set_ylabel("EMA Latency (ms)")
    axs[1].set_xlabel("Question index")
    axs[1].set_title("EMA Latency over questions")
    axs[1].legend()

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "ema_curves.png")
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved -> {out_path}")

# ============================== 主评测 ==============================
def mmlu_eval_multi_subject(
    subjects: List[str],
    split: str,
    per_subject: int,
    seed: int,
    ema_state: Dict[str, Dict[str, Optional[float]]],
):
    random.seed(seed)
    np.random.seed(seed)

    agents = build_agents()
    orch = Orchestrator(agents, ema_state)

    gold_all: List[str] = []
    orch_all: List[str] = []
    open_all: List[str] = []
    deep_all: List[str] = []
    xai_all: List[str] = []
    vote_all: List[str] = []

    orch_latencies: List[int] = []
    open_latencies: List[int] = []
    deep_latencies: List[int] = []
    xai_latencies: List[int] = []
    vote_latencies: List[int] = []

    per_subj_stats: Dict[str, Dict[str, List[str]]] = {
        subj: {"gold": [], "orch": [], "open": [], "deep": [], "xai": [], "vote": []}
        for subj in subjects
    }

    # EMA 曲线记录：每题结束后记录一次 snapshot
    ema_history: Dict[str, Dict[str, List[float]]] = {
        name: {"quality": [], "latency_ms": []}
        for name in ema_state.keys()
    }

    def record_ema_snapshot():
        for name, st in ema_state.items():
            q = st.get("ema_quality")
            l = st.get("ema_latency_ms")
            if q is None:
                qv = float("nan")
            else:
                qv = float(q)
            if l is None:
                lv = float("nan")
            else:
                lv = float(l)
            ema_history[name]["quality"].append(qv)
            ema_history[name]["latency_ms"].append(lv)

    for subj in subjects:
        print(f"\n[LOAD] cais/mmlu, subject={subj}, split={split}")
        ds = load_dataset("cais/mmlu", subj, split=split)

        valid_idxs = [i for i, ex in enumerate(ds) if len(ex["choices"]) >= 4]
        if len(valid_idxs) > per_subject:
            idxs = np.random.choice(valid_idxs, size=per_subject, replace=False)
        else:
            idxs = valid_idxs

        for j, i in enumerate(idxs, 1):
            ex = ds[int(i)]
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = int(ex["answer"])
            if gold_idx < 0 or gold_idx >= 4:
                continue

            prompt = (
                f"Question: {q}\n"
                f"Options:\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
            )

            orch_ans, lat_ms = orch.orchestrator_answer(prompt)
            orch_ans = normalize_to_letter_4(orch_ans)
            orch_latencies.append(lat_ms)

            # ==== 三个基线 + EMA 更新（基于是否答对） ====
            openai_ans, lat_o, name_o, ti_o, to_o, http_o = single_agent_baseline("agent-openai", prompt, agents)
            deepseek_ans, lat_d, name_d, ti_d, to_d, http_d = single_agent_baseline("agent-deepseek", prompt, agents)
            xai_ans,    lat_x, name_x, ti_x, to_x, http_x   = single_agent_baseline("agent-xai",    prompt, agents)

            openai_ans  = normalize_to_letter_4(openai_ans)
            deepseek_ans= normalize_to_letter_4(deepseek_ans)
            xai_ans     = normalize_to_letter_4(xai_ans)

            open_latencies.append(lat_o)
            deep_latencies.append(lat_d)
            xai_latencies.append(lat_x)

            gold_letter = LABELS_4[gold_idx]

            # cost & stability 估算
            cost_o = calc_call_cost(name_o, ti_o, to_o)
            cost_d = calc_call_cost(name_d, ti_d, to_d)
            cost_x = calc_call_cost(name_x, ti_x, to_x)

            stab_o = (http_o == 200)
            stab_d = (http_d == 200)
            stab_x = (http_x == 200)

            # 这题上三个 agent 的正确情况
            correct_o = (openai_ans == gold_letter) if openai_ans in LABELS_4 else False
            correct_d = (deepseek_ans == gold_letter) if deepseek_ans in LABELS_4 else False
            correct_x = (xai_ans   == gold_letter)   if xai_ans   in LABELS_4 else False

            # 更新 EMA
            update_agent_ema(ema_state, name_o, correct_o, lat_o, cost_o, stab_o)
            update_agent_ema(ema_state, name_d, correct_d, lat_d, cost_d, stab_d)
            update_agent_ema(ema_state, name_x, correct_x, lat_x, cost_x, stab_x)

            # VOTE
            vote_ans = vote_4(openai_ans, deepseek_ans, xai_ans)
            lat_vote = lat_o + lat_d + lat_x
            vote_latencies.append(lat_vote)

            gold_all.append(gold_letter)
            orch_all.append(orch_ans)
            open_all.append(openai_ans)
            deep_all.append(deepseek_ans)
            xai_all.append(xai_ans)
            vote_all.append(vote_ans)

            per_subj_stats[subj]["gold"].append(gold_letter)
            per_subj_stats[subj]["orch"].append(orch_ans)
            per_subj_stats[subj]["open"].append(openai_ans)
            per_subj_stats[subj]["deep"].append(deepseek_ans)
            per_subj_stats[subj]["xai"].append(xai_ans)
            per_subj_stats[subj]["vote"].append(vote_ans)

            print(
                f"[{subj} Q{j}/{len(idxs)}] GOLD={gold_letter} | "
                f"ORCH={orch_ans} | OPENAI={openai_ans} | "
                f"DEEPSEEK={deepseek_ans} | XAI={xai_ans} | VOTE={vote_ans} | lat_ORCH={lat_ms}ms"
            )

            # 记录这题之后的 EMA 快照（画曲线用）
            record_ema_snapshot()

    def acc(ys, ps):
        return sum(1 for y, p in zip(ys, ps) if y == p) / max(1, len(ys))

    acc_orch = acc(gold_all, orch_all)
    acc_open = acc(gold_all, open_all)
    acc_deep = acc(gold_all, deep_all)
    acc_xai  = acc(gold_all, xai_all)
    acc_vote = acc(gold_all, vote_all)

    print("\n[GLOBAL ACC]")
    print(f"  ORCH     = {acc_orch:.3f}")
    print(f"  OPENAI   = {acc_open:.3f}")
    print(f"  DEEPSEEK = {acc_deep:.3f}")
    print(f"  XAI      = {acc_xai:.3f}")
    print(f"  VOTE     = {acc_vote:.3f}")

    # 分学科 ACC
    print("\n[PER-SUBJECT ACC]")
    per_subject_rows = []
    for subj in subjects:
        g = per_subj_stats[subj]["gold"]
        ao   = acc(g, per_subj_stats[subj]["orch"])
        aope = acc(g, per_subj_stats[subj]["open"])
        adeep= acc(g, per_subj_stats[subj]["deep"])
        ax   = acc(g, per_subj_stats[subj]["xai"])
        av   = acc(g, per_subj_stats[subj]["vote"])
        per_subject_rows.append((subj, ao, aope, adeep, ax, av))
        line = (
            f"  {subj:26s} ORCH={ao:.3f}  OPENAI={aope:.3f}  "
            f"DEEPSEEK={adeep:.3f}  XAI={ax:.3f}  VOTE={av:.3f}"
        )
        print(line)

    # ORCH 混淆矩阵
    cm = confusion_matrix(gold_all, orch_all, labels=LABELS_4)
    print("\n[CONFUSION MATRIX ORCH (raw counts)]")
    print("     A  B  C  D")
    for i, row in enumerate(cm):
        label = LABELS_4[i]
        line = f"{label}: " + " ".join(f"{int(x):3d}" for x in row)
        print(line)

    # McNemar：ORCH vs DEEPSEEK
    b, c, chi2, p = mcnemar_test(gold_all, orch_all, deep_all)
    print("\n[MCNEMAR] ORCH vs DEEPSEEK")
    print(f"  b = ORCH correct & DEEPSEEK wrong = {b}")
    print(f"  c = ORCH wrong  & DEEPSEEK correct = {c}")
    print(f"  b+c = {b+c}, chi2 = {chi2:.4f}, p ≈ {p:.6f}")
    if p < 0.05:
        print("  ==> 差异在 95% 置信水平下显著 (p < 0.05)")
    else:
        print("  ==> 差异在 95% 置信水平下不显著")

    # 平均延迟
    avg_lat_orch = mean(orch_latencies)
    avg_lat_open = mean(open_latencies)
    avg_lat_deep = mean(deep_latencies)
    avg_lat_xai  = mean(xai_latencies)
    avg_lat_vote = mean(vote_latencies)

    # ========== 图 1: 全局 ACC 柱状图 ==========
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    names = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    vals  = [acc_orch, acc_open, acc_deep, acc_xai, acc_vote]
    ax.bar(names, vals)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("MMLU (4-choice, multi-subject)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    out_path = os.path.join(PLOT_DIR, "global_acc.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 2: 分学科 ACC 柱状图 ==========
    subj_names = [r[0] for r in per_subject_rows]
    orch_accs = [r[1] for r in per_subject_rows]
    open_accs = [r[2] for r in per_subject_rows]
    deep_accs = [r[3] for r in per_subject_rows]
    xai_accs  = [r[4] for r in per_subject_rows]
    vote_accs = [r[5] for r in per_subject_rows]

    x = np.arange(len(subj_names))
    width = 0.16

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - 2*width, orch_accs, width, label="ORCH")
    ax.bar(x - 1*width, open_accs, width, label="OPENAI")
    ax.bar(x + 0*width, deep_accs, width, label="DEEPSEEK")
    ax.bar(x + 1*width, xai_accs,  width, label="XAI")
    ax.bar(x + 2*width, vote_accs, width, label="VOTE")
    ax.set_xticks(x)
    ax.set_xticklabels(subj_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Subject Accuracy (ORCH vs baselines + VOTE)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "per_subject_acc.png")
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 3: ORCH 混淆矩阵 ==========
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xticks(range(len(LABELS_4)))
    ax.set_yticks(range(len(LABELS_4)))
    ax.set_xticklabels(LABELS_4)
    ax.set_yticklabels(LABELS_4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix (ORCH, A–D)")
    for i in range(len(LABELS_4)):
        for j in range(len(LABELS_4)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(PLOT_DIR, "orch_confusion.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 4: McNemar (ORCH vs DEEPSEEK) ==========
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(["b", "c"], [b, c])
    ax.set_ylabel("Count")
    ax.set_title("McNemar ORCH vs DEEPSEEK")
    text = f"b = {b}\nc = {c}\nchi2 = {chi2:.3f}\np = {p:.4f}"
    if p < 0.05:
        text += "\n(p < 0.05, significant)"
    else:
        text += "\n(p >= 0.05, not significant)"
    ymax = max(b, c) if (b or c) else 1
    ax.text(0.6, ymax, text, ha="left", va="top")
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "mcnemar_orch_vs_deepseek.png")
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 5: 价格 + 平均延迟表格 ==========
    header = [
        "Model",
        "Currency",
        "Input/1M",
        "CachedInput/1M",
        "Output/1M",
        "Avg Latency (ms/question)",
    ]
    rows = []

    # ORCH & VOTE：组合策略，用文字描述价格关系
    rows.append([
        "ORCH (3-agent + merge)",
        "-",
        "≈ 3×(O+D+X)",
        "-",
        "-",
        f"{avg_lat_orch:.0f}",
    ])
    rows.append([
        "VOTE (3-model)",
        "-",
        "≈ O + D + X",
        "-",
        "-",
        f"{avg_lat_vote:.0f}",
    ])

    p_open = PRICE_PER_M.get("agent-openai", {})
    rows.append([
        "OpenAI gpt-4o-mini",
        p_open.get("currency", "-"),
        p_open.get("input", "-"),
        p_open.get("input_cached", "-"),
        p_open.get("output", "-"),
        f"{avg_lat_open:.0f}",
    ])

    p_deep = PRICE_PER_M.get("agent-deepseek", {})
    rows.append([
        "Deepseek-chat",
        p_deep.get("currency", "-"),
        p_deep.get("input", "-"),
        p_deep.get("input_cached", "-"),
        p_deep.get("output", "-"),
        f"{avg_lat_deep:.0f}",
    ])

    p_xai = PRICE_PER_M.get("agent-xai", {})
    rows.append([
        "XAI grok-2-latest",
        p_xai.get("currency", "-"),
        p_xai.get("input", "-"),
        p_xai.get("input_cached", "-"),
        p_xai.get("output", "-"),
        f"{avg_lat_xai:.0f}",
    ])

    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    ax.set_title("Price (per 1M tokens) & Avg Latency", pad=10)
    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "price_latency_table.png")
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 6: EMA 曲线 ==========
    plot_ema_curves(ema_history)

    # ========== 打印每个 agent 的最终 EMA 指标 ==========
    print("\n[EMA STATS PER AGENT]")
    for name, st in ema_state.items():
        print(
            f"  {name:14s} | "
            f"ema_quality={st.get('ema_quality')} | "
            f"ema_latency_ms={st.get('ema_latency_ms')} | "
            f"ema_cost={st.get('ema_cost')} | "
            f"ema_stability={st.get('ema_stability')}"
        )

    # 保存 EMA 状态
    save_ema_state(ema_state)

# ============================== REPL（可选） ==============================
def repl():
    agents = build_agents()
    print("\n[BOOT] Agents:")
    for a in agents:
        print(f"  - {a.name} kind={a.kind}")
    print("\nType your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            q = input(">> ").strip()
        except EOFError:
            break
        if not q or q.lower() == "exit":
            print("bye.")
            break
        a0 = agents[0]
        res = a0.infer(OrchestrateRequest(q))
        print(clip_text(res.result_text, 2000))

# ============================== CLI ==============================
def main():
    ap = argparse.ArgumentParser(
        description="MMLU 4-choice multi-subject: ORCH(EMA-based panel) + 3 baselines + VOTE + EMA plots"
    )
    ap.add_argument("--mode", choices=["eval", "repl"], default="eval",
                    help="eval=多学科评测; repl=单Agent交互")
    ap.add_argument("--per_subject", type=int, default=30,
                    help="每个学科抽取题目数 (default=30)")
    ap.add_argument("--num_subjects", type=int, default=len(SUBJECTS_10),
                    help=f"使用前 N 个学科 (1~{len(SUBJECTS_10)})")
    args = ap.parse_args()

    if args.mode == "repl":
        repl()
    else:
        n_subj = max(1, min(len(SUBJECTS_10), args.num_subjects))
        subjects = SUBJECTS_10[:n_subj]

        # 载入 / 初始化 EMA 状态
        ema_state = load_ema_state(EMA_AGENT_NAMES)

        mmlu_eval_multi_subject(
            subjects,
            split="test",
            per_subject=args.per_subject,
            seed=CFG["SEED"],
            ema_state=ema_state,
        )

if __name__ == "__main__":
    main()

