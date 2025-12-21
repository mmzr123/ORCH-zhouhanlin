# -*- coding: utf-8 -*-
"""
mmlu_router_4choice_5plots_vote.py

- 多学科 MMLU 4 选一评测
- ORCH: 三 Agent 并行分析（≤500字）+ 合并 Agent 只出 A/B/C/D
- 基线: OPENAI / DEEPSEEK / XAI 各自直接选一次
- VOTE: 三基线直接 vote，多数表决
- 评测设置：
  * 10 个学科，每科默认 30 题（可改）
  * 温度 temperature=0.0（避免随机波动）
- 生成 5 张图（在 plots_4choice/ 下）：
  1) global_acc.png                全局 ACC 柱状图（ORCH + 3 基线 + VOTE）
  2) per_subject_acc.png           分学科 ACC 柱状图（ORCH + 3 基线 + VOTE）
  3) orch_confusion.png            ORCH 4×4 混淆矩阵
  4) mcnemar_orch_vs_deepseek.png  McNemar (ORCH vs DEEPSEEK)
  5) price_latency_table.png       价格 + 平均延迟表格（ORCH / VOTE / 三模型）
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
OAI_KEY   = os.getenv("OPENAI_API_KEY",   "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")
OAI_MODEL = os.getenv("OPENAI_MODEL",     "gpt-4o-mini")
OAI_BASE  = os.getenv("OPENAI_BASE",      "https://api.openai.com/v1")

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY", "sk-c181d17907ff4d848eca8225244aa67f")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL",   "deepseek-chat")
DEEPSEEK_BASE  = os.getenv("DEEPSEEK_BASE",    "https://api.deepseek.com/v1")

GROQ_KEY   = os.getenv("GROQ_API_KEY",   "")
GROQ_MODEL = os.getenv("GROQ_MODEL",     "llama3-8b-8192")

XAI_KEY   = os.getenv("XAI_API_KEY",   "xai-iTEf2LYg5Jvd4YIHmSmihv7gQWEv5XkiEKuzRwQ4lqBLwUsl4tndgGSnVsn6DCzBa3gRgTl0Wuuowag9")
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

# 10 个学科，每科默认 30 题
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

PLOT_DIR = "plots_4choice-1109"
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
        "input": 2.0,        # 缓存未命中
        "input_cached": 0.2, # 缓存命中
        "output": None,
    },
    "agent-xai": {
        "currency": "USD",
        "input": 2.0,
        "input_cached": None,
        "output": 10.0,
    },
}

# ============================== 工具函数 ==============================
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
    def __init__(self, agents: List[Agent]):
        self.agents = [a for a in agents if getattr(a, "kind", "") in ("llm", "local")]

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
        t0 = time.time()
        llms = [a for a in self.agents if getattr(a, "kind", "") == "llm"]
        if not llms:
            res = self.agents[0].infer(OrchestrateRequest(prompt))
            return normalize_to_letter_4(res.result_text), res.latency_ms

        dispatcher = llms[0]
        subqs = self._decompose(dispatcher, prompt)

        preferred = ["agent-openai", "agent-deepseek", "agent-xai"]
        panel: List[Agent] = []
        for nm in preferred:
            a = next((x for x in llms if x.name == nm), None)
            if a:
                panel.append(a)
        if len(panel) < 3:
            for a in llms:
                if a not in panel and len(panel) < 3:
                    panel.append(a)
        if not panel:
            panel = llms[:1]

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
) -> Tuple[str, int]:
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
    return letter, res.latency_ms

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

# ============================== 主评测 ==============================
def mmlu_eval_multi_subject(
    subjects: List[str],
    split: str = "test",
    per_subject: int = 30,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    agents = build_agents()
    orch = Orchestrator(agents)

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

            openai_ans, lat_o = single_agent_baseline("agent-openai", prompt, agents)
            deepseek_ans, lat_d = single_agent_baseline("agent-deepseek", prompt, agents)
            xai_ans,    lat_x = single_agent_baseline("agent-xai",    prompt, agents)

            openai_ans = normalize_to_letter_4(openai_ans)
            deepseek_ans = normalize_to_letter_4(deepseek_ans)
            xai_ans = normalize_to_letter_4(xai_ans)

            open_latencies.append(lat_o)
            deep_latencies.append(lat_d)
            xai_latencies.append(lat_x)

            vote_ans = vote_4(openai_ans, deepseek_ans, xai_ans)
            # VOTE 从三个基线得到结果，时延视为三次调用之和
            lat_vote = lat_o + lat_d + lat_x
            vote_latencies.append(lat_vote)

            gold_letter = LABELS_4[gold_idx]

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
    ax.set_title("MMLU (4-choice, 10 subjects)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    out_path = os.path.join(PLOT_DIR, "global_acc.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")
    plt.close(fig)

    # ========== 图 2: 分学科 ACC 柱状图（ORCH + 3 基线 + VOTE） ==========
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
        description="MMLU 4-choice multi-subject: ORCH(3-agent analyses + merge) + 3 baselines + VOTE + 5 plots"
    )
    ap.add_argument("--mode", choices=["eval", "repl"], default="eval",
                    help="eval=多学科评测; repl=单Agent交互")
    ap.add_argument("--per_subject", type=int, default=30,
                    help="每个学科抽取题目数 (default=30)")
    args = ap.parse_args()

    if args.mode == "repl":
        repl()
    else:
        mmlu_eval_multi_subject(
            SUBJECTS_10,
            split="test",
            per_subject=args.per_subject,
            seed=CFG["SEED"],
        )

if __name__ == "__main__":
    main()
