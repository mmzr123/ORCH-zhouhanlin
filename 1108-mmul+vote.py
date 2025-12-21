# -*- coding: utf-8 -*-
"""
mmlu_router_4choice_with_vote.py

- 10 个 MMLU 学科 × 每科 30 题（4 选一）
- ORCH: 多 Agent 结构化分析（≤500 字）+ 合并 Agent 输出 A/B/C/D
- 单模型基线: OPENAI / DEEPSEEK / XAI
- VOTE: 多模型只投票 A/B/C/D，多数表决

输出：
  - 控制台：全局 ACC、分学科 ACC、ORCH 混淆矩阵、McNemar (ORCH vs DEEPSEEK)
  - 图像（plots_4choice/ 下 5 张 png）：
      1) global_acc.png
      2) per_subject_acc.png
      3) orch_confusion.png
      4) mcnemar_orch_vs_deepseek.png
      5) price_latency.png
"""

from __future__ import annotations
import os, json, time, random, math, collections, re, argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== Optional metrics (Prometheus) ==========
try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
except Exception:
    class _Noop:
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): return None
        def set(self, *a, **k): return None
        def observe(self, *a, **k): return None
    def start_http_server(*a, **k): return None
    Counter = Gauge = Histogram = Summary = lambda *a, **k: _Noop()

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

import requests

PLOTS_DIR = "plots_4choice3"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================== 内联 API-KEY（若环境变量未设才使用） ==============================
# ⚠️ 这里不要放真实 key，自行替换或用环境变量。
INLINE_KEYS = {
    "DEEPSEEK_API_KEY": "sk-c181d17907ff4d848eca8225244aa67f",
    "GROQ_API_KEY": "",
    "XAI_API_KEY": "xai-iTEf2LYg5Jvd4YIHmSmihv7gQWEv5XkiEKuzRwQ4lqBLwUsl4tndgGSnVsn6DCzBa3gRgTl0Wuuowag9",
}
for k, v in INLINE_KEYS.items():
    if v and not os.getenv(k):
        os.environ[k] = v

# ============================== Providers ==============================
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

# 价格表（每 1M tokens），仅用于图表展示
PRICE = {
    "openai":     {"in": 0.15, "cached_in": 0.075, "out": 0.60, "currency": "USD"},
    "deepseek":   {"in": 2.0,  "cached_in": 0.2,   "out": None, "currency": "CNY"},
    "grok":       {"in": 2.0,  "cached_in": None,  "out": 10.0, "currency": "USD"},
}

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
    "OPENAI_COMPAT_PROFILES": [
        {"name":"openai","base_url":OAI_BASE,"api_key":OAI_KEY,"model":OAI_MODEL,
         "enabled":bool(OAI_KEY), "init_speed_ms":800, "init_quality":+0.02, "init_reliab":0.90},
        {"name":"deepseek","base_url":DEEPSEEK_BASE,"api_key":DEEPSEEK_KEY,"model":DEEPSEEK_MODEL,
         "enabled":bool(DEEPSEEK_KEY), "init_speed_ms":1000, "init_quality":+0.01, "init_reliab":0.88},

    ],
    "GROQ": {"api_key":GROQ_KEY, "model":GROQ_MODEL,
             "enabled":bool(GROQ_KEY), "init_speed_ms":600, "init_quality":+0.02,"init_reliab":0.86},
    "XAI":  {"api_key":XAI_KEY,  "model":XAI_MODEL,  "base_url":XAI_BASE,
             "enabled":bool(XAI_KEY), "init_speed_ms":700, "init_quality":+0.02,"init_reliab":0.86},

    "LLM_TEMPERATURE": 0.0,      # 固定为 0，避免采样波动
    "LLM_MAX_TOKENS":  900,      # 容纳≤500字分析
    "ANSWER_MAX_CHARS": 500,
    "TIMEOUT_S":       60,

    # EMA-only（无 trust）
    "EMA_ALPHA":    0.15,
    "BASE_LAT_MS":  1200,
    "BASE_RELIAB":  0.80,
    "BASE_QUALITY": 0.02,

    # 组合得分权重
    "W_RELIAB": 0.35,
    "W_QUAL":   0.30,
    "W_SPEED": -0.00012,
    "W_LEN":    0.00002,
    "W_COST":  -0.08,
    "LEN_CAP":  8000,

    "BANDIT": "ucb",
    "SEED":   42,

    "STATE_PATH": "agent_metrics.json",
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "0")),
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])

# =============== Utils ===============
LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def normalize_to_letter(text: str) -> str:
    if not text:
        return "?"
    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    t = (text or "").strip().upper()
    for L in ["A", "B", "C", "D"]:
        if re.match(rf"^\s*{L}\b", t):
            return L
    return "?"

def clip_text(t: str, n: int = 1000) -> str:
    t = t or ""
    return t if len(t) <= n else t[:n] + " ...[TRUNCATED]"

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def approx_tokens_from_text(text: str) -> int:
    return max(1, int(len(text) / 4))

# =============== Metrics (EMA 等) ===============
METRIC_AGENT_CALLS = Counter("agent_calls_total", "Total calls per agent", ["agent"])
METRIC_AGENT_ACCEPT = Counter("agent_accept_total", "Accepted outputs per agent", ["agent"])
METRIC_AGENT_LAT_MS  = Histogram("agent_latency_ms", "Latency per agent (ms)", ["agent"])
METRIC_AGENT_COST_K  = Summary("agent_cost_per_ktok", "Cost per 1k tokens", ["agent"])
METRIC_AGENT_EMA_LAT = Gauge("agent_ema_latency_ms", "EMA latency per agent (ms)", ["agent"])
METRIC_AGENT_EMA_REL = Gauge("agent_ema_reliability", "EMA reliability per agent", ["agent"])
METRIC_AGENT_EMA_QUAL= Gauge("agent_ema_quality", "EMA quality per agent", ["agent"])
METRIC_AGENT_EMA_LEN = Gauge("agent_ema_len", "EMA text length per agent", ["agent"])

class MetricsStore:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        self.global_calls = 0
        self.load()
    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.data = loaded.get("agents", {})
                self.global_calls = loaded.get("global_calls", 0)
        except Exception:
            self.data = {}
    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"agents": self.data, "global_calls": self.global_calls},
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] failed to save metrics: {e}")
    def ensure(self, name: str, init_speed_ms: int, init_qual: float, init_rel: float):
        if name not in self.data:
            self.data[name] = {
                "ema_latency_ms": init_speed_ms,
                "ema_reliability": init_rel,
                "ema_quality": init_qual,
                "ema_cost_per_ktok": 0.0,
                "ema_len": 400.0,
                "calls": 0, "accepted": 0,
                "bandit": {"alpha": 1.0, "beta": 1.0},
                "ucb": {"n": 0, "sum_reward": 0.0},
            }
    def update_after_call(self, name: str, latency_ms: int, ok_http: bool, text_len: int,
                          est_quality_boost: float, cost_per_ktok: float):
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        d = self.data[name]
        a = CFG["EMA_ALPHA"]
        d["ema_latency_ms"] = (1 - a) * d["ema_latency_ms"] + a * latency_ms
        d["ema_reliability"] = (1 - a) * d["ema_reliability"] + a * (1.0 if ok_http else 0.0)
        d["ema_len"] = (1 - a) * d["ema_len"] + a * min(text_len, CFG["LEN_CAP"])
        d["ema_quality"] = (1 - a) * d["ema_quality"] + a * est_quality_boost
        d["ema_cost_per_ktok"] = (1 - a) * d["ema_cost_per_ktok"] + a * max(0.0, cost_per_ktok)
        d["calls"] += 1
        self.global_calls += 1
        METRIC_AGENT_EMA_LAT.labels(agent=name).set(d["ema_latency_ms"])
        METRIC_AGENT_EMA_REL.labels(agent=name).set(d["ema_reliability"])
        METRIC_AGENT_EMA_QUAL.labels(agent=name).set(d["ema_quality"])
        METRIC_AGENT_EMA_LEN.labels(agent=name).set(d["ema_len"])
    def mark_accept(self, name: str, reward: float = 1.0):
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        d = self.data[name]
        d["accepted"] += 1
        d["bandit"]["alpha"] += reward
        d["ucb"]["n"] += 1
        d["ucb"]["sum_reward"] += reward
        METRIC_AGENT_ACCEPT.labels(agent=name).inc()
        self.save()
    def mark_reject(self, name: str, reward: float = 0.0):
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        d = self.data[name]
        d["bandit"]["beta"] += (1.0 - reward)
        d["ucb"]["n"] += 1
        d["ucb"]["sum_reward"] += reward
        self.save()
    def snapshot(self, name: str) -> Dict[str, Any]:
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        return dict(self.data[name])

METRICS = MetricsStore(CFG["STATE_PATH"])

def composite_prob(name: str, text: str, http_status: Optional[int]) -> float:
    snap = METRICS.snapshot(name)
    W = CFG
    rel = snap["ema_reliability"]
    qual = snap["ema_quality"]
    lat = snap["ema_latency_ms"]
    L   = min(len(text or ""), CFG["LEN_CAP"])
    cost= snap["ema_cost_per_ktok"]
    p = (W["W_RELIAB"]*rel +
         W["W_QUAL"]*(0.70 + qual) +
         W["W_SPEED"]*lat +
         W["W_LEN"]*L +
         W["W_COST"]*cost +
         0.10)
    return max(0.05, min(0.98, p))

def bandit_score(name: str, t: int) -> float:
    snap = METRICS.snapshot(name)
    if CFG["BANDIT"] == "thompson":
        alpha = snap["bandit"]["alpha"]
        beta  = snap["bandit"]["beta"]
        return random.betavariate(alpha, beta)
    n = snap["ucb"]["n"]
    s = snap["ucb"]["sum_reward"]
    if n == 0:
        return 1e9
    mean = s / n
    c = 1.2
    return mean + c * math.sqrt(math.log(max(t, 2)) / n)

# =============== Agents ===============
@runtime_checkable
class Agent(Protocol):
    name: str
    kind: str
    def infer(self, req: "OrchestrateRequest") -> "AgentResult": ...

@dataclass
class AgentResult:
    result_text: str
    prob: float
    reliability: float
    latency_ms: int
    http_status: Optional[int] = None
    error: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_per_ktok: float = 0.0

def print_agent_result(prefix: str, res: AgentResult):
    print(f"{prefix} -> (p={res.prob:.3f}, reliab={res.reliability:.2f}, "
          f"latency={res.latency_ms}ms, http={res.http_status}, "
          f"cost/1k=${res.cost_per_ktok:.3f})")
    print("TEXT:")
    print(clip_text(res.result_text))
    print("-"*86)

class BaseAgent:
    def __init__(self, name: str, kind: str = "llm"):
        self.name = name
        self.kind = kind
        METRICS.ensure(self.name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
    def request(self, url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Tuple[int, Dict[str,Any], float]:
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
        return resp.status_code, (resp.json() if resp.content else {}), (time.time()-t0)*1000.0
    def _calc_cost(self, provider: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> float:
        # 这里只做粗略估计，用 PRICE 表的 in/out 美元；ORCH 总体表格用的是手填价格
        if provider not in PRICE:
            return 0.0
        ti = tokens_in or 0
        to = tokens_out or 0
        p  = PRICE[provider]
        pin = p["in"] or 0.0
        pout= p["out"] or 0.0
        return (pin*ti + pout*to) / 1_000_000.0
    def _after(self, http: int, text: str, latency_ms: int,
               est_quality_boost: float, provider: str,
               tokens_in: Optional[int], tokens_out: Optional[int]) -> float:
        if tokens_out is None:
            tokens_out = approx_tokens_from_text(text or "")
        cost_k = self._calc_cost(provider, tokens_in, tokens_out)
        METRICS.update_after_call(self.name, latency_ms, http==200, len(text or ""), est_quality_boost, cost_k)
        METRIC_AGENT_CALLS.labels(agent=self.name).inc()
        METRIC_AGENT_LAT_MS.labels(agent=self.name).observe(max(1, latency_ms))
        METRIC_AGENT_COST_K.labels(agent=self.name).observe(max(0.0, cost_k))
        return cost_k
    def infer_fail(self, req: "OrchestrateRequest", err: str, provider: str, latency_ms: int=1200) -> AgentResult:
        cost_k = self._after(500, err, latency_ms, -0.03, provider, None, approx_tokens_from_text(err))
        p = composite_prob(self.name, err, 500)
        return AgentResult(f"[{self.name} ERROR] {err}\nEcho: {req.query[:200]}",
                           p, METRICS.snapshot(self.name)["ema_reliability"],
                           int(latency_ms), 500, err, 0,
                           approx_tokens_from_text(err), cost_k)

class OpenAICompatAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, provider_key: str):
        super().__init__(name, kind="llm")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider_key = provider_key
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                cost_k = self._after(200, text, 1200, -0.02, "openai", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200)
                rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 1200, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res)
                return res
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type":"application/json"}
            payload = {
                "model": self.model,
                "messages":[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content":req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() \
                   or json.dumps(data)[:2000]
            usage = data.get("usage",{})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.00, self.provider_key, ti, to)
            p = composite_prob(self.name, text, code)
            rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", self.provider_key)
            print_agent_result(f"CALL {self.name} (ERROR)", res)
            return res

class GroqAgent(BaseAgent):
    def __init__(self, name: str, api_key: str, model: str):
        super().__init__(name, kind="llm")
        self.api_key = api_key
        self.model = model
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                cost_k = self._after(200, text, 800, -0.02, "openai", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200)
                rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 800, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res)
                return res
            url = f"{GROQ_BASE}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type":"application/json"}
            payload = {
                "model": self.model,
                "messages":[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content":req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() \
                   or json.dumps(data)[:2000]
            usage = data.get("usage",{})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.01, "openai", ti, to)
            p = composite_prob(self.name, text, code)
            rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", "openai")
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
                cost_k = self._after(200, text, 800, -0.02, "grok", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200)
                rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 800, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res)
                return res
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type":"application/json"}
            payload = {
                "model": self.model,
                "messages":[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content":req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() \
                   or json.dumps(data)[:2000]
            usage = data.get("usage",{})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.02, "grok", ti, to)
            p = composite_prob(self.name, text, code)
            rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", "grok")
            print_agent_result(f"CALL {self.name} (ERROR)", res)
            return res

class RuleAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent-rule", kind="rule")
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        text = f"[agent-rule] (rule) Please follow policy. Echo: {req.query[:200]}"
        cost_k = self._after(200, text, 100, -0.02, "openai", None, approx_tokens_from_text(text))
        p = composite_prob(self.name, text, 200)
        res = AgentResult(text, p, METRICS.snapshot(self.name)["ema_reliability"],
                          100, 200, None, 0, approx_tokens_from_text(text), cost_k)
        print_agent_result("CALL agent-rule", res)
        return res

class EchoAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent-echo", kind="local")
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        text = f"[agent-echo] Echo: {req.query[:200]}"
        cost_k = self._after(200, text, 200, -0.03, "openai", None, approx_tokens_from_text(text))
        p = composite_prob(self.name, text, 200)
        res = AgentResult(text, p, METRICS.snapshot(self.name)["ema_reliability"],
                          200, 200, None, 0, approx_tokens_from_text(text), cost_k)
        print_agent_result("CALL agent-echo", res)
        return res

# =============== Task/Orchestrator ===============
@dataclass
class OrchestrateRequest:
    query: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestrateOutcome:
    chosen_agent: str
    final_text: str
    final_prob: float
    attempts: int
    total_latency_ms: int

def choose_main_ranked(agents: List[Agent], query: str, t_round: int) -> List[Tuple[int,float,float,float,Dict[str,Any]]]:
    llms = [i for i,a in enumerate(agents) if getattr(a,"kind","")=="llm"]
    scored = []
    for idx in llms:
        a = agents[idx]
        b = bandit_score(a.name, t_round)
        dyn = composite_prob(a.name, "probe", 200)
        snap = METRICS.snapshot(a.name)
        mix = 0.6*b + 0.4*dyn
        scored.append((idx, mix, b, dyn, snap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def pick_panel(agents: List[Agent], main_idx: int, k:int=3) -> List[int]:
    cands = [i for i,a in enumerate(agents) if getattr(a,"kind","")=="llm" and i!=main_idx]
    random.shuffle(cands)
    panel = [main_idx] + cands[:max(0, k-1)]
    return panel[:k]

class Orchestrator:
    """多 Agent 结构化分析（≤500 字）+ 合并 Agent 输出 A/B/C/D"""

    def __init__(self, agents: List[Agent]):
        self.agents = [a for a in agents if getattr(a, "kind","") in ("llm","rule","local")]
        self.t_round = 0

    def _one_agent_analysis(self, agent: Agent, full_prompt: str, subq: str) -> AgentResult:
        ana_tpl = (
            "请围绕该四选一多项选择题进行'结构化分析'，严格控制在500字以内；"
            "按【要点】列出依据与排除逻辑，最后给出'暂定答案(A/B/C/D)'。\n"
            "子问题：{subq}\n\n{prompt}\n"
        )
        q = ana_tpl.format(subq=subq, prompt=full_prompt)
        res = agent.infer(OrchestrateRequest(q))
        trimmed = truncate_chars(res.result_text, CFG["ANSWER_MAX_CHARS"])
        return AgentResult(trimmed, res.prob, res.reliability, res.latency_ms,
                           res.http_status, res.error, res.tokens_in,
                           res.tokens_out, res.cost_per_ktok)

    def _decompose(self, dispatcher: Agent, prompt: str) -> List[str]:
        tpl = (
            "将下面的四选一题拆分为3个互补的分析子问题，覆盖：概念判别、选项排除、推理核验，每条不超过40字。\n"
            "只输出三行，每行一个子问题，无其它多余内容。\n\n"
            f"{prompt}\n"
        )
        res = dispatcher.infer(OrchestrateRequest(tpl))
        lines = [ln.strip(" -•\t") for ln in res.result_text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[:3]
        return [
            "定义与关键概念是否正确？",
            "逐项排除各选项的错误及矛盾点？",
            "最终候选与题干是否完全一致？",
        ]

    def _merge(self, merger: Agent, analyses: List[Tuple[str,str]], full_prompt:str) -> str:
        bullet = "\n\n".join([f"[{name}] {txt}" for name,txt in analyses])
        merge_tpl = (
            "你将看到三名分析员对同一四选一题的分析（均≤500字）。"
            "请综合证据，仅输出最终选项字母（A/B/C/D），不得附加解释或其它字符。\n\n"
            f"{full_prompt}\n\n{bullet}\n\n最终答案："
        )
        res = merger.infer(OrchestrateRequest(merge_tpl))
        return normalize_to_letter(res.result_text)

    def answer(self, prompt: str) -> Tuple[str,int]:
        t0 = time.time()
        self.t_round += 1
        # 选择 dispatcher / merger / panel：用 EMA+UCB 的 router
        ranked = choose_main_ranked(self.agents, prompt, self.t_round)
        if not ranked:
            # 兜底：随便用一个 LLM 或 echo
            llms = [a for a in self.agents if getattr(a,"kind","")=="llm"]
            dispatcher = merger = llms[0] if llms else self.agents[0]
            panel = [dispatcher]
        else:
            main_idx = ranked[0][0]
            dispatcher = self.agents[main_idx]
            merger = self.agents[main_idx]
            panel_idx = pick_panel(self.agents, main_idx, k=3)
            panel = [self.agents[i] for i in panel_idx]

        subqs = self._decompose(dispatcher, prompt)

        analyses: List[Tuple[str,str]] = []
        with ThreadPoolExecutor(max_workers=len(panel)) as ex:
            futs = []
            for ag, sq in zip(panel, subqs):
                futs.append(ex.submit(self._one_agent_analysis, ag, prompt, sq))
            for i, fu in enumerate(as_completed(futs)):
                res = fu.result()
                # as_completed 顺序不一定与 panel 相同，这里简单按完成顺序记录
                analyses.append((f"agent-{i}", res.result_text))

        letter = self._merge(merger, analyses, prompt)
        total_ms = int((time.time()-t0)*1000)
        return letter, total_ms

# =============== Build agents ===============
def build_agents() -> List[Agent]:
    agents: List[Agent] = []
    for p in CFG["OPENAI_COMPAT_PROFILES"]:
        if p["enabled"]:
            provider_key = p["name"] if p["name"] in PRICE else "openai"
            agents.append(OpenAICompatAgent(f"agent-{p['name']}",
                                            p["base_url"], p["api_key"],
                                            p["model"], provider_key))
    if CFG["GROQ"]["enabled"]:
        agents.append(GroqAgent("agent-groq", CFG["GROQ"]["api_key"], CFG["GROQ"]["model"]))
    if CFG["XAI"]["enabled"]:
        agents.append(XaiGrokAgent("agent-xai", CFG["XAI"]["api_key"],
                                   CFG["XAI"]["model"], CFG["XAI"]["base_url"]))
    agents.append(RuleAgent())
    if not any(getattr(a,"kind","")=="llm" for a in agents):
        agents.insert(0, EchoAgent())
    return agents

# =============== Baselines ===============
def single_agent_baseline(agent_name_prefix: str, prompt: str, agents: List[Agent]) -> Tuple[str,int]:
    chosen = None
    for a in agents:
        if a.name.startswith(agent_name_prefix) and getattr(a,"kind","")=="llm":
            chosen = a
            break
    if chosen is None:
        for a in agents:
            if getattr(a,"kind","") == "llm":
                chosen = a
                break
    if chosen is None:
        return "?", 0
    q = (
        "You are given a 4-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        f"{prompt}\n"
    )
    res = chosen.infer(OrchestrateRequest(q))
    return normalize_to_letter(res.result_text), res.latency_ms

def multi_agent_vote_abcd(panel_agents: List[Agent], prompt: str) -> Tuple[str, Dict[str,str]]:
    """
    多模型只投票 A/B/C/D，多数表决
    """
    ASK_TEMPLATE = (
        "You are given a 4-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        "{q}\n"
    )
    q = ASK_TEMPLATE.format(q=prompt)
    votes: Dict[str,str] = {}
    for ag in panel_agents:
        res = ag.infer(OrchestrateRequest(q))
        letter = normalize_to_letter(res.result_text)
        votes[ag.name] = letter

    counter = collections.Counter(votes.values())
    if not counter:
        return "?", votes
    top_letter, top_cnt = counter.most_common(1)[0]

    # 极端情况：三人各不相同
    if len(counter) == 3 and top_cnt == 1:
        preferred_order = ["agent-openai", "agent-deepseek", "agent-xai"]
        for name in preferred_order:
            for ag_name, L in votes.items():
                if ag_name.startswith(name):
                    return L, votes
        return top_letter, votes

    return top_letter, votes

# =============== 评测主逻辑 ===============
SUBJECTS = [
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

def sample_indices(ds, n: int) -> List[int]:
    idxs = [i for i, ex in enumerate(ds) if len(ex["choices"]) >= 4]
    random.shuffle(idxs)
    return idxs[:min(n, len(idxs))]

def mmlu_eval_allsubjects(per_subject_n: int = 30, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    agents = build_agents()
    orch = Orchestrator(agents)

    # 构造投票 panel（openai + deepseek + xai 优先）
    llm_agents = [a for a in agents if getattr(a,"kind","")=="llm"]
    pref_names = ["agent-openai", "agent-deepseek", "agent-xai"]
    vote_panel: List[Agent] = []
    for nm in pref_names:
        a = next((x for x in llm_agents if x.name == nm), None)
        if a:
            vote_panel.append(a)
    if len(vote_panel) < 3:
        for a in llm_agents:
            if a not in vote_panel and len(vote_panel) < 3:
                vote_panel.append(a)
    if not vote_panel:
        vote_panel = llm_agents[:1] or agents[:1]

    gold_all: List[str] = []
    orch_all: List[str] = []
    open_all: List[str] = []
    deep_all: List[str] = []
    xai_all:  List[str] = []
    vote_all: List[str] = []

    # latency
    lat_orch = lat_open = lat_deep = lat_xai = 0.0

    # per-subject
    per_subject_stats: Dict[str, Dict[str,List[str]]] = {
        s: {"gold":[], "orch":[], "open":[], "deep":[], "xai":[], "vote":[]} for s in SUBJECTS
    }

    total_q = 0

    for subj in SUBJECTS:
        print(f"\n[LOAD] cais/mmlu, subject={subj}")
        ds = load_dataset("cais/mmlu", subj, split="test")
        idxs = sample_indices(ds, per_subject_n)

        for local_i, i in enumerate(idxs, 1):
            ex = ds[i]
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = int(ex["answer"])
            if len(choices) < 4:
                continue
            prompt = (
                f"Question: {q}\n"
                f"Options:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            )

            orch_ans, lat_ms = orch.answer(prompt)
            open_ans,  lat_o  = single_agent_baseline("agent-openai", prompt, agents)
            deep_ans,  lat_d  = single_agent_baseline("agent-deepseek", prompt, agents)
            xai_ans,   lat_x  = single_agent_baseline("agent-xai",     prompt, agents)
            vote_ans,  _votes = multi_agent_vote_abcd(vote_panel, prompt)

            gold_letter = ["A","B","C","D"][gold_idx]
            gold_all.append(gold_letter)
            orch_all.append(orch_ans)
            open_all.append(open_ans)
            deep_all.append(deep_ans)
            xai_all.append(xai_ans)
            vote_all.append(vote_ans)

            per_subject_stats[subj]["gold"].append(gold_letter)
            per_subject_stats[subj]["orch"].append(orch_ans)
            per_subject_stats[subj]["open"].append(open_ans)
            per_subject_stats[subj]["deep"].append(deep_ans)
            per_subject_stats[subj]["xai"].append(xai_ans)
            per_subject_stats[subj]["vote"].append(vote_ans)

            lat_orch += lat_ms
            lat_open += lat_o
            lat_deep += lat_d
            lat_xai  += lat_x

            total_q += 1
            print(f"[{subj} Q{local_i}/{len(idxs)}] GOLD={gold_letter} | "
                  f"ORCH={orch_ans} | OPENAI={open_ans} | "
                  f"DEEPSEEK={deep_ans} | XAI={xai_ans} | VOTE={vote_ans} | "
                  f"lat_ORCH={lat_ms}ms")

    def acc(ys, ps):
        return sum(1 for y,p in zip(ys,ps) if y==p) / max(1, len(ys))

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

    print("\n[PER-SUBJECT ACC]")
    per_subject_acc: Dict[str, Dict[str,float]] = {}
    for subj in SUBJECTS:
        st = per_subject_stats[subj]
        acc_s_orch = acc(st["gold"], st["orch"])
        acc_s_open = acc(st["gold"], st["open"])
        acc_s_deep = acc(st["gold"], st["deep"])
        acc_s_xai  = acc(st["gold"], st["xai"])
        acc_s_vote = acc(st["gold"], st["vote"])
        per_subject_acc[subj] = {
            "orch": acc_s_orch,
            "open": acc_s_open,
            "deep": acc_s_deep,
            "xai":  acc_s_xai,
            "vote": acc_s_vote,
        }
        print(f"  {subj:26s} ORCH={acc_s_orch:.3f}  OPENAI={acc_s_open:.3f}  "
              f"DEEPSEEK={acc_s_deep:.3f}  XAI={acc_s_xai:.3f}  VOTE={acc_s_vote:.3f}")

    # 混淆矩阵（ORCH）
    labels = ["A","B","C","D"]
    cm = confusion_matrix(gold_all, orch_all, labels=labels)
    print("\n[CONFUSION MATRIX ORCH (raw counts)]")
    print("     A   B   C   D")
    for i, L in enumerate(labels):
        row = " ".join(f"{cm[i,j]:3d}" for j in range(len(labels)))
        print(f"{L}:  {row}")

    # McNemar (ORCH vs DEEPSEEK)
    b = c = 0
    for y, po, pd in zip(gold_all, orch_all, deep_all):
        orch_correct = (po == y)
        deep_correct = (pd == y)
        if orch_correct and not deep_correct:
            b += 1
        elif (not orch_correct) and deep_correct:
            c += 1
    if b + c > 0:
        # continuity-corrected McNemar
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        # df=1 -> CDF = 2*Phi(sqrt(x)) - 1
        z = math.sqrt(chi2)
        phi = 0.5 * (1 + math.erf(z / math.sqrt(2.0)))
        cdf = 2*phi - 1
        p_val = max(0.0, 1 - cdf)
    else:
        chi2 = 0.0
        p_val = 1.0

    print("\n[MCNEMAR] ORCH vs DEEPSEEK")
    print(f"  b = ORCH correct & DEEPSEEK wrong = {b}")
    print(f"  c = ORCH wrong  & DEEPSEEK correct = {c}")
    print(f"  b+c = {b+c}, chi2 = {chi2:.4f}, p ≈ {p_val:.6f}")
    if p_val < 0.05:
        print("  ==> 差异在 95% 置信水平下显著 (p < 0.05)")
    else:
        print("  ==> 差异在 95% 置信水平下不显著")

    # 平均延迟
    if total_q > 0:
        avg_lat_orch = lat_orch / total_q
        avg_lat_open = lat_open / total_q
        avg_lat_deep = lat_deep / total_q
        avg_lat_xai  = lat_xai  / total_q
    else:
        avg_lat_orch = avg_lat_open = avg_lat_deep = avg_lat_xai = 0.0

    print("\n[AVG LATENCY] (ms / question)")
    print(f"  ORCH     ≈ {avg_lat_orch:.0f} ms")
    print(f"  OPENAI   ≈ {avg_lat_open:.0f} ms")
    print(f"  DEEPSEEK ≈ {avg_lat_deep:.0f} ms")
    print(f"  XAI      ≈ {avg_lat_xai:.0f} ms")

    # ================= 画图 =================
    # 1) 全局 ACC 柱状图（含 VOTE）
    fig1 = plt.figure(figsize=(6,4))
    ax1 = fig1.add_subplot(1,1,1)
    models = ["ORCH","OPENAI","DEEPSEEK","XAI","VOTE"]
    accs   = [acc_orch, acc_open, acc_deep, acc_xai, acc_vote]
    ax1.bar(models, accs)
    ax1.set_ylim(0,1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Global Accuracy (4-choice MMLU, 10 subjects × 30 questions)")
    for i,v in enumerate(accs):
        ax1.text(i, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig1.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "global_acc.png")
    fig1.savefig(path1, dpi=160)
    print(f"[PLOT] saved -> {path1}")

    # 2) 分学科 ACC 柱状图
    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.add_subplot(1,1,1)
    idx = np.arange(len(SUBJECTS))
    width = 0.16
    orch_vals = [per_subject_acc[s]["orch"] for s in SUBJECTS]
    open_vals = [per_subject_acc[s]["open"] for s in SUBJECTS]
    deep_vals = [per_subject_acc[s]["deep"] for s in SUBJECTS]
    xai_vals  = [per_subject_acc[s]["xai"]  for s in SUBJECTS]
    vote_vals = [per_subject_acc[s]["vote"] for s in SUBJECTS]

    ax2.bar(idx - 2*width, orch_vals, width, label="ORCH")
    ax2.bar(idx - width,   open_vals, width, label="OPENAI")
    ax2.bar(idx,           deep_vals, width, label="DEEPSEEK")
    ax2.bar(idx + width,   xai_vals,  width, label="XAI")
    ax2.bar(idx + 2*width, vote_vals, width, label="VOTE")

    ax2.set_xticks(idx)
    ax2.set_xticklabels(SUBJECTS, rotation=45, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0,1.0)
    ax2.set_title("Per-subject Accuracy (4-choice MMLU)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "per_subject_acc.png")
    fig2.savefig(path2, dpi=160)
    print(f"[PLOT] saved -> {path2}")

    # 3) ORCH 混淆矩阵
    fig3 = plt.figure(figsize=(4.5,4))
    ax3 = fig3.add_subplot(1,1,1)
    im = ax3.imshow(cm.astype(float), interpolation="nearest")
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Gold")
    ax3.set_title("Confusion Matrix (ORCH)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax3.text(j, i, f"{cm[i,j]:d}", ha="center", va="center", fontsize=8)
    fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()
    path3 = os.path.join(PLOTS_DIR, "orch_confusion.png")
    fig3.savefig(path3, dpi=160)
    print(f"[PLOT] saved -> {path3}")

    # 4) McNemar (ORCH vs DEEPSEEK) 柱状图
    fig4 = plt.figure(figsize=(5,4))
    ax4 = fig4.add_subplot(1,1,1)
    ax4.bar(["b: ORCH✅ & DEEP❌", "c: ORCH❌ & DEEP✅"], [b, c])
    ax4.set_ylabel("Count")
    ax4.set_title(f"McNemar ORCH vs DEEPSEEK\nchi2={chi2:.3f}, p≈{p_val:.4f}")
    for i,v in enumerate([b,c]):
        ax4.text(i, v+0.5, str(v), ha="center", va="bottom")
    fig4.tight_layout()
    path4 = os.path.join(PLOTS_DIR, "mcnemar_orch_vs_deepseek.png")
    fig4.savefig(path4, dpi=160)
    print(f"[PLOT] saved -> {path4}")

    # 5) 价格 + 平均延迟 表格
    fig5 = plt.figure(figsize=(10,3))
    ax5 = fig5.add_subplot(1,1,1)
    ax5.axis("off")
    col_labels = ["Model", "Currency", "Input/1M", "CachedInput/1M", "Output/1M",
                  "Avg Latency (ms/question)"]
    table_data = [
        ["ORCH (3-agent + merge)", "-",   "-",    "-",    "-",   f"{avg_lat_orch:.0f}"],
        ["OpenAI gpt-4o-mini",     "USD", "0.15", "0.075","0.60",f"{avg_lat_open:.0f}"],
        ["Deepseek-chat",          "CNY", "2.0",  "0.2",  "-",   f"{avg_lat_deep:.0f}"],
        ["XAI grok-2-latest",      "USD", "2.0",  "-",    "10.0",f"{avg_lat_xai:.0f}"],
    ]
    table = ax5.table(cellText=table_data,
                      colLabels=col_labels,
                      loc="center")
    table.scale(1, 1.4)
    ax5.set_title("Price (per 1M tokens) & Avg Latency")
    fig5.tight_layout()
    path5 = os.path.join(PLOTS_DIR, "price_latency.png")
    fig5.savefig(path5, dpi=160)
    print(f"[PLOT] saved -> {path5}")

def main():
    ap = argparse.ArgumentParser(description="4-choice MMLU: ORCH + baselines + VOTE (10 subjects × 30)")
    ap.add_argument("--per_subject_n", type=int, default=30,
                    help="每个学科抽取题目数（默认 30）")
    args = ap.parse_args()
    mmlu_eval_allsubjects(per_subject_n=args.per_subject_n, seed=CFG["SEED"])

if __name__ == "__main__":
    main()
