# -*- coding: utf-8 -*-
# mmlu_pro_router_allinone.py
# ---------------------------------------------------------------------------------
# 一键运行：MMLU-Pro 评测（10选 A–J）
# - ORCH：同一原题并行发给 3 个 LLM（≤500字分析），合并 Agent 固定为 XAI，仅输出最终字母(A–J)
# - 三个基线：OPENAI / DEEPSEEK / XAI 各自直接回答一次（不拆、不比、不合并）
# - 控制台逐题打印形如：
#   [Q97/100] GOLD=B | ORCH=A | OPENAI=A | DEEPSEEK=B | XAI=C | lat=6302ms
# - 生成图：12.png（四柱图 + ORCH 混淆矩阵）
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, math, re, string, argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== Metrics (降级可用) =====
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

# ===== 评测依赖 =====
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("需要安装 `datasets`（pip install datasets）") from e
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("需要安装 `numpy`（pip install numpy）") from e
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("需要安装 `matplotlib`（pip install matplotlib）") from e
try:
    from sklearn.metrics import confusion_matrix
except Exception as e:
    raise RuntimeError("需要安装 `scikit-learn`（pip install scikit-learn）") from e

import requests

# ============================== 内联 API-KEY（若环境变量未设才使用） ==============================
INLINE_KEYS = {
    "OPENAI_API_KEY": "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",     # 可留空；若已设置环境变量会优先使用
    "DEEPSEEK_API_KEY": "sk-c181d17907ff4d848eca8225244aa67f",
    "GROQ_API_KEY": "",
    "XAI_API_KEY":   "xai-iTEf2LYg5Jvd4YIHmSmihv7gQWEv5XkiEKuzRwQ4lqBLwUsl4tndgGSnVsn6DCzBa3gRgTl0Wuuowag9",
}
for k, v in INLINE_KEYS.items():
    if v and not os.getenv(k):
        os.environ[k] = v

# ============================== Keys & Provider 配置 ==============================
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

LOCAL_BASE  = os.getenv("LOCAL_LLM_BASE", "")
LOCAL_KEY   = "EMPTY"
LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct")

PRICE = {
    "openai":     {"in": 0.005, "out": 0.015},
    "deepseek":   {"in": 0.002, "out": 0.004},
    "grok":       {"in": 0.005, "out": 0.010},
    "groq":       {"in": 0.001, "out": 0.002},
    "local-llama":{"in": 0.000, "out": 0.000},
    "echo":       {"in": 0.000, "out": 0.000},
}

SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Answer directly.\n"
    "2) Do NOT ask the user for more info.\n"
    "3) No templates or meta-commentary.\n"
    "4) Be specific, verifiable and concise.\n"
    "5) If info is missing, make the minimum reasonable assumption and state it.\n"
    "6) Prefer formal definitions/proofs when applicable.\n"
)

CFG: Dict[str, Any] = {
    # ---- Agents ----
    "OPENAI_COMPAT_PROFILES": [
        {"name":"openai","base_url":OAI_BASE,"api_key":OAI_KEY,"model":OAI_MODEL,
         "enabled":bool(OAI_KEY), "init_speed_ms":320, "init_quality":+0.02, "init_reliab":0.90},
        {"name":"deepseek","base_url":DEEPSEEK_BASE,"api_key":DEEPSEEK_KEY,"model":DEEPSEEK_MODEL,
         "enabled":bool(DEEPSEEK_KEY), "init_speed_ms":320, "init_quality":+0.01, "init_reliab":0.88},
        {"name":"local-llama","base_url":LOCAL_BASE,"api_key":LOCAL_KEY,"model":LOCAL_MODEL,
         "enabled":bool(LOCAL_BASE), "init_speed_ms":180, "init_quality":-0.01, "init_reliab":0.80},
    ],
    "GROQ": {"api_key":GROQ_KEY, "model":GROQ_MODEL, "enabled":bool(GROQ_KEY), "init_speed_ms":260, "init_quality":+0.02,"init_reliab":0.86},
    "XAI":  {"api_key":XAI_KEY,  "model":XAI_MODEL,  "base_url":XAI_BASE,"enabled":bool(XAI_KEY), "init_speed_ms":270, "init_quality":+0.02,"init_reliab":0.86},

    # ---- LLM call ----
    "LLM_TEMPERATURE": 0.2,
    "LLM_MAX_TOKENS": 900,         # 每个 Agent 允许的最大 tokens（足以覆盖≤500字分析）
    "ANSWER_MAX_CHARS": 500,       # 单 agent 分析输出长度上限（字符）
    "TIMEOUT_S": 60,

    # ---- EMA & scoring（无 trust）----
    "EMA_ALPHA": 0.15,
    "BASE_LAT_MS": 800,
    "BASE_RELIAB": 0.80,
    "BASE_QUALITY": 0.02,

    "W_RELIAB": 0.35,
    "W_QUAL":   0.30,
    "W_SPEED": -0.00012,
    "W_LEN":    0.00002,
    "W_COST":  -0.08,
    "LEN_CAP": 8000,

    "BANDIT": "ucb",
    "SEED": 42,

    # ---- Metrics ----
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "0")),
}

random.seed(CFG["SEED"])

# =============== Utils ===============
def divider(title=""):
    print("\n" + "="*86)
    if title: print(f"[{title}]")
    print("="*86)

def tab(rows, headers):
    cols = [len(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            if i < len(cols):
                cols[i] = max(cols[i], len(str(c)))
            else:
                cols.append(len(str(c)))
    col_fmts = [f'{{:<{w}}}' for w in cols]
    fmt = '  ' + ' | '.join(col_fmts)
    print(fmt.format(*[str(h) for h in headers]))
    print('  ' + '-+-'.join('-' * w for w in cols))
    for r in rows:
        cells = [str(x) for x in r] + [''] * (len(cols) - len(r))
        print(fmt.format(*cells))

def clip_text(t: str, n: int=1000) -> str:
    t = t or "";  return t if len(t)<=n else (t[:n] + " ...[TRUNCATED]")

def approx_tokens_from_text(text: str) -> int: return max(1, int(len(text)/4))

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

# =============== Metrics（无 trust） ===============
METRIC_AGENT_CALLS = Counter("agent_calls_total", "Total calls per agent", ["agent"])
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
                with open(self.path,"r",encoding="utf-8") as f:
                    loaded = json.load(f)
                self.data = loaded.get("agents", {})
                self.global_calls = loaded.get("global_calls", 0)
        except Exception:
            self.data = {}
    def save(self):
        try:
            with open(self.path,"w",encoding="utf-8") as f:
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
                "calls": 0,
                "accepted": 0,
                "bandit": {"alpha":1.0, "beta":1.0},
                "ucb": {"n":0, "sum_reward":0.0},
            }
    def update_after_call(self, name: str, latency_ms: int, ok_http: bool, text_len: int,
                          est_quality_boost: float, cost_per_ktok: float):
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        d = self.data[name]; a = CFG["EMA_ALPHA"]
        d["ema_latency_ms"] = (1-a)*d["ema_latency_ms"] + a*latency_ms
        d["ema_reliability"] = (1-a)*d["ema_reliability"] + a*(1.0 if ok_http else 0.0)
        d["ema_len"] = (1-a)*d["ema_len"] + a*min(text_len, CFG["LEN_CAP"])
        d["ema_quality"] = (1-a)*d["ema_quality"] + a*est_quality_boost
        d["ema_cost_per_ktok"] = (1-a)*d["ema_cost_per_ktok"] + a*max(0.0, cost_per_ktok)
        d["calls"] += 1; self.global_calls += 1
        METRIC_AGENT_EMA_LAT.labels(agent=name).set(d["ema_latency_ms"])
        METRIC_AGENT_EMA_REL.labels(agent=name).set(d["ema_reliability"])
        METRIC_AGENT_EMA_QUAL.labels(agent=name).set(d["ema_quality"])
        METRIC_AGENT_EMA_LEN.labels(agent=name).set(d["ema_len"])
    def snapshot(self, name: str) -> Dict[str, Any]:
        self.ensure(name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
        return dict(self.data[name])

METRICS = MetricsStore("agent_metrics.json")

# =============== Agent 抽象/实现 ===============
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

def composite_prob(name: str, text: str, http_status: Optional[int]) -> float:
    snap = METRICS.snapshot(name)
    W = CFG
    rel = snap["ema_reliability"]; qual= snap["ema_quality"]; lat = snap["ema_latency_ms"]
    L   = min(len(text or ""), CFG["LEN_CAP"]); cost= snap["ema_cost_per_ktok"]
    p = (W["W_RELIAB"]*rel + W["W_QUAL"]*(0.70 + qual) + W["W_SPEED"]*lat + W["W_LEN"]*L + W["W_COST"]*cost + 0.10)
    return max(0.05, min(0.98, p))

def print_agent_result(prefix: str, res: AgentResult):
    print(f"{prefix} -> (p={res.prob:.3f}, reliab={res.reliability:.2f}, latency={res.latency_ms}ms, http={res.http_status}, cost/1k=${res.cost_per_ktok:.3f})")
    print("TEXT:"); print(clip_text(res.result_text)); print("-"*86)

class BaseAgent:
    def __init__(self, name: str, kind: str = "llm"):
        self.name = name; self.kind = kind
        METRICS.ensure(self.name, CFG["BASE_LAT_MS"], CFG["BASE_QUALITY"], CFG["BASE_RELIAB"])
    def request(self, url: str, headers: Dict[str,str], payload: Dict[str,Any]) -> Tuple[int, Dict[str,Any], float]:
        t0 = time.time(); resp = requests.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
        return resp.status_code, (resp.json() if resp.content else {}), (time.time()-t0)*1000.0
    def _calc_cost(self, provider: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> float:
        price = PRICE.get(provider, {"in":0.0,"out":0.0}); ti = tokens_in or 0; to = tokens_out or 0
        return (price["in"]*ti + price["out"]*to) / 1000.0
    def _after(self, http: int, text: str, latency_ms: int,
               est_quality_boost: float, provider: str,
               tokens_in: Optional[int], tokens_out: Optional[int]) -> float:
        if tokens_out is None: tokens_out = approx_tokens_from_text(text or "")
        cost_k = self._calc_cost(provider, tokens_in, tokens_out)
        METRICS.update_after_call(self.name, latency_ms, http==200, len(text or ""), est_quality_boost, cost_k)
        METRIC_AGENT_CALLS.labels(agent=self.name).inc()
        METRIC_AGENT_LAT_MS.labels(agent=self.name).observe(max(1, latency_ms))
        METRIC_AGENT_COST_K.labels(agent=self.name).observe(max(0.0, cost_k))
        return cost_k
    def infer_fail(self, req: "OrchestrateRequest", err: str, provider: str, latency_ms: int=1200) -> AgentResult:
        cost_k = self._after(500, err, latency_ms, -0.03, provider, None, approx_tokens_from_text(err))
        p = composite_prob(self.name, err, 500)
        return AgentResult(f"[{self.name} ERROR] {err}\nEcho: {req.query[:200]}", p, METRICS.snapshot(self.name)["ema_reliability"], int(latency_ms), 500, err, 0, approx_tokens_from_text(err), cost_k)

class OpenAICompatAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, provider_key: str):
        super().__init__(name, kind="llm")
        self.base_url=base_url.rstrip("/"); self.api_key=api_key; self.model=model; self.provider_key=provider_key
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                cost_k = self._after(200, text, 120, -0.02, "echo", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200); rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 120, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res); return res
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
            payload = {"model": self.model,
                       "messages":[{"role":"system","content": SYSTEM_PROMPT},
                                   {"role":"user","content":req.query}],
                       "temperature": CFG["LLM_TEMPERATURE"], "max_tokens": CFG["LLM_MAX_TOKENS"]}
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() or json.dumps(data)[:2000]
            usage = data.get("usage",{}); ti = usage.get("prompt_tokens"); to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.00, self.provider_key, ti, to)
            p = composite_prob(self.name, text, code); rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res); return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", self.provider_key)
            print_agent_result(f"CALL {self.name} (ERROR)", res); return res

class GroqAgent(BaseAgent):
    def __init__(self, name: str, api_key: str, model: str):
        super().__init__(name, kind="llm"); self.api_key=api_key; self.model=model
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                cost_k = self._after(200, text, 120, -0.02, "echo", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200); rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 120, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res); return res
            url = f"{GROQ_BASE}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
            payload = {"model": self.model,
                       "messages":[{"role":"system","content": SYSTEM_PROMPT},
                                   {"role":"user","content":req.query}],
                       "temperature": CFG["LLM_TEMPERATURE"], "max_tokens": CFG["LLM_MAX_TOKENS"]}
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() or json.dumps(data)[:2000]
            usage = data.get("usage",{}); ti = usage.get("prompt_tokens"); to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.01, "groq", ti, to)
            p = composite_prob(self.name, text, code); rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res); return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", "groq")
            print_agent_result(f"CALL {self.name} (ERROR)", res); return res

class XaiGrokAgent(BaseAgent):
    def __init__(self, name: str, api_key: str, model: str, base_url: str):
        super().__init__(name, kind="llm"); self.api_key=api_key; self.model=model; self.base_url=base_url.rstrip("/")
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                cost_k = self._after(200, text, 120, -0.02, "echo", None, approx_tokens_from_text(text))
                p = composite_prob(self.name, text, 200); rel = METRICS.snapshot(self.name)["ema_reliability"]
                res = AgentResult(text, p, rel, 120, 200, None, 0, approx_tokens_from_text(text), cost_k)
                print_agent_result(f"CALL {self.name}", res); return res
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}
            payload = {"model": self.model,
                       "messages":[{"role":"system","content": SYSTEM_PROMPT},
                                   {"role":"user","content":req.query}],
                       "temperature": CFG["LLM_TEMPERATURE"], "max_tokens": CFG["LLM_MAX_TOKENS"]}
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() or json.dumps(data)[:2000]
            usage = data.get("usage",{}); ti = usage.get("prompt_tokens"); to = usage.get("completion_tokens")
            cost_k = self._after(code, text, int(dur), +0.02, "grok", ti, to)
            p = composite_prob(self.name, text, code); rel = METRICS.snapshot(self.name)["ema_reliability"]
            res = AgentResult(f"[{self.name}] {text}", p, rel, int(dur), code, None, ti, to, cost_k)
            print_agent_result(f"CALL {self.name}", res); return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}", "grok")
            print_agent_result(f"CALL {self.name} (ERROR)", res); return res

class RuleAgent(BaseAgent):
    def __init__(self): super().__init__("agent-rule", kind="rule")
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        text = f"[agent-rule] (rule) Echo: {req.query[:200]}"
        cost_k = self._after(200, text, 60, -0.02, "echo", None, approx_tokens_from_text(text))
        p = composite_prob(self.name, text, 200)
        res = AgentResult(text, p, METRICS.snapshot(self.name)["ema_reliability"], 60, 200, None, 0, approx_tokens_from_text(text), cost_k)
        print_agent_result(f"CALL agent-rule", res); return res

class EchoAgent(BaseAgent):
    def __init__(self): super().__init__("agent-echo", kind="local")
    def infer(self, req: "OrchestrateRequest") -> AgentResult:
        text = f"[agent-echo] Echo: {req.query[:200]}"
        cost_k = self._after(200, text, 120, -0.03, "echo", None, approx_tokens_from_text(text))
        p = composite_prob(self.name, text, 200)
        res = AgentResult(text, p, METRICS.snapshot(self.name)["ema_reliability"], 120, 200, None, 0, approx_tokens_from_text(text), cost_k)
        print_agent_result(f"CALL agent-echo", res); return res

# =============== 任务/编排 ===============
@dataclass
class OrchestrateRequest:
    query: str
    context: Dict[str, Any] = field(default_factory=dict)

# 10 选字母表（最多支持到 26 选）
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER10 = ALPHABET[:10]  # A..J

LETTER_RE_10 = re.compile(r"\b([A-J])\b", re.IGNORECASE)
def normalize_to_letter_10(text: str) -> str:
    if not text: return "?"
    m = LETTER_RE_10.search(text.strip())
    if m: return m.group(1).upper()
    t = text.strip().upper()
    for L in LETTER10:
        if re.match(rf"^\s*{L}\b", t): return L
    return "?"

def normalize_to_letter_generic(text: str, n_options: int) -> str:
    letters = ALPHABET[:max(1, min(n_options, 26))]
    pat = re.compile(rf"\b([{letters}])\b", re.IGNORECASE)
    if not text: return "?"
    m = pat.search(text.strip())
    if m: return m.group(1).upper()
    t = text.strip().upper()
    for L in letters:
        if re.match(rf"^\s*{L}\b", t): return L
    return "?"

# 取字段名（鲁棒）
def pick_field(ex: dict, candidates: List[str], default=None):
    for k in candidates:
        if k in ex: return k
    return default

# =============== Orchestrator（并行三 Agent + XAI 合并） ===============
class Orchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = [a for a in agents if getattr(a, "kind","") in ("llm","local","rule")]

    def _one_agent_analysis(self, agent: Agent, full_prompt: str) -> AgentResult:
        ana_tpl = (
            "请围绕该多项选择题进行'结构化分析'，严格控制在500字以内；"
            "按【要点】列出依据与排除逻辑，最后给出'暂定答案(仅A–J中的一个字母)'。\n\n"
            "{prompt}\n"
        )
        q = ana_tpl.format(prompt=full_prompt)
        res = agent.infer(OrchestrateRequest(q))
        trimmed = truncate_chars(res.result_text, CFG["ANSWER_MAX_CHARS"])
        return AgentResult(trimmed, res.prob, res.reliability, res.latency_ms, res.http_status, res.error,
                           res.tokens_in, res.tokens_out, res.cost_per_ktok)

    def _merge_xai(self, merger: Agent, analyses: List[Tuple[str,str]], full_prompt:str) -> str:
        bullet = "\n\n".join([f"[{name}] {txt}" for name,txt in analyses])
        merge_tpl = (
            "你将看到三名分析员对同一十选一题的分析（均≤500字）。"
            "请综合证据，仅输出最终选项字母（A–J），不得附加解释或其它字符。\n\n"
            f"{full_prompt}\n\n{bullet}\n\n最终答案："
        )
        res = merger.infer(OrchestrateRequest(merge_tpl))
        return normalize_to_letter_10(res.result_text)

    def orchestrator_answer(self, prompt: str) -> Tuple[str,int]:
        t0 = time.time()
        llms = [a for a in self.agents if getattr(a,"kind","")=="llm"]
        if not llms:
            # 本地兜底
            ans = "?"
            return ans, int((time.time()-t0)*1000)

        # 三个并行分析 agent：优先 OPENAI / DEEPSEEK / GROQ / 其它
        ordered = []
        # 按名字粗略排个序，保证多样性
        pref = ["agent-openai", "agent-deepseek", "agent-groq", "agent-xai", "agent-local", "agent-echo"]
        for p in pref:
            for a in llms:
                if a.name.startswith(p) and a not in ordered:
                    ordered.append(a)
        for a in llms:
            if a not in ordered:
                ordered.append(a)
        panel = ordered[:3] if len(ordered) >= 3 else (ordered + llms)[:3]

        # 合并 Agent 固定为 XAI（若没有可用 XAI，则退化到第一个 llm）
        merger = next((a for a in llms if a.name.startswith("agent-xai")), llms[0])

        analyses: List[Tuple[str,str]] = []
        lat_sum = 0
        with ThreadPoolExecutor(max_workers=len(panel)) as ex:
            futs = [ex.submit(self._one_agent_analysis, ag, prompt) for ag in panel]
            for i, fu in enumerate(as_completed(futs)):
                res = fu.result()
                # 注意：as_completed 的返回顺序与 panel 顺序不同，标注时使用完成顺序下标会错位
                # 这里统一按完成顺序标注 generic 名称
                ag_name = getattr(panel[i], "name", f"agent-{i+1}")
                analyses.append((ag_name, res.result_text))
                lat_sum += res.latency_ms

        final_letter = self._merge_xai(merger, analyses, prompt)
        total_ms = int((time.time()-t0)*1000)
        return final_letter, max(total_ms, lat_sum)

# =============== 装配 ===============
def build_agents() -> List[Agent]:
    agents: List[Agent] = []
    for p in CFG["OPENAI_COMPAT_PROFILES"]:
        if p["enabled"]:
            provider_key = p["name"] if p["name"] in PRICE else "openai"
            agents.append(OpenAICompatAgent(f"agent-{p['name']}", p["base_url"], p["api_key"], p["model"], provider_key))
    if CFG["GROQ"]["enabled"]:
        agents.append(GroqAgent("agent-groq", CFG["GROQ"]["api_key"], CFG["GROQ"]["model"]))
    if CFG["XAI"]["enabled"]:
        agents.append(XaiGrokAgent("agent-xai", CFG["XAI"]["api_key"], CFG["XAI"]["model"], CFG["XAI"]["base_url"]))
    agents.append(RuleAgent())
    if not any(getattr(a,"kind","")=="llm" for a in agents):
        agents.insert(0, EchoAgent())
    return agents

# =============== Baseline：单模型直接回答（A–J） ===============
def single_agent_baseline(agent_name_prefix: str, prompt: str, agents: List[Agent], n_options: int) -> str:
    chosen = None
    for a in agents:
        if a.name.startswith(agent_name_prefix) and getattr(a,"kind","")=="llm":
            chosen = a; break
    if chosen is None:
        for a in agents:
            if getattr(a,"kind","") == "llm":
                chosen = a; break
    if chosen is None:
        return "?"

    letters = ALPHABET[:max(1, min(n_options, 26))]
    ASK_TEMPLATE = (
        "You are given a multiple-choice question with {N} options (letters {L0}–{L1}).\n"
        "Pick the single correct option and return ONLY its letter.\n\n{q}\nAnswer:"
    )
    q = ASK_TEMPLATE.format(N=n_options, L0=letters[0], L1=letters[-1], q=prompt)
    res = chosen.infer(OrchestrateRequest(q))
    return normalize_to_letter_generic(res.result_text, n_options)

# =============== MMLU-Pro 评测 ===============
def mmlu_pro_eval(domain: Optional[str], split: str, max_exs: int, seed:int=42):
    random.seed(seed); np.random.seed(seed)

    print(f"[LOAD] TIGER-Lab/MMLU-Pro, split={split}")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)  # 只有 default 配置

    # 字段鲁棒识别
    ex0 = ds[0]
    q_key  = pick_field(ex0, ["question", "prompt"], "question")
    opt_key= pick_field(ex0, ["options", "choices"], "options")
    ans_k  = pick_field(ex0, ["answer_index", "answer_idx", "answer"], "answer_index")
    cat_k  = pick_field(ex0, ["category", "domain", "subject", "discipline"], None)

    # 可选 domain 过滤（如 law / math / biology 等，忽略大小写；支持包含匹配）
    if domain and domain.lower() != "all" and cat_k:
        dom = domain.lower()
        ds = ds.filter(lambda ex: dom in str(ex[cat_k]).lower())

    if max_exs and len(ds) > max_exs:
        ds = ds.select(range(max_exs))

    agents = build_agents()
    orch = Orchestrator(agents)

    gold_list: List[str] = []
    orch_pred: List[str] = []
    bl_openai: List[str] = []
    bl_deepseek: List[str] = []
    bl_xai: List[str] = []

    N = len(ds)
    for i, ex in enumerate(ds):
        q = ex[q_key]
        options = ex[opt_key]
        # 容错：若有个别样本不是 10 选，按实际长度出题与解析（最多 26）
        n_opts = max(2, min(len(options), 26))
        letters = ALPHABET[:n_opts]
        gold_idx = int(ex[ans_k])
        gold_letter = letters[gold_idx]

        prompt = (
            f"Question: {q}\n"
            + "\n".join([f"{letters[j]}. {options[j]}" for j in range(n_opts)])
        )

        # ORCH：并行三 agent + XAI 合并
        orch_ans, latency_ms = orch.orchestrator_answer(prompt)

        # 三基线：OPENAI / DEEPSEEK / XAI
        openai_ans = single_agent_baseline("agent-openai", prompt, agents, n_opts)
        deepseek_ans = single_agent_baseline("agent-deepseek", prompt, agents, n_opts)
        xai_ans     = single_agent_baseline("agent-xai",     prompt, agents, n_opts)

        gold_list.append(gold_letter); orch_pred.append(orch_ans)
        bl_openai.append(openai_ans);  bl_deepseek.append(deepseek_ans); bl_xai.append(xai_ans)

        print(f"[Q{i+1}/{N}] GOLD={gold_letter} | ORCH={orch_ans} | OPENAI={openai_ans} | DEEPSEEK={deepseek_ans} | XAI={xai_ans} | lat={latency_ms}ms")

    # 汇总 ACC
    def acc(ys, ps): return sum(1 for y,p in zip(ys,ps) if y==p) / max(1, len(ys))
    acc_orch = acc(gold_list, orch_pred)
    acc_open = acc(gold_list, bl_openai)
    acc_deep = acc(gold_list, bl_deepseek)
    acc_xai  = acc(gold_list, bl_xai)

    print(f"\n[ACC] ORCH={acc_orch:.3f} | OPENAI={acc_open:.3f} | DEEPSEEK={acc_deep:.3f} | XAI={acc_xai:.3f}")

    # 混淆矩阵（按出现过的标签字母排序）
    uniq_letters = sorted(list(set(gold_list + orch_pred)), key=lambda c: ALPHABET.index(c) if c in ALPHABET else 999)
    cm = confusion_matrix(gold_list, orch_pred, labels=uniq_letters)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # 画图：四柱 + ORCH CM
    fig = plt.figure(figsize=(12, 4.8))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(["ORCH","OPENAI","DEEPSEEK","XAI"], [acc_orch, acc_open, acc_deep, acc_xai])
    ax1.set_ylim(0,1.0); ax1.set_title(f"MMLU-Pro Accuracy ({domain or 'all'})"); ax1.set_ylabel("Accuracy")

    ax2 = fig.add_subplot(1,2,2)
    im = ax2.imshow(cm_norm, interpolation="nearest")
    ax2.set_xticks(range(len(uniq_letters))); ax2.set_yticks(range(len(uniq_letters)))
    ax2.set_xticklabels(uniq_letters); ax2.set_yticklabels(uniq_letters)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Gold"); ax2.set_title("Confusion Matrix (ORCH)")
    for r in range(len(uniq_letters)):
        for c in range(len(uniq_letters)):
            ax2.text(c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center")

    fig.tight_layout()
    out_path = "12.png"
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")

# =============== REPL（可选） ===============
def repl():
    agents = build_agents()
    if CFG["METRICS_PORT"]>0:
        divider(f"METRICS EXPORTER : http://127.0.0.1:{CFG['METRICS_PORT']}/metrics")
        start_http_server(CFG["METRICS_PORT"])

    divider("BOOT")
    rows=[]
    for a in agents:
        s=METRICS.snapshot(a.name)
        rows.append([a.name, getattr(a,"kind","?"),
                     f"{s['ema_latency_ms']:.0f}ms", f"{s['ema_reliability']:.2f}",
                     f"{s['ema_quality']:+.3f}", f"cost_k={s['ema_cost_per_ktok']:.3f}", s.get("calls",0)])
    tab(rows, ["agent","kind","ema_latency","ema_reliab","ema_qual","ema_cost/k","calls"])
    print("\nType your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            q = input(">> ").strip()
        except EOFError:
            break
        if not q or q.lower()=="exit": print("bye."); break
        a0 = next((a for a in agents if getattr(a,"kind","")=="llm"), agents[0])
        res = a0.infer(OrchestrateRequest(q))
        print(clip_text(res.result_text, 2000))

# =============== CLI ===============
def main():
    ap = argparse.ArgumentParser(description="MMLU-Pro quick eval (3-agent analyses; XAI merge; 10-choice)")
    ap.add_argument("--mode", choices=["eval","repl"], default="eval", help="eval=评测; repl=交互")
    ap.add_argument("--domain", default="law", help="按类别过滤（如 law / math / biology / all）")
    ap.add_argument("--split", default="test", choices=["test","validation","dev"], help="数据划分")
    ap.add_argument("--max_exs", type=int, default=100, help="最多抽取题目数")
    args = ap.parse_args()

    if args.mode == "repl":
        repl()
    else:
        mmlu_pro_eval(args.domain, args.split, args.max_exs, seed=CFG["SEED"])

if __name__ == "__main__":
    main()
