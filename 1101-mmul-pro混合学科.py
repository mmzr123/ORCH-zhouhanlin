# -*- coding: utf-8 -*-
# mmlu_router_allinone.py  (MMLU-Pro: 10-choice A–J)
# ---------------------------------------------------------------------------------
# 一键运行：ORCH(并行三Agent分析≤500字 + 合并Agent=固定XAI只出A–J) + 三条单模型基线
# - 数据：TIGER-Lab/MMLU-Pro（十选一、含category；默认split=test，仅取options==10）
# - 控制台逐题打印： [Q97/100] GOLD=B | ORCH=A | OPENAI=A | DEEPSEEK=B | XAI=C | lat=6302ms
# - 生成图：12.png（四柱图 + ORCH 10×10混淆矩阵）
# 依赖：pip install datasets numpy matplotlib scikit-learn requests
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, math, collections, re, argparse, string
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
)

CFG: Dict[str, Any] = {
    "OPENAI_COMPAT_PROFILES": [
        {"name":"openai","base_url":OAI_BASE,"api_key":OAI_KEY,"model":OAI_MODEL,
         "enabled":bool(OAI_KEY), "init_speed_ms":320, "init_quality":+0.02, "init_reliab":0.90},
        {"name":"deepseek","base_url":DEEPSEEK_BASE,"api_key":DEEPSEEK_KEY,"model":DEEPSEEK_MODEL,
         "enabled":bool(DEEPSEEK_KEY), "init_speed_ms":320, "init_quality":+0.01, "init_reliab":0.88},
        {"name":"local-llama","base_url":LOCAL_BASE,"api_key":LOCAL_KEY,"model":LOCAL_MODEL,
         "enabled":bool(LOCAL_BASE), "init_speed_ms":180, "init_quality":-0.01, "init_reliab":0.80},
    ],
    "XAI":  {"api_key":XAI_KEY,  "model":XAI_MODEL,  "base_url":XAI_BASE,"enabled":bool(XAI_KEY),
             "init_speed_ms":270, "init_quality":+0.02,"init_reliab":0.86},

    "LLM_TEMPERATURE": 0.2,
    "LLM_MAX_TOKENS": 900,         # 容纳≤500字分析
    "ANSWER_MAX_CHARS": 500,
    "TIMEOUT_S": 60,

    # EMA-only（无 trust）
    "EMA_ALPHA": 0.15,
    "BASE_LAT_MS": 800,
    "BASE_RELIAB": 0.80,
    "BASE_QUALITY": 0.02,

    "W_RELIAB": 0.35, "W_QUAL": 0.30, "W_SPEED": -0.00012, "W_LEN": 0.00002, "W_COST": -0.08,
    "LEN_CAP": 8000,

    "BANDIT": "ucb",
    "SEED": 42,

    "MAX_WORKERS": 8,
    "STATE_PATH": "agent_metrics.json",

    "METRICS_PORT": int(os.getenv("METRICS_PORT", "0")),
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])

# =============== Utils ===============
A_J = [chr(ord('A')+i) for i in range(10)]
LETTER10_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)

def normalize_to_letter_10(text: str) -> str:
    if not text: return "?"
    m = LETTER10_RE.search(text.strip())
    if m: return m.group(1).upper()
    t = (text or "").strip().upper()
    for L in A_J:
        if re.match(rf"^\s*{L}\b", t): return L
    return "?"

def clip_text(t: str, n: int=1000) -> str:
    t = t or "";  return t if len(t)<=n else (t[:n] + " ...[TRUNCATED]")

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

# =============== Metrics (no trust) ===============
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
                "calls": 0, "accepted": 0,
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

METRICS = MetricsStore(CFG["STATE_PATH"])

def composite_prob(name: str, text: str, http_status: Optional[int]) -> float:
    snap = METRICS.snapshot(name)
    W = CFG
    rel = snap["ema_reliability"]; qual= snap["ema_quality"]; lat = snap["ema_latency_ms"]
    L   = min(len(text or ""), CFG["LEN_CAP"]); cost= snap["ema_cost_per_ktok"]
    p = (W["W_RELIAB"]*rel + W["W_QUAL"]*(0.70 + qual) + W["W_SPEED"]*lat + W["W_LEN"]*L + W["W_COST"]*cost + 0.10)
    return max(0.05, min(0.98, p))

def approx_tokens_from_text(text: str) -> int: return max(1, int(len(text)/4))

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
        text = f"[agent-rule] (rule) Please follow policy. Echo: {req.query[:200]}"
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

class Orchestrator:
    """分发同一原题到三个 agent（≤500字分析），合并判定由 XAI 固定输出 A–J"""
    def __init__(self, agents: List[Agent]):
        self.agents = [a for a in agents if getattr(a, "kind","") in ("llm","rule","local")]

    def _one_agent_analysis(self, agent: Agent, full_prompt: str) -> AgentResult:
        ana_tpl = (
            "请围绕以下十选一多项选择题进行结构化分析，严格控制在500字以内；"
            "按【要点】列出依据与排除逻辑，最后给出'暂定答案(仅A~J中的一个字母)'。\n\n{prompt}\n"
        )
        q = ana_tpl.format(prompt=full_prompt)
        res = agent.infer(OrchestrateRequest(q))
        trimmed = truncate_chars(res.result_text, CFG["ANSWER_MAX_CHARS"])
        return AgentResult(trimmed, res.prob, res.reliability, res.latency_ms, res.http_status, res.error,
                           res.tokens_in, res.tokens_out, res.cost_per_ktok)

    def _merge_by_xai(self, analyses: List[Tuple[str,str]], full_prompt:str) -> str:
        # 固定使用 XAI 作为合并判定Agent（若不可用则回退首个 LLM）
        merger = next((a for a in self.agents if a.name == "agent-xai"), None)
        if merger is None:
            merger = next((a for a in self.agents if getattr(a,"kind","")=="llm"), self.agents[0])
        bullet = "\n\n".join([f"[{name}] {txt}" for name,txt in analyses])
        merge_tpl = (
            "你将看到三名分析员对同一十选一题的分析（均≤500字）。"
            "请综合证据，仅输出最终选项字母（A/B/C/D/E/F/G/H/I/J），不得附加解释或其它字符。\n\n"
            f"{full_prompt}\n\n{bullet}\n\n最终答案："
        )
        res = merger.infer(OrchestrateRequest(merge_tpl))
        return normalize_to_letter_10(res.result_text)

    def answer(self, prompt: str, panel_agents: List[Agent]) -> Tuple[str,int]:
        t0 = time.time()
        analyses: List[Tuple[str,str]] = []
        with ThreadPoolExecutor(max_workers=len(panel_agents)) as ex:
            futs = [ex.submit(self._one_agent_analysis, ag, prompt) for ag in panel_agents]
            for i, fu in enumerate(as_completed(futs)):
                res = fu.result()
                # 注意：as_completed顺序与panel_agents顺序不同，这里统一记录 agent 名称
                analyses.append((panel_agents[i].name, res.result_text))
        letter = self._merge_by_xai(analyses, prompt)
        return letter, int((time.time()-t0)*1000)

# =============== Build agents ===============
def build_agents() -> List[Agent]:
    agents: List[Agent] = []
    for p in CFG["OPENAI_COMPAT_PROFILES"]:
        if p["enabled"]:
            provider_key = p["name"] if p["name"] in PRICE else "openai"
            agents.append(OpenAICompatAgent(f"agent-{p['name']}", p["base_url"], p["api_key"], p["model"], provider_key))
    if CFG["XAI"]["enabled"]:
        agents.append(XaiGrokAgent("agent-xai", CFG["XAI"]["api_key"], CFG["XAI"]["model"], CFG["XAI"]["base_url"]))
    agents.append(RuleAgent())
    if not any(getattr(a,"kind","")=="llm" for a in agents):
        agents.insert(0, EchoAgent())
    return agents

# =============== Baselines (single agent direct) ===============
def single_agent_baseline(agent_name_prefix: str, prompt: str, agents: List[Agent]) -> str:
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
    q = (
        "You are given a 10-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, D, E, F, G, H, I, or J.\n\n"
        f"{prompt}\n"
    )
    res = chosen.infer(OrchestrateRequest(q))
    return normalize_to_letter_10(res.result_text)

# =============== HTTP Agents finished, now Orchestrated Eval ===============
def sample_mmlu_pro(ds, max_exs: int, min_categories: int = 8) -> List[int]:
    """
    从 MMLU-Pro 测试集按类别均匀抽样，避免单一学科（如全是 law）。
    仅保留 options 长度==10 的样本。
    """
    idx_all = [i for i, opts in enumerate(ds["options"]) if isinstance(opts, list) and len(opts) == 10]
    cats = [ds["category"][i] for i in idx_all]
    by_cat: Dict[str, List[int]] = {}
    for i, c in zip(idx_all, cats):
        by_cat.setdefault(c, []).append(i)
    cat_list = sorted(by_cat.keys())
    random.shuffle(cat_list)
    if len(cat_list) < min_categories:
        # 若可用类别少于阈值，也尽量均匀覆盖
        min_categories = len(cat_list)
    pick_cats = cat_list[:min_categories]

    per_cat = max(1, max_exs // max(1, len(pick_cats)))
    chosen: List[int] = []
    for c in pick_cats:
        pool = by_cat[c]
        random.shuffle(pool)
        chosen.extend(pool[:per_cat])
    # 如果数量仍不足，继续补齐其它类别
    if len(chosen) < max_exs:
        remain = [i for i in idx_all if i not in chosen]
        random.shuffle(remain)
        chosen.extend(remain[:max_exs - len(chosen)])
    return chosen[:max_exs]

def mmlu_pro_eval(max_exs: int, seed:int=42):
    random.seed(seed); np.random.seed(seed)

    # 加载 MMLU-Pro（仅使用 split=test）
    print("[LOAD] TIGER-Lab/MMLU-Pro (split=test)")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")  # fields: question, options(list), answer_index, category, ...
    # 过滤十选一
    idxs = sample_mmlu_pro(ds, max_exs=max_exs, min_categories=8)

    agents = build_agents()
    orch = Orchestrator(agents)

    # Panel agents：优先 openai/deepseek/xai；若不足则补齐任意 LLM
    llm_agents = [a for a in agents if getattr(a,"kind","")=="llm"]
    pref_names = ["agent-openai", "agent-deepseek", "agent-xai"]
    panel = []
    for nm in pref_names:
        a = next((x for x in llm_agents if x.name == nm), None)
        if a: panel.append(a)
    if len(panel) < 3:
        for a in llm_agents:
            if a not in panel and len(panel) < 3:
                panel.append(a)
    if not panel:
        panel = llm_agents[:1] or agents[:1]

    gold_list: List[str] = []
    orch_pred: List[str] = []
    bl_openai: List[str] = []
    bl_deepseek: List[str] = []
    bl_xai: List[str] = []

    for qi, i in enumerate(idxs, 1):
        q = ds["question"][i]
        opts = ds["options"][i]   # list of 10
        gold_idx = int(ds["answer_index"][i])
        # 构造十选一题面
        prompt = "Question: " + str(q).strip() + "\nOptions:\n" + \
                 "\n".join([f"{A_J[k]}. {opts[k]}" for k in range(10)])

        # ORCH：三并行 + XAI 合并
        orch_ans, latency_ms = orch.answer(prompt, panel)

        # 三基线
        openai_ans = single_agent_baseline("agent-openai", prompt, agents)
        deepseek_ans = single_agent_baseline("agent-deepseek", prompt, agents)
        xai_ans     = single_agent_baseline("agent-xai",     prompt, agents)

        gold_letter = A_J[gold_idx] if 0 <= gold_idx < 10 else "?"
        gold_list.append(gold_letter)
        orch_pred.append(orch_ans)
        bl_openai.append(openai_ans)
        bl_deepseek.append(deepseek_ans)
        bl_xai.append(xai_ans)

        # 控制台逐题
        print(f"[Q{qi}/{len(idxs)}] GOLD={gold_letter} | ORCH={orch_ans} | OPENAI={openai_ans} | DEEPSEEK={deepseek_ans} | XAI={xai_ans} | lat={latency_ms}ms")

    # 汇总 ACC
    def acc(ys, ps): return sum(1 for y,p in zip(ys,ps) if y==p) / max(1, len(ys))
    acc_orch = acc(gold_list, orch_pred)
    acc_open = acc(gold_list, bl_openai)
    acc_deep = acc(gold_list, bl_deepseek)
    acc_xai  = acc(gold_list, bl_xai)
    print(f"\n[ACC] ORCH={acc_orch:.3f} | OPENAI={acc_open:.3f} | DEEPSEEK={acc_deep:.3f} | XAI={acc_xai:.3f}")

    # 混淆矩阵（ORCH，A–J）
    labels = A_J
    cm = confusion_matrix(gold_list, orch_pred, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # 画图：四柱 + ORCH 10×10 CM
    fig = plt.figure(figsize=(13.5,5.0))
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(["ORCH","OPENAI","DEEPSEEK","XAI"], [acc_orch, acc_open, acc_deep, acc_xai])
    ax1.set_ylim(0,1.0)
    ax1.set_title("MMLU-Pro Accuracy (10-choice)")
    ax1.set_ylabel("Accuracy")

    ax2 = fig.add_subplot(1,2,2)
    im = ax2.imshow(cm_norm, interpolation="nearest")
    ax2.set_xticks(range(len(labels))); ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Gold")
    ax2.set_title("Confusion Matrix (ORCH, A–J)")
    for ii in range(len(labels)):
        for jj in range(len(labels)):
            ax2.text(jj, ii, f"{cm_norm[ii, jj]:.2f}", ha="center", va="center")

    fig.tight_layout()
    out_path = "MMLU-Pro (10-choice).png"
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")

# =============== Simple REPL (optional) ===============
def repl():
    agents = build_agents()
    if CFG["METRICS_PORT"]>0:
        print(f"[Metrics] http://127.0.0.1:{CFG['METRICS_PORT']}/metrics")
        start_http_server(CFG["METRICS_PORT"])

    print("\n[BOOT] Agents:")
    for a in agents:
        s=METRICS.snapshot(a.name)
        print(f"  - {a.name:14s} kind={a.kind:5s} ema_lat={s['ema_latency_ms']:.0f}ms ema_rel={s['ema_reliability']:.2f} ema_qual={s['ema_quality']:+.3f}")

    print("\nType your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            q = input(">> ").strip()
        except EOFError:
            break
        if not q or q.lower()=="exit": print("bye."); break
        # 单Agent直答（演示）
        a0 = next((a for a in agents if getattr(a,"kind","")=="llm"), agents[0])
        res = a0.infer(OrchestrateRequest(q))
        print(clip_text(res.result_text, 2000))

# =============== HTTP Agent classes that rely on BaseAgent ===============
# （放在后面只是为了文件结构清晰；类定义已在上方）

# =============== CLI ===============
def main():
    ap = argparse.ArgumentParser(description="MMLU-Pro (10-choice) : ORCH(3-agent analyses + XAI merge) + 3 baselines")
    ap.add_argument("--mode", choices=["eval","repl"], default="eval", help="eval=十选一评测; repl=交互直答")
    ap.add_argument("--max_exs", type=int, default=100, help="最多抽取题目数（按类别均匀抽样）")
    args = ap.parse_args()

    if args.mode == "repl":
        repl()
    else:
        mmlu_pro_eval(max_exs=args.max_exs, seed=CFG["SEED"])

# ===== Implement HTTP LLM infer (after main to keep top clean) =====
class _HTTPMixin:
    pass

# Reuse OpenAICompatAgent / XaiGrokAgent defined above (already complete)

if __name__ == "__main__":
    main()
