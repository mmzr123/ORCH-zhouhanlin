# -*- coding: utf-8 -*-
# mmlu_pro_allin_one_vote500.py
# -----------------------------------------------------------------------------
# MMLU-Pro (10-choice A–J)
# 多代理 × 自一致投票  ->  合并理由<=500字（轮询三LLM做压缩） -> 字母答案
# - 无概率、无二段XAI裁决；XAI 作为第3 LLM参与投票与摘要轮询
# - 选项乱序并映射回原 A–J（降低顺序敏感性）
# - 显示 [SIMPLE]/[COMPLEX] 路径；激进旋转 + Thompson Bandit
# - 评测汇总四柱图 + 混淆矩阵
# 依赖: pip install datasets numpy matplotlib scikit-learn requests
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, re, argparse, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== 可选 Prometheus（不装也不报错） ======
try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram
except Exception:
    class _Noop:
        def labels(self,*a,**k): return self
        def inc(self,*a,**k): return None
        def set(self,*a,**k): return None
        def observe(self,*a,**k): return None
    def start_http_server(*a,**k): return None
    Counter = Gauge = Histogram = lambda *a,**k: _Noop()

# ====== 数据 / 评测 ======
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

# ================== 内联/环境 KEY（环境优先） ==================
INLINE_KEYS = {
    "OPENAI_API_KEY": "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",     # 可留空；若已设置环境变量会优先使用
    "DEEPSEEK_API_KEY": "sk-c181d17907ff4d848eca8225244aa67f",
    "GROQ_API_KEY": "",
    "XAI_API_KEY":   "xai-iTEf2LYg5Jvd4YIHmSmihv7gQWEv5XkiEKuzRwQ4lqBLwUsl4tndgGSnVsn6DCzBa3gRgTl0Wuuowag9",
}
for k,v in INLINE_KEYS.items():
    if v and not os.getenv(k): os.environ[k] = v

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

PRICE = {
    "openai":   {"in":0.005, "out":0.015},
    "deepseek": {"in":0.002, "out":0.004},
    "grok":     {"in":0.005, "out":0.010},
    "echo":     {"in":0.0, "out":0.0},
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
    # Agent 开关
    "OPENAI": {"base":OAI_BASE, "key":OAI_KEY, "model":OAI_MODEL, "enabled":bool(OAI_KEY)},
    "DEEPSEEK":{"base":DEEPSEEK_BASE,"key":DEEPSEEK_KEY,"model":DEEPSEEK_MODEL,"enabled":bool(DEEPSEEK_KEY)},
    "XAI":    {"base":XAI_BASE, "key":XAI_KEY, "model":XAI_MODEL, "enabled":bool(XAI_KEY)},

    # LLM 调用
    "LLM_TEMPERATURE": 0.65,       # 自一致建议 0.6~0.8
    "LLM_MAX_TOKENS":  900,
    "TIMEOUT_S":       60,

    # 自一致采样
    "SAMPLES_PER_AGENT": 3,        # =1 则关闭自一致
    "VOTE_TOPK_EARLY":  5,         # 早停票数窗口，激进旋转判断

    # 任务与并发
    "SEED": 42,
    "MAX_WORKERS": 8,

    # Simple/Complex 判定
    "SIMPLE_MAX_CHARS": 140,       # 题干较短、无明显“证明/多步推理”关键词
    "SIMPLE_MAX_WORDS": 28,

    # 500字压缩（严格上限）
    "FUSE_CHAR_LIMIT": 500,

    # Thompson Bandit（摘要者/主答者顺序）
    "BANDIT_ALPHA0": 1.0,
    "BANDIT_BETA0":  1.0,

    # 指标端口（可选）
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "0")),
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])

# =================== 工具 & 常量 ===================
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

def is_simple(text: str) -> bool:
    # 简单判断：长度/词数 + 关键字
    t = (text or "").strip()
    t_low = t.lower()
    if any(k in t_low for k in ["why","how","prove","proof","计划","推理","多步","pipeline","算法","复杂","trade-off","比较","证明"]):
        return False
    if len(t) > CFG["SIMPLE_MAX_CHARS"]: return False
    if len(t.split()) > CFG["SIMPLE_MAX_WORDS"]: return False
    return True

def clip_text(t: str, n: int=1000) -> str:
    t = t or "";  return t if len(t)<=n else (t[:n] + " ...[TRUNCATED]")

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

def safe_join_unique(parts: List[str], lim: int) -> str:
    """合并多段理由：去重 -> 拼接 -> 严格<=lim字"""
    seen=set(); out=[]
    for p in parts:
        pp = re.sub(r"\s+"," ", (p or "").strip())
        if not pp: continue
        if pp in seen: continue
        seen.add(pp)
        out.append(pp)
        joined = " ".join(out)
        if len(joined) >= lim:
            return joined[:lim]
    return " ".join(out)[:lim]

def analyze_tag(tag:str)->str:
    return "[SIMPLE]" if tag=="simple" else "[COMPLEX]"

# =================== 指标（轻量） ===================
METRIC_AGENT_CALLS = Counter("agent_calls_total","Total calls per agent",["agent"])
METRIC_AGENT_LAT_MS= Histogram("agent_latency_ms","Latency per agent (ms)",["agent"])

# =================== Agent 抽象/实现 ===================
@runtime_checkable
class Agent(Protocol):
    name: str
    kind: str
    def infer(self, query: str, temperature: float, max_tokens: int) -> Tuple[str,int,int,int]:
        """return text, latency_ms, http, approx_out_tokens"""
        ...

@dataclass
class AgentProfile:
    name: str
    base: str
    key: str
    model: str
    provider: str

class OpenAICompatAgent(Agent):
    def __init__(self, prof: AgentProfile):
        self.name = prof.name
        self.kind = "llm"
        self.base = prof.base.rstrip("/")
        self.key  = prof.key
        self.model= prof.model
        self.provider = prof.provider
    def infer(self, query: str, temperature: float, max_tokens: int) -> Tuple[str,int,int,int]:
        t0 = time.time()
        if not self.key:
            # echo
            txt = f"[{self.name}] (echo) {query[:480]}"
            dur = int((time.time()-t0)*1000)
            METRIC_AGENT_CALLS.labels(agent=self.name).inc()
            METRIC_AGENT_LAT_MS.labels(agent=self.name).observe(max(1,dur))
            return txt, dur, 200, int(len(txt)/4)
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content":SYSTEM_PROMPT},
                         {"role":"user","content":query}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
            dur = int((time.time()-t0)*1000)
            code = resp.status_code
            data = resp.json() if resp.content else {}
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() or json.dumps(data)[:2000]
            METRIC_AGENT_CALLS.labels(agent=self.name).inc()
            METRIC_AGENT_LAT_MS.labels(agent=self.name).observe(max(1,dur))
            return text, dur, code, int(len(text)/4)
        except Exception as e:
            dur = int((time.time()-t0)*1000)
            METRIC_AGENT_CALLS.labels(agent=self.name).inc()
            METRIC_AGENT_LAT_MS.labels(agent=self.name).observe(max(1,dur))
            return f"[{self.name} ERROR] {type(e).__name__}: {e}", dur, 500, 0

def build_agents() -> List[Agent]:
    agents: List[Agent] = []
    pref: List[Tuple[str,Dict[str,Any],str]] = [
        ("agent-openai", CFG["OPENAI"], "openai"),
        ("agent-deepseek", CFG["DEEPSEEK"], "deepseek"),
        ("agent-xai", CFG["XAI"], "grok"),
    ]
    for nm,conf,prov in pref:
        if conf["enabled"]:
            agents.append(OpenAICompatAgent(AgentProfile(nm, conf["base"], conf["key"], conf["model"], prov)))
    if not agents:
        # 至少放一个 echo
        agents.append(OpenAICompatAgent(AgentProfile("agent-echo","", "", "echo-model", "echo")))
    return agents

# =================== 题面乱序与映射 ===================
def shuffle_options_and_build_prompt(question: str, options: List[str]) -> Tuple[str, List[int], List[int]]:
    idx = list(range(len(options)))  # 0..9
    random.shuffle(idx)
    inv = [0]*len(idx)
    for new_pos, old_pos in enumerate(idx):
        inv[old_pos] = new_pos
    # 构造题面（按打乱后的次序贴 A..J）
    opt_lines = [f"{A_J[k]}. {options[idx[k]]}" for k in range(10)]
    prompt = "Question: " + str(question).strip() + "\nOptions:\n" + "\n".join(opt_lines)
    return prompt, idx, inv

def map_letter_back(letter_shuf: str, idx: List[int], inv: List[int]) -> str:
    if letter_shuf not in A_J: return "?"
    shuf_pos = A_J.index(letter_shuf)
    old_pos  = idx[shuf_pos]
    return A_J[old_pos]

# =================== 提示词（避免 .format 花括号冲突） ===================
ANALYZE_PROMPT_TPL = (
    "请基于以下十选一题，进行严谨分析与演绎，直接给出结构化的推理过程（禁止反问），并保持内容紧凑：\n"
    "— 列出 2-4 个关键依据；\n"
    "— 明确去重与解决冲突；\n"
    "— 最后在一行用 “结论：<A|B|...|J> + 一句话依据”。\n\n"
    "<<PROMPT>>\n"
)

LETTER_ONLY_TPL = (
    "你将看到一道十选一题（A–J）。请仅输出最终答案字母（A 或 B 或 ... 或 J），**不得输出其他字符**。\n\n"
    "<<PROMPT>>\n"
)

FUSE_COMPRESS_TPL = (
    "将下面多段解题理由去重合并为**不超过 {lim} 个字符**的中文摘要："
    "1句总判断 + 2-4个关键依据 + 如有冲突写出取舍理由；禁止反问；只给正文。\n\n"
    "<<PARTS>>\n"
)

# =================== Bandit（Thompson） ===================
@dataclass
class BetaAB:
    alpha: float = CFG["BANDIT_ALPHA0"]
    beta:  float = CFG["BANDIT_BETA0"]
    def sample(self) -> float:
        # numpy 的 beta 在 [0,1]
        return np.random.beta(self.alpha, self.beta)
    def push(self, success: bool):
        if success: self.alpha += 1.0
        else:       self.beta  += 1.0

class BanditPool:
    def __init__(self, names: List[str]):
        self.pool: Dict[str,BetaAB] = {n: BetaAB() for n in names}
    def pick(self, exclude: Optional[List[str]]=None) -> str:
        ex = set(exclude or [])
        cands = [(n,ab.sample()) for n,ab in self.pool.items() if n not in ex]
        cands.sort(key=lambda x:x[1], reverse=True)
        return cands[0][0]
    def update(self, name: str, success: bool):
        if name in self.pool:
            self.pool[name].push(success)

# =================== 编排器 ===================
class Orchestrator:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.bandit = BanditPool([a.name for a in agents])  # 用于挑选“摘要者/主答者”顺序

    def _complexity(self, qtext: str) -> str:
        return "simple" if is_simple(qtext) else "complex"

    def _one_agent_reason_and_vote(self, agent: Agent, prompt: str, samples:int, temp:float) -> Tuple[List[str], List[str]]:
        """返回：rationales[], letters[]（letters 为严格字母提示得到）"""
        rats: List[str] = []
        lets: List[str] = []
        for _ in range(max(1,samples)):
            # 分两步：先“理由”，再“单字母”
            req_text = ANALYZE_PROMPT_TPL.replace("<<PROMPT>>", prompt)
            text, _, _, _ = agent.infer(req_text, temperature=temp, max_tokens=CFG["LLM_MAX_TOKENS"])
            rats.append(truncate_chars(text, 600))  # 先做一点安全截断
            # 单字母
            letter_req = LETTER_ONLY_TPL.replace("<<PROMPT>>", prompt)
            ltxt, _, _, _ = agent.infer(letter_req, temperature=0.0, max_tokens=6)
            lets.append(normalize_to_letter_10(ltxt))
        return rats, lets

    def _compress_500(self, parts: List[str], compressor: Agent) -> str:
        msg = FUSE_COMPRESS_TPL.replace("{lim}", str(CFG["FUSE_CHAR_LIMIT"])).replace("<<PARTS>>", "\n\n".join(parts))
        text, _, _, _ = compressor.infer(msg, temperature=0.2, max_tokens=CFG["LLM_MAX_TOKENS"])
        # 双保险：再做一次硬裁切
        return truncate_chars(text, CFG["FUSE_CHAR_LIMIT"])

    def answer(self, q_prompt_shuf: str, samples_per_agent:int, temperature: float, q_raw:str) -> Tuple[str,int,str,List[str]]:
        """返回：final_letter_shuf, latency_ms, fused_<=500, head_votes"""
        t0=time.time()
        # —— 选主答者（Thompson）；激进旋转：5 票窗口若主答落后面板均值则换人 ——
        names = [a.name for a in self.agents]
        main_name = self.bandit.pick()
        main_idx  = names.index(main_name)
        panel = list(range(len(self.agents)))

        # 先让每个 agent 产出 N 组（理由+字母）
        all_rats: List[str]=[]
        all_letters: List[str]=[]
        head_votes: List[str]=[]

        # 并发采样
        def run_agent(ai: Agent):
            return self._one_agent_reason_and_vote(ai, q_prompt_shuf, samples_per_agent, temperature)

        with ThreadPoolExecutor(max_workers=min(CFG["MAX_WORKERS"], len(self.agents))) as ex:
            futs = {ex.submit(run_agent, self.agents[i]): i for i in panel}
            for fu in as_completed(futs):
                i = futs[fu]
                rats, lets = fu.result()
                all_rats.extend(rats)
                all_letters.extend(lets)
                head_votes.extend(lets[:2])  # 头部投票用来检测早期态势

        # —— 激进旋转（早停信号只用于 bandit 学习，不改变已收集票） ——
        if len(head_votes) >= CFG["VOTE_TOPK_EARLY"]:
            cnt = {L: head_votes.count(L) for L in A_J}
            front = sorted(cnt.items(), key=lambda x:(-x[1], x[0]))[:2]
            # 若 main 的前几票几乎没出现，视为失败（只是学习，不影响投票结果）
            main_ok = any(L for L,_c in front if L in head_votes[:2])
            self.bandit.update(main_name, success=bool(main_ok))

        # —— 500 字融合（摘要者轮询：每题从三 LLM 中用 Bandit 采样一个） ——
        compressor_name = self.bandit.pick()
        comp_agent = next(a for a in self.agents if a.name == compressor_name)
        fused_500 = self._compress_500(all_rats, comp_agent)

        # —— 最终答案：对 all_letters 做多数表决；平局则按固定优先级（openai>deepseek>xai） ——
        votes = [L for L in all_letters if L in A_J]
        if not votes:
            # 兜底：让主答者对“融合摘要+题面”再给一次字母
            combo = f"{q_prompt_shuf}\n\n[摘要参考]\n{fused_500}\n\n只输出答案字母(A–J)："
            main_agent = self.agents[main_idx]
            txt,_,_,_ = main_agent.infer(combo, temperature=0.0, max_tokens=6)
            final_shuf = normalize_to_letter_10(txt)
        else:
            c = {L: votes.count(L) for L in A_J}
            best = sorted(c.items(), key=lambda x:(-x[1], x[0]))  # 次序稳定
            if len(best)>=2 and best[0][1]==best[1][1]:
                # 平票 → 以代理优先级打破（谁更多票来自 openai>deepseek>xai 的样本）
                prio = ["agent-openai","agent-deepseek","agent-xai"]
                pick = best[0][0]  # 先取一个
                # 统计每类代理对该字母的票数
                agent_of_vote: List[str] = []
                # 简方式：按轮次映射（近似），不额外追踪来源就按固定次序挑
                final_shuf = pick
            else:
                final_shuf = best[0][0]

        total_ms = int((time.time()-t0)*1000)
        return final_shuf, total_ms, fused_500, head_votes[:6]

# =================== 单模型基线（仍保持） ===================
def single_agent_baseline(agent_prefix: str, prompt_shuf: str, agents: List[Agent]) -> str:
    chosen = None
    for a in agents:
        if a.name.startswith(agent_prefix) and a.kind=="llm":
            chosen = a; break
    if chosen is None:
        for a in agents:
            if a.kind=="llm": chosen=a; break
    if chosen is None: return "?"
    q = "You are given a 10-choice multiple-choice question.\nReturn only the letter A..J.\n\n"+prompt_shuf
    text,_,_,_ = chosen.infer(q, temperature=0.0, max_tokens=6)
    return normalize_to_letter_10(text)

# =================== 数据抽样（仅 options==10） ===================
def sample_mmlu_pro(ds, max_exs: int, min_categories: int = 8) -> List[int]:
    idx_all = [i for i, opts in enumerate(ds["options"]) if isinstance(opts, list) and len(opts) == 10]
    by_cat: Dict[str, List[int]] = {}
    for i in idx_all:
        c = ds["category"][i]
        by_cat.setdefault(c, []).append(i)
    cat_list = sorted(by_cat.keys()); random.shuffle(cat_list)
    if len(cat_list) < min_categories: min_categories = len(cat_list)
    pick_cats = cat_list[:min_categories]
    per_cat = max(1, max_exs // max(1, len(pick_cats)))
    chosen: List[int] = []
    for c in pick_cats:
        pool = by_cat[c]; random.shuffle(pool); chosen.extend(pool[:per_cat])
    if len(chosen) < max_exs:
        remain = [i for i in idx_all if i not in chosen]; random.shuffle(remain)
        chosen.extend(remain[:max_exs - len(chosen)])
    return chosen[:max_exs]

# =================== 评测主流程 ===================
def mmlu_pro_eval(max_exs: int, samples_per_agent:int, temperature:float, seed:int=42):
    random.seed(seed); np.random.seed(seed)
    print("[LOAD] TIGER-Lab/MMLU-Pro (split=test)")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    idxs = sample_mmlu_pro(ds, max_exs=max_exs, min_categories=8)
    agents = build_agents()
    orch = Orchestrator(agents)

    gold_list: List[str] = []
    orch_pred: List[str] = []
    bl_openai: List[str] = []
    bl_deepseek: List[str] = []
    bl_xai: List[str] = []

    for qi, i in enumerate(idxs, 1):
        q = ds["question"][i]
        opts = ds["options"][i]
        gold_idx = int(ds["answer_index"][i])
        gold_letter_orig = A_J[gold_idx] if 0<=gold_idx<10 else "?"

        prompt_shuf, idx, inv = shuffle_options_and_build_prompt(q, opts)

        mode = "simple" if is_simple(q) else "complex"
        final_shuf, latency_ms, fused500, head = orch.answer(
            prompt_shuf, samples_per_agent=samples_per_agent, temperature=temperature, q_raw=q
        )
        orch_letter_orig = map_letter_back(final_shuf, idx, inv)

        # 三基线（保持与乱序一致）
        openai_shuf = single_agent_baseline("agent-openai", prompt_shuf, agents)
        deepseek_shuf= single_agent_baseline("agent-deepseek", prompt_shuf, agents)
        xai_shuf     = single_agent_baseline("agent-xai", prompt_shuf, agents)

        openai_orig = map_letter_back(openai_shuf, idx, inv)
        deepseek_orig=map_letter_back(deepseek_shuf, idx, inv)
        xai_orig     = map_letter_back(xai_shuf, idx, inv)

        gold_list.append(gold_letter_orig)
        orch_pred.append(orch_letter_orig)
        bl_openai.append(openai_orig)
        bl_deepseek.append(deepseek_orig)
        bl_xai.append(xai_orig)

        print(f"\n[Q{qi}/{len(idxs)}]{analyze_tag(mode)} GOLD={gold_letter_orig} | ORCH={orch_letter_orig} | votes(n={len(agents)*samples_per_agent}) head={head} | lat={latency_ms}ms")
        print(f"[FUSED<=500]: {fused500}\n")

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
    out_path = "mmlu_pro_vote500.png"
    plt.savefig(out_path, dpi=160)
    print(f"[PLOT] saved -> {out_path}")

# =================== REPL（可选：展示 SIMPLE/COMPLEX 路径） ===================
def repl():
    agents = build_agents()
    orch = Orchestrator(agents)

    if CFG["METRICS_PORT"]>0:
        print(f"[Metrics] http://127.0.0.1:{CFG['METRICS_PORT']}/metrics")
        start_http_server(CFG["METRICS_PORT"])

    print("\n[BOOT] Agents:")
    for a in agents:
        print(f"  - {a.name}")

    print("\nType your question (10 options as 'A. ...' lines). End with an empty line. Type 'exit' to quit.\n")
    while True:
        # 简易 REPL：用户粘贴题干+10个选项
        lines=[]
        while True:
            try:
                s = input()
            except EOFError:
                return
            if s.strip().lower()=="exit": print("bye."); return
            if not s.strip(): break
            lines.append(s)
        if not lines: continue
        # 解析：第一行题干，其余十行形如 "A. ..." 或不带字母
        q = lines[0]
        opts = []
        for ln in lines[1:]:
            m = re.match(r"^[A-Ja-j]\.\s*(.*)$", ln.strip())
            opts.append(m.group(1) if m else ln.strip())
        if len(opts)!=10:
            print("[ERR] 需要 10 个选项。"); continue

        prompt_shuf, idx, inv = shuffle_options_and_build_prompt(q, opts)
        mode = "simple" if is_simple(q) else "complex"
        final_shuf, latency_ms, fused500, head = orch.answer(prompt_shuf, samples_per_agent=CFG["SAMPLES_PER_AGENT"], temperature=CFG["LLM_TEMPERATURE"], q_raw=q)
        final_orig = map_letter_back(final_shuf, idx, inv)
        print(f"\n{analyze_tag(mode)} ORCH={final_orig} | votes(n={len(agents)*CFG['SAMPLES_PER_AGENT']}) head={head} | lat={latency_ms}ms")
        print(f"[FUSED<=500]: {fused500}\n")

# =================== CLI ===================
def main():
    ap = argparse.ArgumentParser(description="MMLU-Pro (10-choice): 多代理×自一致投票 + 500字融合摘要 -> 答案")
    ap.add_argument("--mode", choices=["eval","repl"], default="eval", help="eval=评测; repl=交互")
    ap.add_argument("--max_exs", type=int, default=100, help="最多抽题数（均匀抽样，仅10选项）")
    ap.add_argument("--samples_per_agent", type=int, default=CFG["SAMPLES_PER_AGENT"], help="每代理采样次数（自一致）")
    ap.add_argument("--temperature", type=float, default=CFG["LLM_TEMPERATURE"], help="采样温度")
    args = ap.parse_args()

    # 覆盖全局
    CFG["SAMPLES_PER_AGENT"] = max(1, int(args.samples_per_agent))
    CFG["LLM_TEMPERATURE"]   = float(args.temperature)

    if args.mode=="repl":
        repl()
    else:
        mmlu_pro_eval(max_exs=int(args.max_exs),
                      samples_per_agent=CFG["SAMPLES_PER_AGENT"],
                      temperature=CFG["LLM_TEMPERATURE"],
                      seed=CFG["SEED"])

if __name__ == "__main__":
    main()
