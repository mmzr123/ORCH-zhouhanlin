# -*- coding: utf-8 -*-
# mmlu_router_4choice_multi.py
# -----------------------------------------------------------------------------
# 4 选一 MMLU，多学科评测：
# - ORCH：三 Agent 并行分析（≤500 字）+ 合并 Agent（不再固定 XAI，使用第一个 LLM）
# - 三条基线：OPENAI / DEEPSEEK / XAI 单次直接作答
# - 默认：10 个学科 × 每科 30 题 ≈ 300 题
# - 输出：
#   * 全局 ACC（ORCH + 三基线）
#   * 分学科 ACC
#   * ORCH 的 4×4 混淆矩阵（A–D）
#   * McNemar 显著性检验（ORCH vs DEEPSEEK）
#   * 图像全部保存在 plots_4choice/ 目录下：
#       - global_acc.png
#       - orch_confusion.png
#       - per_subject_orch_vs_deepseek.png
# 依赖：pip install datasets numpy matplotlib scikit-learn requests
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, math, argparse, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# =============== 基本配置 ===============
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

SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Answer directly.\n"
    "2) Do NOT ask the the user for more info.\n"
    "3) No templates or meta-commentary.\n"
    "4) Be specific, verifiable and concise.\n"
    "5) If info is missing, make the minimum reasonable assumption and state it.\n"
)

CFG: Dict[str, Any] = {
    "LLM_TEMPERATURE": 0.0,        # 固定采样，避免随机波动
    "LLM_MAX_TOKENS": 900,         # 足够容纳 ≤500 字分析
    "TIMEOUT_S": 60,
    "ANSWER_MAX_CHARS": 500,
    "SEED": 42,
}

random.seed(CFG["SEED"])
np.random.seed(CFG["SEED"])

LABELS_4 = ["A", "B", "C", "D"]

# 10 个默认学科（可根据需要修改）
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

PLOT_DIR = "plots_4choice"
os.makedirs(PLOT_DIR, exist_ok=True)

# =============== 工具函数 ===============
def clip_text(t: str, n: int = 1000) -> str:
    t = t or ""
    return t if len(t) <= n else t[:n] + " ...[TRUNCATED]"

def truncate_chars(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]

LETTER4_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def normalize_to_letter_4(text: str) -> str:
    """
    将模型输出严格归一到 A/B/C/D，失败返回 '?'。
    """
    if not text:
        return "?"
    t = (text or "").strip()

    # 优先匹配独立 A-D
    m = LETTER4_RE.search(t)
    if m:
        return m.group(1).upper()

    # 再尝试匹配以 A-D 开头的行
    for line in t.splitlines():
        line = line.strip().upper()
        m2 = re.match(r"^([ABCD])\b", line)
        if m2:
            return m2.group(1)

    return "?"

def approx_tokens_from_text(text: str) -> int:
    return max(1, int(len(text) / 4))

# =============== Agent 抽象 ===============
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

@dataclass
class OrchestrateRequest:
    query: str
    context: Dict[str, Any] = field(default_factory=dict)

def print_agent_result(prefix: str, res: AgentResult):
    print(f"{prefix} -> (latency={res.latency_ms}ms, http={res.http_status})")
    print("TEXT:")
    print(clip_text(res.result_text))
    print("-" * 86)

# =============== HTTP LLM 基类 ===============
class BaseAgent:
    def __init__(self, name: str, kind: str = "llm"):
        self.name = name
        self.kind = kind

    def request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], float]:
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
        dur = (time.time() - t0) * 1000.0
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        return resp.status_code, data, dur

    def infer_fail(self, req: OrchestrateRequest, err: str, latency_ms: int = 1200) -> AgentResult:
        text = f"[{self.name} ERROR] {err}\nEcho: {req.query[:200]}"
        return AgentResult(text, int(latency_ms), 500,
                           error=err,
                           tokens_in=0,
                           tokens_out=approx_tokens_from_text(text))

class OpenAICompatAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        super().__init__(name, kind="llm")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def infer(self, req: OrchestrateRequest) -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                res = AgentResult(text, 50, 200, None, 0, approx_tokens_from_text(text))
                print_agent_result(f"CALL {self.name}", res)
                return res

            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            if not text:
                text = json.dumps(data)[:2000]
            usage = data.get("usage", {})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            res = AgentResult(f"[{self.name}] {text}", int(dur), code, None, ti, to)
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

    def infer(self, req: OrchestrateRequest) -> AgentResult:
        try:
            if not self.api_key:
                text = f"[{self.name}] (echo) {req.query[:480]}"
                res = AgentResult(text, 50, 200, None, 0, approx_tokens_from_text(text))
                print_agent_result(f"CALL {self.name}", res)
                return res

            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.query},
                ],
                "temperature": CFG["LLM_TEMPERATURE"],
                "max_tokens": CFG["LLM_MAX_TOKENS"],
            }
            code, data, dur = self.request(url, headers, payload)
            text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
            if not text:
                text = json.dumps(data)[:2000]
            usage = data.get("usage", {})
            ti = usage.get("prompt_tokens")
            to = usage.get("completion_tokens")
            res = AgentResult(f"[{self.name}] {text}", int(dur), code, None, ti, to)
            print_agent_result(f"CALL {self.name}", res)
            return res
        except Exception as e:
            res = self.infer_fail(req, f"{type(e).__name__}: {e}")
            print_agent_result(f"CALL {self.name} (ERROR)", res)
            return res

class EchoAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent-echo", kind="local")

    def infer(self, req: OrchestrateRequest) -> AgentResult:
        text = f"[agent-echo] Echo: {req.query[:200]}"
        res = AgentResult(text, 20, 200, None, 0, approx_tokens_from_text(text))
        print_agent_result("CALL agent-echo", res)
        return res

# =============== 构建 Agent 列表 ===============
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

# =============== Orchestrator：三并行分析 + 合并（不固定 XAI） ===============
class Orchestrator:
    """
    用第一个 LLM 作为 dispatcher + merger，不固定 XAI。
    panel Agent 优先选择 openai / deepseek / xai 三个。
    """
    def __init__(self, agents: List[Agent]):
        self.agents = [a for a in agents if getattr(a, "kind", "") in ("llm", "local")]

    def _decompose(self, dispatcher: Agent, prompt: str) -> List[str]:
        tpl = (
            "将下面的四选一题拆分为3个互补的分析子问题，覆盖：概念判别、选项排除、推理核验，"
            "每条不超过40字。只输出三行，每行一个子问题，无其它内容。\n\n"
            f"{prompt}\n"
        )
        res = dispatcher.infer(OrchestrateRequest(tpl))
        lines = [ln.strip(" -•\t") for ln in res.result_text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[:3]
        # 兜底
        return [
            "定义与关键概念是否正确？",
            "逐个选项排除其不成立原因？",
            "综合前两步验证最优选项是否与题干一致？",
        ]

    def _one_agent_analysis(self, agent: Agent, full_prompt: str, subq: str) -> AgentResult:
        ana_tpl = (
            "请围绕以下四选一多项选择题进行结构化分析，严格控制在500字以内；"
            "按【要点】列出依据与排除逻辑，最后给出“暂定答案(仅A/B/C/D中的一个字母)”。\n"
            f"子问题：{subq}\n\n"
            f"{full_prompt}\n"
        )
        q = ana_tpl
        res = agent.infer(OrchestrateRequest(q))
        trimmed = truncate_chars(res.result_text, CFG["ANSWER_MAX_CHARS"])
        return AgentResult(trimmed, res.latency_ms, res.http_status,
                           res.error, res.tokens_in, res.tokens_out)

    def _merge(self, analyses: List[Tuple[str, str]], full_prompt: str) -> str:
        # 合并 Agent：不固定 XAI，使用第一个 LLM（通常是 agent-openai）
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
            # 兜底 echo
            return "?", int((time.time() - t0) * 1000)

        dispatcher = llms[0]
        subqs = self._decompose(dispatcher, prompt)

        # panel：优先 openai / deepseek / xai
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

# =============== 单 Agent 基线 ===============
def single_agent_baseline(agent_name_prefix: str, prompt: str, agents: List[Agent]) -> str:
    chosen = None
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
        return "?"

    q = (
        "You are given a 4-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        f"{prompt}\n"
    )
    res = chosen.infer(OrchestrateRequest(q))
    return normalize_to_letter_4(res.result_text)

# =============== 多学科 4 选一 MMLU 评测 ===============
def mmlu_eval_multi_subject(subjects: List[str],
                            split: str = "test",
                            per_subject: int = 30,
                            seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"[CFG] subjects={subjects}, per_subject={per_subject}, split={split}")

    agents = build_agents()
    orch = Orchestrator(agents)

    # 按学科存预测
    per_subj_stats = {
        s: {"gold": [], "orch": [], "openai": [], "deepseek": [], "xai": []}
        for s in subjects
    }

    # 全局
    gold_all: List[str] = []
    orch_all: List[str] = []
    open_all: List[str] = []
    deep_all: List[str] = []
    xai_all:  List[str] = []

    for subj in subjects:
        print(f"\n[LOAD] cais/mmlu, subject={subj}, split={split}")
        ds = load_dataset("cais/mmlu", subj, split=split)

        valid_idxs = [i for i, ex in enumerate(ds) if isinstance(ex["choices"], list) and len(ex["choices"]) >= 4]
        if not valid_idxs:
            print(f"[WARN] subject={subj} has no 4-choice questions, skip.")
            continue

        if len(valid_idxs) > per_subject:
            idxs = list(np.random.choice(valid_idxs, size=per_subject, replace=False))
        else:
            idxs = valid_idxs

        print(f"[INFO] subject={subj} use {len(idxs)} examples.")

        for k, i in enumerate(idxs, 1):
            ex = ds[i]
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = int(ex["answer"])
            if gold_idx < 0 or gold_idx > 3:
                continue

            prompt = (
                f"Question: {q}\n"
                f"Options:\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
            )

            orch_ans, latency_ms = orch.orchestrator_answer(prompt)
            openai_ans   = single_agent_baseline("agent-openai",  prompt, agents)
            deepseek_ans = single_agent_baseline("agent-deepseek", prompt, agents)
            xai_ans      = single_agent_baseline("agent-xai",      prompt, agents)

            gold_letter = LABELS_4[gold_idx]
            orch_ans    = normalize_to_letter_4(orch_ans)
            openai_ans  = normalize_to_letter_4(openai_ans)
            deepseek_ans= normalize_to_letter_4(deepseek_ans)
            xai_ans     = normalize_to_letter_4(xai_ans)

            gold_all.append(gold_letter)
            orch_all.append(orch_ans)
            open_all.append(openai_ans)
            deep_all.append(deepseek_ans)
            xai_all.append(xai_ans)

            per_subj_stats[subj]["gold"].append(gold_letter)
            per_subj_stats[subj]["orch"].append(orch_ans)
            per_subj_stats[subj]["openai"].append(openai_ans)
            per_subj_stats[subj]["deepseek"].append(deepseek_ans)
            per_subj_stats[subj]["xai"].append(xai_ans)

            print(f"[{subj} Q{k}/{len(idxs)}] GOLD={gold_letter} | ORCH={orch_ans} | "
                  f"OPENAI={openai_ans} | DEEPSEEK={deepseek_ans} | XAI={xai_ans} | lat={latency_ms}ms")

    # ===== 汇总 ACC =====
    def acc(ys, ps):
        return sum(1 for y, p in zip(ys, ps) if y == p) / max(1, len(ys))

    acc_orch = acc(gold_all, orch_all)
    acc_open = acc(gold_all, open_all)
    acc_deep = acc(gold_all, deep_all)
    acc_xai  = acc(gold_all, xai_all)

    print("\n[GLOBAL ACC]")
    print(f"  ORCH     = {acc_orch:.3f}")
    print(f"  OPENAI   = {acc_open:.3f}")
    print(f"  DEEPSEEK = {acc_deep:.3f}")
    print(f"  XAI      = {acc_xai:.3f}")

    # ===== 分学科 ACC =====
    print("\n[PER-SUBJECT ACC]")
    rows = []
    for subj in subjects:
        g = per_subj_stats[subj]["gold"]
        if not g:
            continue
        ao   = acc(g, per_subj_stats[subj]["orch"])
        aope = acc(g, per_subj_stats[subj]["openai"])
        adeep= acc(g, per_subj_stats[subj]["deepseek"])
        ax   = acc(g, per_subj_stats[subj]["xai"])
        rows.append((subj, ao, aope, adeep, ax))
        print(f"  {subj:25s}  ORCH={ao:.3f}  OPENAI={aope:.3f}  DEEPSEEK={adeep:.3f}  XAI={ax:.3f}")

    # ===== ORCH 混淆矩阵 =====
    cm = confusion_matrix(gold_all, orch_all, labels=LABELS_4)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    print("\n[CONFUSION MATRIX ORCH (raw counts)]")
    print("     " + "  ".join(LABELS_4))
    for i, lab in enumerate(LABELS_4):
        print(f"{lab}: " + "  ".join(f"{cm[i, j]:3d}" for j in range(len(LABELS_4))))

    # ===== McNemar：ORCH vs DEEPSEEK =====
    b = c = 0  # b: ORCH 正确 / DEEPSEEK 错误；c: ORCH 错误 / DEEPSEEK 正确
    for y, o, d in zip(gold_all, orch_all, deep_all):
        if o == "?" or d == "?":
            continue
        if o == y and d != y:
            b += 1
        elif o != y and d == y:
            c += 1
    n = b + c
    if n == 0:
        p_mcnemar = 1.0
        print("\n[MCNEMAR] no discordant pairs, p=1.0 (无法判断差异)")
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n  # continuity correction
        z = math.sqrt(chi2)
        Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        p_mcnemar = 2 * (1 - Phi(z))
        print("\n[MCNEMAR] ORCH vs DEEPSEEK")
        print(f"  b = ORCH correct & DEEPSEEK wrong = {b}")
        print(f"  c = ORCH wrong  & DEEPSEEK correct = {c}")
        print(f"  b+c = {n}, chi2 = {chi2:.4f}, p ≈ {p_mcnemar:.4g}")
        if p_mcnemar < 0.05:
            print("  ==> 差异在 95% 置信水平下显著 (p < 0.05)")
        else:
            print("  ==> 差异未达到 95% 显著水平 (p ≥ 0.05)")

    # ===== 画图：全部放在 plots_4choice/ 目录，分开保存 =====

    # (1) ORCH vs 三基线 ACC 柱状图
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.bar(["ORCH", "OPENAI", "DEEPSEEK", "XAI"],
            [acc_orch, acc_open, acc_deep, acc_xai])
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Global Accuracy (4-choice MMLU)")
    fig1.tight_layout()
    out1 = os.path.join(PLOT_DIR, "global_acc.png")
    fig1.savefig(out1, dpi=160)
    print(f"[PLOT] saved -> {out1}")

    # (2) ORCH 混淆矩阵热力图
    fig2 = plt.figure(figsize=(4.5, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    im = ax2.imshow(cm_norm, interpolation="nearest")
    ax2.set_xticks(range(len(LABELS_4)))
    ax2.set_yticks(range(len(LABELS_4)))
    ax2.set_xticklabels(LABELS_4)
    ax2.set_yticklabels(LABELS_4)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Gold")
    ax2.set_title("ORCH Confusion Matrix (A–D)")
    for i in range(len(LABELS_4)):
        for j in range(len(LABELS_4)):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    out2 = os.path.join(PLOT_DIR, "orch_confusion.png")
    fig2.savefig(out2, dpi=160)
    print(f"[PLOT] saved -> {out2}")

    # (3) 分学科 ORCH vs DEEPSEEK 柱状图（看增益）
    fig3 = plt.figure(figsize=(10, 4.5))
    ax3 = fig3.add_subplot(1, 1, 1)
    subj_names = [r[0] for r in rows]
    orch_subj  = [r[1] for r in rows]
    deep_subj  = [r[3] for r in rows]  # (subj, ao, aope, adeep, ax)
    x = np.arange(len(subj_names))
    width = 0.35
    ax3.bar(x - width / 2, deep_subj, width, label="DEEPSEEK")
    ax3.bar(x + width / 2, orch_subj, width, label="ORCH")
    ax3.set_xticks(x)
    ax3.set_xticklabels(subj_names, rotation=60, ha="right", fontsize=8)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Per-subject Acc: ORCH vs DEEPSEEK")
    ax3.legend()
    fig3.tight_layout()
    out3 = os.path.join(PLOT_DIR, "per_subject_orch_vs_deepseek.png")
    fig3.savefig(out3, dpi=160)
    print(f"[PLOT] saved -> {out3}")

# =============== 简单 REPL（可选） ===============
def repl():
    agents = build_agents()
    print("\n[BOOT] Agents:")
    for a in agents:
        print(f"  - {a.name} (kind={getattr(a, 'kind', '?')})")
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

# =============== CLI ===============
def main():
    ap = argparse.ArgumentParser(
        description="4-choice MMLU multi-subject eval: ORCH (3-agent analyses + merge) + 3 baselines"
    )
    ap.add_argument("--mode", choices=["eval", "repl"], default="eval",
                    help="eval=多学科评测; repl=交互直答")
    ap.add_argument("--per_subject", type=int, default=30,
                    help="每个学科抽多少题 (默认30)")
    args = ap.parse_args()

    if args.mode == "repl":
        repl()
    else:
        mmlu_eval_multi_subject(SUBJECTS_10, split="test",
                                per_subject=args.per_subject,
                                seed=CFG["SEED"])

if __name__ == "__main__":
    main()
