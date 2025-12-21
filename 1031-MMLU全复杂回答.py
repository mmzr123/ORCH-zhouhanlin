# mmlu_router_merge_baseline.py
# ---------------------------------------------------------------------------------
# 单轮复杂流程（固定为复杂）：三 Agent 作答 → 合并器一票定音（A/B/C/D）
# + 三条单模型基线（OpenAI / DeepSeek / XAI）
# 依赖: pip install datasets numpy matplotlib scikit-learn requests
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, argparse, re, collections
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== third-party =====
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

# ============================== Keys & Endpoints ==============================
# 用环境变量，未配置则自动禁用某 agent（脚本会优雅降级）
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

# ============================== Config =======================================
SYSTEM_SOLVE = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Think step by step internally but output explicitly one of A, B, C, or D.\n"
    "2) Return the letter on the first line, then one short rationale sentence on the second line.\n"
    "3) Be concise and factual; no meta.\n"
)
SYSTEM_MERGE = (
    "You are the merger for multiple-choice questions.\n"
    "Given the question, options, and 3 agent votes with rationales, do:\n"
    "1) Note consensus vs. disagreement briefly.\n"
    "2) Output ONLY a single letter A/B/C/D as the final answer on the LAST line.\n"
    "No extra boilerplate."
)

ASK_TEMPLATE = (
    "You are given a multiple-choice question. Pick the single correct option.\n"
    "Return only the letter A, B, C, or D (first line), then one short rationale sentence (second line).\n\n"
    "Question: {q}\n"
    "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
)

MERGE_TEMPLATE = (
    "Question: {q}\n"
    "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Agent votes with rationales:\n"
    " - {a1_name}: {a1_vote} | {a1_reason}\n"
    " - {a2_name}: {a2_vote} | {a2_reason}\n"
    " - {a3_name}: {a3_vote} | {a3_reason}\n\n"
    "Now produce the final single letter (A/B/C/D) as the last line."
)

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def normalize_letter(text: str) -> str:
    if not text: return "?"
    # 优先首行匹配
    first = text.strip().splitlines()[0] if text.strip() else ""
    m = LETTER_RE.search(first)
    if m: return m.group(1).upper()
    # 退化：全文找第一个 A-D
    m2 = LETTER_RE.search(text.strip())
    return m2.group(1).upper() if m2 else "?"


def split_vote_and_reason(text: str) -> Tuple[str, str]:
    if not text: return "?", ""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    vote = normalize_letter(lines[0] if lines else text)
    reason = ""
    if len(lines) >= 2:
        reason = lines[1][:220]
    return vote, reason


# ============================== Agents =======================================
class ChatAgent:
    def __init__(self, name: str, base_url: str, api_key: str, model: str, provider_key: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider_key  # openai / deepseek / xai

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 256,
              timeout: int = 60) -> Tuple[int, Dict[str, Any], float]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        t0 = time.time()
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        dur = (time.time() - t0) * 1000.0
        data = resp.json() if resp.content else {}
        return resp.status_code, data, dur

    def solve_once(self, prompt: str, system_prompt: str = SYSTEM_SOLVE, temperature: float = 0.2,
                   max_tokens: int = 256) -> Tuple[str, str, int]:
        if not self.enabled:
            # echo fallback
            out = f"A\necho({self.name})"
            return out, "echo", 80
        code, data, dur = self._post(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens
        )
        text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if not text:
            text = json.dumps(data)[:2000]
        return text, ("ok" if code == 200 else f"http{code}"), int(dur)

    def merge_once(self, prompt: str, system_prompt: str = SYSTEM_MERGE, temperature: float = 0.1,
                   max_tokens: int = 128) -> Tuple[str, str, int]:
        # 使用此 agent 作为合并裁决者
        if not self.enabled:
            # echo fallback：简单多数
            votes = re.findall(r"\b([ABCD])\b", prompt, flags=re.I)
            if votes:
                # 多数
                cnt = collections.Counter([v.upper() for v in votes])
                pick = cnt.most_common(1)[0][0]
                return pick, "echo-merge", 10
            return "A", "echo-merge", 10
        code, data, dur = self._post(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens
        )
        text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if not text:
            text = json.dumps(data)[:2000]
        return text, ("ok" if code == 200 else f"http{code}"), int(dur)


# 装配可用 agents
def build_agents() -> List[ChatAgent]:
    agents: List[ChatAgent] = []
    if OAI_KEY:
        agents.append(ChatAgent("agent-openai", OAI_BASE, OAI_KEY, OAI_MODEL, "openai"))
    if DEEPSEEK_KEY:
        agents.append(ChatAgent("agent-deepseek", DEEPSEEK_BASE, DEEPSEEK_KEY, DEEPSEEK_MODEL, "deepseek"))
    if XAI_KEY:
        agents.append(ChatAgent("agent-xai", XAI_BASE, XAI_KEY, XAI_MODEL, "xai"))
    # 如果一个都没有，则给一个 echo（禁用真实接口）
    if not agents:
        agents.append(ChatAgent("agent-echo", "https://example.invalid", "", "echo", "echo"))
    return agents


# ============================== Orchestrator（固定复杂） ======================
@dataclass
class VoteRecord:
    agent: str
    raw_text: str
    vote: str
    reason: str
    latency_ms: int
    ok: bool


def orchestrate_one_question(agents: List[ChatAgent], q: str, A: str, B: str, C: str, D: str,
                             merge_agent: Optional[ChatAgent] = None,
                             temperature: float = 0.2) -> Tuple[str, List[VoteRecord], int]:
    """
    三 Agent 并行作答 -> 合并 Agent 输出最终 A/B/C/D
    返回：final_letter, 明细, 组合延迟
    """
    if not agents:
        return "?", [], 0
    # 选择三个作答 agent（如果不足 3，就用现有）
    solve_agents = agents[:3]
    # 合并者：优先选 openai；否则用第一个可用；再不行 echo 合并
    merge_agent = merge_agent or next((a for a in agents if a.name.startswith("agent-openai")), agents[0])

    ask = ASK_TEMPLATE.format(q=q, A=A, B=B, C=C, D=D)
    votes: List[VoteRecord] = []
    total_ms = 0

    with ThreadPoolExecutor(max_workers=len(solve_agents)) as ex:
        futs = {ex.submit(a.solve_once, ask, SYSTEM_SOLVE, temperature, 192): a for a in solve_agents}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                text, status, dur = fut.result()
                v, r = split_vote_and_reason(text)
                votes.append(VoteRecord(a.name, text, v, r, dur, status == "ok"))
                total_ms += dur
            except Exception as e:
                votes.append(VoteRecord(a.name, f"[ERROR] {e}", "?", "", 0, False))

    # 构建合并提示
    def _safe(idx: int) -> Tuple[str, str, str]:
        if idx < len(votes):
            return votes[idx].agent, votes[idx].vote, votes[idx].reason
        return f"agent-{idx + 1}", "?", ""

    a1_name, a1_vote, a1_reason = _safe(0)
    a2_name, a2_vote, a2_reason = _safe(1)
    a3_name, a3_vote, a3_reason = _safe(2)

    merge_prompt = MERGE_TEMPLATE.format(q=q, A=A, B=B, C=C, D=D,
                                         a1_name=a1_name, a1_vote=a1_vote, a1_reason=a1_reason,
                                         a2_name=a2_name, a2_vote=a2_vote, a2_reason=a2_reason,
                                         a3_name=a3_name, a3_vote=a3_vote, a3_reason=a3_reason)
    merged_text, m_status, m_dur = merge_agent.merge_once(merge_prompt, SYSTEM_MERGE, 0.1, 64)
    total_ms += m_dur
    final_letter = normalize_letter(merged_text)
    # 若合并失败（无字母），回退为三票多数
    if final_letter == "?":
        cnt = collections.Counter([v.vote for v in votes if v.vote in "ABCD"])
        if cnt:
            final_letter = cnt.most_common(1)[0][0]
        else:
            final_letter = "A"
    return final_letter, votes, total_ms


# ============================== Baselines ====================================
def baseline_simple(agent: ChatAgent, q: str, A: str, B: str, C: str, D: str,
                    temperature: float = 0.2) -> str:
    ask = ASK_TEMPLATE.format(q=q, A=A, B=B, C=C, D=D)
    text, status, dur = agent.solve_once(ask, SYSTEM_SOLVE, temperature, 192)
    return normalize_letter(text)


# ============================== Eval Loop (MMLU) =============================
def mmlu_eval(subject: str, split: str, max_exs: int,
              temperature: float = 0.2, seed: int = 42):
    random.seed(seed);
    np.random.seed(seed)

    print(f"[LOAD] cais/mmlu, subject={subject}, split={split}")
    ds = load_dataset("cais/mmlu", subject,
                      split=split)  # Hugging Face 数据集（cais/mmlu）:contentReference[oaicite:2]{index=2}
    if max_exs and len(ds) > max_exs:
        ds = ds.select(range(max_exs))

    agents = build_agents()
    print(f"[AGENTS] enabled = {[a.name for a in agents if a.enabled]} (total={len(agents)})")

    # 合并者优先 openai
    merge_agent = next((a for a in agents if a.name.startswith("agent-openai")), agents[0])

    gold_list, orch_pred = [], []
    openai_pred, deepseek_pred, xai_pred = [], [], []

    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex["question"];
        choices = ex["choices"];
        gold_idx = int(ex["answer"])
        if len(choices) < 4:
            print(f"[SKIP {i}] choices < 4");
            continue
        A, B, C, D = choices[0], choices[1], choices[2], choices[3]

        # 复杂：三 agent + 合并者
        o_ans, votes, total_ms = orchestrate_one_question(agents, q, A, B, C, D, merge_agent, temperature)
        orch_pred.append(o_ans)

        # 三条基线（若可用）
        def find_agent(prefix: str) -> Optional[ChatAgent]:
            for a in agents:
                if a.name.startswith(prefix): return a
            return None

        # OpenAI
        ag = find_agent("agent-openai")
        if ag and ag.enabled:
            openai_pred.append(baseline_simple(ag, q, A, B, C, D, temperature))
        # DeepSeek
        ag = find_agent("agent-deepseek")
        if ag and ag.enabled:
            deepseek_pred.append(baseline_simple(ag, q, A, B, C, D, temperature))
        # xAI
        ag = find_agent("agent-xai")
        if ag and ag.enabled:
            xai_pred.append(baseline_simple(ag, q, A, B, C, D, temperature))

        gold_letter = ["A", "B", "C", "D"][gold_idx]
        gold_list.append(gold_letter)

        # 控制台逐题紧凑输出
        print(f"[Q{i + 1}/{len(ds)}] GOLD={gold_letter} | ORCH={o_ans} "
              f"| OPENAI={(openai_pred[-1] if len(openai_pred) == len(gold_list) else 'NA')} "
              f"| DEEPSEEK={(deepseek_pred[-1] if len(deepseek_pred) == len(gold_list) else 'NA')} "
              f"| XAI={(xai_pred[-1] if len(xai_pred) == len(gold_list) else 'NA')} "
              f"| lat={total_ms}ms")

    dur = time.time() - t0
    print(f"\n[DONE] N={len(gold_list)}, time={dur:.1f}s")

    def acc(ys, ps):
        return sum(1 for y, p in zip(ys, ps) if y == p) / max(1, len(ys))

    acc_orch = acc(gold_list, orch_pred)

    # 注意：基线长度可能 < gold_list（当对应 agent 未启用时跳过），下方仅在非空时显示
    def safe_acc(ps):
        return acc(gold_list[:len(ps)], ps) if ps else None

    acc_oai = safe_acc(openai_pred)
    acc_dpk = safe_acc(deepseek_pred)
    acc_xai = safe_acc(xai_pred)

    print(f"[ACC] Orchestrator (3-agent + merger) = {acc_orch:.3f}")
    print(
        f"[ACC] Baseline(OpenAI)   = {acc_oai:.3f}" if acc_oai is not None else "[ACC] Baseline(OpenAI)   = N/A (agent-openai 未启用)")
    print(
        f"[ACC] Baseline(DeepSeek) = {acc_dpk:.3f}" if acc_dpk is not None else "[ACC] Baseline(DeepSeek) = N/A (agent-deepseek 未启用)")
    print(
        f"[ACC] Baseline(XAI)      = {acc_xai:.3f}" if acc_xai is not None else "[ACC] Baseline(XAI)      = N/A (agent-xai 未启用)")

    # —— 单张图：四路 Accuracy 对比（不可用的不画）
    names, vals = ["Orchestrator"], [acc_orch]
    if acc_oai is not None: names.append("OpenAI");   vals.append(acc_oai)
    if acc_dpk is not None: names.append("DeepSeek"); vals.append(acc_dpk)
    if acc_xai is not None: names.append("XAI");      vals.append(acc_xai)

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(names, vals)
    ax1.set_ylim(0, 1.0)
    ax1.set_title(f"MMLU Accuracy ({subject}, N={len(gold_list)})")
    ax1.set_ylabel("Accuracy")
    fig.tight_layout()
    out_path = f"mmlu_acc_compare_{subject}.png"
    plt.savefig(out_path, dpi=140)
    print(f"[PLOT] saved -> {out_path}")


# ============================== CLI ==========================================
def main():
    ap = argparse.ArgumentParser(description="MMLU: 3-agent complex (merge to A/B/C/D) vs single-agent baselines")
    ap.add_argument("--subject", default="abstract_algebra", help="MMLU 子学科（如 anatomy, abstract_algebra 等）")
    ap.add_argument("--split", default="test", choices=["test", "dev", "validation"], help="数据划分")
    ap.add_argument("--max_exs", type=int, default=120, help="最多抽取题目数")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    args = ap.parse_args()
    mmlu_eval(args.subject, args.split, args.max_exs, temperature=args.temperature)


if __name__ == "__main__":
    main()
