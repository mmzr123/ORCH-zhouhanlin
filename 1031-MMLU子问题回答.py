# mmlu_router_allinone_sc_vote.py
# ---------------------------------------------------------------------------------
# Router（复杂=自一致性 + 多 agent 加权投票 + 多轮尝试） vs 三条单模型基线（OpenAI/DeepSeek/XAI）
# - 复杂：同一题 -> 每个 agent 做 SC 多采样 -> agent 内表决 -> 跨 agent 加权投票 -> 若不确定，多轮继续
# - 基线：不走编排器；简单模式（单次调用），并行执行，加速
# - 控制台逐题输出：gold、Router(复杂)、OpenAI基线、DeepSeek基线、XAI基线
# - 单张图表比较四路 Accuracy
# 依赖: pip install datasets numpy matplotlib scikit-learn requests
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, random, argparse, re, collections
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== deps =====
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

import requests

# ============================== Keys & Provider ==============================
# 用环境变量注入；留空则该 agent 走 echo（便于离线调试）
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
    "openai":   {"in": 0.005, "out": 0.015},
    "deepseek": {"in": 0.002, "out": 0.004},
    "grok":     {"in": 0.005, "out": 0.010},
    "echo":     {"in": 0.000, "out": 0.000},
}

SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Think step by step internally, but only output the final choice letter.\n"
    "2) Return exactly one of A, B, C, or D.\n"
    "3) Be specific, verifiable and concise.\n"
    "4) If info is missing, make the minimum reasonable assumption and state it.\n"
)

CFG: Dict[str, Any] = {
    # ---- Agents ----
    "OPENAI_COMPAT_PROFILES": [
        {"name":"openai","base_url":OAI_BASE,"api_key":OAI_KEY,"model":OAI_MODEL,"enabled":bool(OAI_KEY)},
        {"name":"deepseek","base_url":DEEPSEEK_BASE,"api_key":DEEPSEEK_KEY,"model":DEEPSEEK_MODEL,"enabled":bool(DEEPSEEK_KEY)},
    ],
    "XAI":  {"api_key":XAI_KEY, "model":XAI_MODEL, "base_url":XAI_BASE, "enabled":bool(XAI_KEY)},

    # ---- LLM call ----
    "TIMEOUT_S": 30,
    "MAX_TOKENS": 128,

    # ---- Orchestrator ----
    "SC_SAMPLES": int(os.getenv("SC_SAMPLES", "3")),  # 每 agent 自一致性采样次数
    "ROUNDS":     int(os.getenv("ROUNDS", "2")),      # 最大轮数
    "ROUND_TEMPS": [0.2, 0.6, 0.9],                   # 各轮温度（可短于/长于 ROUNDS；会自动截断/循环）
    "MIN_ACCEPT_WEIGHT": 1.6,                         # 接受阈值：加权总票数 >= 此值（经验值）
    "TIE_EPS": 0.05,                                  # 加权差小于该阈值视作平票 -> 进入下一轮
}

random.seed(42)
np.random.seed(42)

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
def normalize_to_letter(text: str) -> str:
    if not text: return "?"
    m = LETTER_RE.search(text.strip())
    if m: return m.group(1).upper()
    t = text.strip().upper()
    for L in ["A","B","C","D"]:
        if re.match(rf"^\s*{L}\b", t): return L
    return "?"

def clip_text(t: str, n: int=400) -> str:
    return t if len(t)<=n else (t[:n] + " …")

# ======================== Agent Impl =========================
@dataclass
class AgentResult:
    letter: str
    raw_text: str
    http: int
    latency_ms: int
    tokens_in: Optional[int]=None
    tokens_out: Optional[int]=None
    cost_per_ktok: float=0.0
    error: Optional[str]=None

class BaseAgent:
    def __init__(self, name: str, kind: str="llm"):
        self.name = name
        self.kind = kind
        self.session = requests.Session()

    def _cost(self, provider: str, ti: int|None, to: int|None) -> float:
        p = PRICE.get(provider, {"in":0.0,"out":0.0})
        return ((p["in"]*(ti or 0) + p["out"]*(to or 0)) / 1000.0)

    def _call_openai_compat(self, base_url: str, api_key: str, model: str,
                            prompt: str, temperature: float, max_tokens: int,
                            provider_key: str) -> AgentResult:
        t0 = time.time()
        try:
            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
            payload = {
                "model": model,
                "messages":[{"role":"system","content": SYSTEM_PROMPT},
                            {"role":"user","content": prompt}],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
            resp = self.session.post(url, headers=headers, json=payload, timeout=CFG["TIMEOUT_S"])
            dur = int((time.time()-t0)*1000)
            data = resp.json() if resp.content else {}
            text = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip() or json.dumps(data)[:1000]
            usage = data.get("usage",{})
            ti, to = usage.get("prompt_tokens"), usage.get("completion_tokens")
            return AgentResult(normalize_to_letter(text), text, resp.status_code, dur, ti, to, self._cost(provider_key, ti, to))
        except Exception as e:
            dur = int((time.time()-t0)*1000)
            return AgentResult("?", f"[{self.name} ERROR] {type(e).__name__}: {e}", 500, dur, error=str(e))

class OpenAICompatAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str, provider_key: str):
        super().__init__(name); self.base=base_url; self.key=api_key; self.model=model; self.provider_key=provider_key
    def infer(self, prompt: str, temperature: float, max_tokens:int) -> AgentResult:
        if not self.key:
            # echo 模式（无 key 也能跑流程）
            fake = f"[{self.name}] (echo) " + prompt[:120]
            return AgentResult("?", fake, 200, 50, tokens_out=len(fake)//4, cost_per_ktok=0.0)
        return self._call_openai_compat(self.base, self.key, self.model, prompt, temperature, max_tokens, self.provider_key)

class XaiGrokAgent(BaseAgent):
    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        super().__init__(name); self.base=base_url; self.key=api_key; self.model=model
    def infer(self, prompt: str, temperature: float, max_tokens:int) -> AgentResult:
        if not self.key:
            fake = f"[{self.name}] (echo) " + prompt[:120]
            return AgentResult("?", fake, 200, 60, tokens_out=len(fake)//4, cost_per_ktok=0.0)
        return self._call_openai_compat(self.base, self.key, self.model, prompt, temperature, max_tokens, "grok")

# ======================== Build Agents =========================
def _build_agents() -> List[BaseAgent]:
    agents: List[BaseAgent] = []
    for p in CFG["OPENAI_COMPAT_PROFILES"]:
        if p["enabled"]:
            provider_key = p["name"] if p["name"] in PRICE else "openai"
            agents.append(OpenAICompatAgent(f"agent-{p['name']}", p["base_url"], p["api_key"], p["model"], provider_key))
    if CFG["XAI"]["enabled"]:
        agents.append(XaiGrokAgent("agent-xai", CFG["XAI"]["base_url"], CFG["XAI"]["api_key"], CFG["XAI"]["model"]))
    if not agents:  # 全部无钥匙 -> 至少给一个 echo
        agents.append(OpenAICompatAgent("agent-openai", OAI_BASE, "", OAI_MODEL, "openai"))
    return agents

# ======================== Self-Consistency ======================
def sc_vote_one_agent(agent: BaseAgent, prompt: str, samples: int, temperature: float, max_tokens: int) -> Tuple[str, float, List[str]]:
    """
    返回 (agent_consensus_letter, confidence(0..1), raw_texts)
    """
    letters: List[str] = []
    raws: List[str] = []
    for _ in range(max(1, samples)):
        r = agent.infer(prompt, temperature=temperature, max_tokens=max_tokens)
        letters.append(r.letter)
        raws.append(r.raw_text)
    cnt = collections.Counter(letters)
    if not cnt:
        return "?", 0.0, raws
    best_letter, v = cnt.most_common(1)[0]
    conf = v / len(letters)
    return best_letter, float(conf), raws

# ======================== Multi-Agent Weighted Vote =============
def weighted_vote(letter_confs: List[Tuple[str,float,str]]) -> Tuple[str, float, Dict[str,float]]:
    """
    输入: [(letter, conf, agent_name), ...]
    输出: (final_letter, final_margin, weight_by_letter)
         - final_margin: 第一名与第二名加权差
    """
    weights: Dict[str,float] = {"A":0.0,"B":0.0,"C":0.0,"D":0.0}
    for L, conf, _ in letter_confs:
        if L in weights:
            weights[L] += max(0.0, conf)
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    topL, topW = ranked[0]
    secW = ranked[1][1] if len(ranked)>1 else 0.0
    margin = topW - secW
    return topL, margin, weights

# ======================== Router Orchestrator ===================
def orchestrator_answer_all_rounds(agents: List[BaseAgent], prompt: str,
                                   rounds: int, sc_samples: int,
                                   round_temps: List[float],
                                   min_accept_weight: float,
                                   tie_eps: float,
                                   max_tokens:int=CFG["MAX_TOKENS"]) -> Tuple[str, Dict[str,Any]]:
    """
    多轮：
      - 每轮：每个 agent 做 SC -> 跨 agent 加权投票
      - 接受条件：
         1) 胜者累计权重 >= min_accept_weight 且
         2) 胜者与次名差距 > tie_eps
      - 否则：下一轮（温度提高/变化），直到 rounds 用尽
      - 最终：若仍冲突，取最后一轮的胜者字母
    """
    logs = []
    temps = (round_temps or [0.2])
    if rounds <= 0: rounds = 1
    for r in range(rounds):
        T = temps[r] if r < len(temps) else temps[-1]
        per_agent: List[Tuple[str,float,str]] = []
        per_agent_raw: Dict[str,List[str]] = {}
        print(f"  [ROUND {r+1}/{rounds}] T={T}")
        for ag in agents:
            L, conf, raws = sc_vote_one_agent(ag, prompt, sc_samples, T, max_tokens)
            per_agent.append((L, conf, ag.name))
            per_agent_raw[ag.name] = raws
            print(f"    - {ag.name}: SC -> {L} (conf={conf:.2f})")

        final, margin, weights = weighted_vote(per_agent)
        total_w = sum(weights.values())
        logs.append({"round": r+1, "T": T, "per_agent": per_agent, "weights": weights, "margin": margin, "total_w": total_w, "final": final})
        print(f"    = vote -> {final}, weights={weights}, margin={margin:.2f}, total_w={total_w:.2f}")

        if (total_w >= min_accept_weight) and (margin > tie_eps) and final in "ABCD":
            return final, {"rounds_used": r+1, "log": logs}

    # 走到这里：用最后一轮的胜者（容忍轻微平票）
    last = logs[-1]
    return last["final"], {"rounds_used": len(logs), "log": logs}

# ======================== Baselines (simple) ====================
def baselines_simple_parallel(agents: List[BaseAgent], prompt: str,
                              max_tokens:int=32, temperature:float=0.3) -> Dict[str, str]:
    wants = ["agent-openai", "agent-deepseek", "agent-xai"]
    out: Dict[str,str] = { "agent-openai":"?", "agent-deepseek":"?", "agent-xai":"?" }
    futmap = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        for a in agents:
            if any(a.name.startswith(w) for w in wants):
                futmap[ex.submit(a.infer, prompt, temperature, max_tokens)] = a.name
        for fu in as_completed(futmap):
            name = futmap[fu]
            try:
                r = fu.result()
                out[name] = r.letter
            except Exception:
                out[name] = "?"
    return out

# ======================== Eval Loop =============================
def orchestrator_answer(agents: List[BaseAgent], prompt: str,
                        rounds:int, sc_samples:int) -> str:
    letter, meta = orchestrator_answer_all_rounds(
        agents, prompt,
        rounds=rounds,
        sc_samples=sc_samples,
        round_temps=CFG["ROUND_TEMPS"],
        min_accept_weight=CFG["MIN_ACCEPT_WEIGHT"],
        tie_eps=CFG["TIE_EPS"],
        max_tokens=CFG["MAX_TOKENS"]
    )
    # 你也可以在此打印 meta["log"] 进行更详细的调试
    return letter

def mmlu_eval(subject: str, split: str, max_exs: int, rounds:int, sc_samples:int):
    ASK_TEMPLATE = (
        "You are given a multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        "Question: {q}\n"
        "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        "Answer:"
    )
    print(f"[LOAD] cais/mmlu, subject={subject}, split={split}")
    ds = load_dataset("cais/mmlu", subject, split=split)
    if max_exs and len(ds) > max_exs:
        ds = ds.select(range(max_exs))

    router_agents = _build_agents()
    baseline_agents = _build_agents()  # 独立实例，避免潜在会话耦合

    gold_list, orch_pred = [], []
    openai_pred, deepseek_pred, xai_pred = [], [], []

    t0 = time.time()
    for i, ex in enumerate(ds):
        q = ex["question"]; choices = ex["choices"]; gold_idx = int(ex["answer"])
        if len(choices) < 4:
            print(f"[SKIP {i}] choices < 4"); continue
        prompt = ASK_TEMPLATE.format(q=q, A=choices[0], B=choices[1], C=choices[2], D=choices[3])

        # —— Router（复杂，自一致性 + 多 agent 投票 + 多轮尝试）
        try:
            o_ans = orchestrator_answer(router_agents, prompt, rounds=rounds, sc_samples=sc_samples)
        except Exception as e:
            o_ans = "?"
            print(f"[WARN][Router] Q{i} error: {e}")

        # —— 三条单模型（基线）并行・简单模式（不经过编排器）
        b_out = baselines_simple_parallel(baseline_agents, prompt, max_tokens=32, temperature=0.3)

        gold_letter = ["A","B","C","D"][gold_idx]
        gold_list.append(gold_letter)
        orch_pred.append(o_ans)
        openai_pred.append(b_out.get("agent-openai","?"))
        deepseek_pred.append(b_out.get("agent-deepseek","?"))
        xai_pred.append(b_out.get("agent-xai","?"))

        q_short = (q[:120]+"...") if len(q) > 120 else q
        print(f"[Q{i+1}/{len(ds)}] GOLD={gold_letter} | ROUTER={o_ans} | OPENAI={openai_pred[-1]} | DEEPSEEK={deepseek_pred[-1]} | XAI={xai_pred[-1]}")
        print(f"  Question: {q_short}")

    dur = time.time() - t0
    print(f"\n[DONE] N={len(gold_list)}, time={dur:.1f}s")

    def acc(ys, ps):
        return sum(1 for y,p in zip(ys,ps) if y==p) / max(1, len(ys))

    acc_orch = acc(gold_list, orch_pred)
    acc_oai  = acc(gold_list, openai_pred)   if any(p!="?" for p in openai_pred)   else None
    acc_dpk  = acc(gold_list, deepseek_pred) if any(p!="?" for p in deepseek_pred) else None
    acc_xai  = acc(gold_list, xai_pred)      if any(p!="?" for p in xai_pred)      else None

    print(f"[ACC] Router(复杂)    = {acc_orch:.3f}")
    print(f"[ACC] Baseline(OpenAI)   = {acc_oai:.3f}"   if acc_oai is not None else "[ACC] Baseline(OpenAI)   = N/A (agent-openai 未启用)")
    print(f"[ACC] Baseline(DeepSeek) = {acc_dpk:.3f}"   if acc_dpk is not None else "[ACC] Baseline(DeepSeek) = N/A (agent-deepseek 未启用)")
    print(f"[ACC] Baseline(XAI)      = {acc_xai:.3f}"   if acc_xai is not None else "[ACC] Baseline(XAI)      = N/A (agent-xai 未启用)")

    # —— 单张图：四路 Accuracy 对比（不可用的基线不画）
    names, vals = ["Router"], [acc_orch]
    if acc_oai is not None: names.append("OpenAI");   vals.append(acc_oai)
    if acc_dpk is not None: names.append("DeepSeek"); vals.append(acc_dpk)
    if acc_xai is not None: names.append("XAI");      vals.append(acc_xai)

    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(1,1,1)
    ax1.bar(names, vals)
    ax1.set_ylim(0,1.0)
    ax1.set_title(f"MMLU Accuracy ({subject}, N={len(gold_list)})")
    ax1.set_ylabel("Accuracy")
    fig.tight_layout()
    out_path = f"mmlu_acc_compare_{subject}.png"
    plt.savefig(out_path, dpi=140)
    print(f"[PLOT] saved -> {out_path}")

# ======================== CLI =============================
def main():
    ap = argparse.ArgumentParser(description="MMLU: Router(自一致性+多agent投票+多轮) vs 三基线")
    ap.add_argument("--mode", choices=["eval"], default="eval")
    ap.add_argument("--subject", default="abstract_algebra")
    ap.add_argument("--split", default="test", choices=["test","dev","validation"])
    ap.add_argument("--max_exs", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=CFG["ROUNDS"], help="多轮尝试次数")
    ap.add_argument("--sc", type=int, default=CFG["SC_SAMPLES"], help="每 agent 自一致性采样数")
    args = ap.parse_args()

    if args.mode == "eval":
        mmlu_eval(args.subject, args.split, args.max_exs, rounds=args.rounds, sc_samples=args.sc)

if __name__ == "__main__":
    main()
