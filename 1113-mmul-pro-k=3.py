# -*- coding: utf-8 -*-
"""
MMLU-Pro (10-choice, 10 dirs × per_dir)
- OPENAI / DEEPSEEK / XAI 三基线
- VOTE (三家投票)
- ORCH：EMA 路由 + 条件自洽（仅合并器阶段，k 次）
- 输出 6 张图到文件夹：修改EMA-MMUL-PRO-1110
- EMA 状态文件：ema_state_mmlu_pro_1110.json
"""

from __future__ import annotations
import os, time, math, random, re, collections, json
from typing import List, Dict, Tuple, Optional

# ===== 依赖 =====
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
from requests.exceptions import ReadTimeout, RequestException

# ========== 路径 ==========
PLOTS_DIR = "修改EMA-MMUL-PRO-1113"
os.makedirs(PLOTS_DIR, exist_ok=True)
EMA_STATE_PATH = "ema_state_mmlu_pro_1113.json"

# ========== API 配置（从环境变量读取，留空将使用本地 echo）==========
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE      = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c181d17907ff4d848eca8225244aa67f")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")

XAI_API_KEY      = os.getenv("XAI_API_KEY", "xai-iTEf2LYg5Jvd4YIHmSmihv7gQWEv5XkiEKuzRwQ4lqBLwUsl4tndgGSnVsn6DCzBa3gRgTl0Wuuowag9")
XAI_MODEL        = os.getenv("XAI_MODEL", "grok-2-latest")
XAI_BASE         = os.getenv("XAI_BASE", "https://api.x.ai/v1")


SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Answer directly.\n"
    "2) Do NOT ask the user for more info.\n"
    "3) No templates or meta-commentary.\n"
    "4) Be specific, verifiable and concise.\n"
    "5) If info is missing, make the minimum reasonable assumption and state it.\n"
)

# ========== EMA 配置 ==========
EMA_AGENT_NAMES  = ["OPENAI", "DEEPSEEK", "XAI"]
EMA_ALPHA_QUALITY   = 0.2
EMA_ALPHA_LATENCY   = 0.2
EMA_ALPHA_COST      = 0.2
EMA_ALPHA_STABILITY = 0.2

PRICE_PER_M = {
    "OPENAI": {"currency": "USD", "input": 0.15, "output": 0.60},
    "DEEPSEEK": {"currency": "CNY", "input": 2.0, "output": None},
    "XAI": {"currency": "USD", "input": 2.0, "output": 10.0},
}

def init_empty_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    st: Dict[str, Dict[str, Optional[float]]] = {}
    for n in agent_names:
        st[n] = {
            "ema_quality": None,
            "ema_latency_ms": None,
            "ema_cost": None,
            "ema_stability": None,
            # 可选：ema_quality_var
        }
    return st

def load_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not os.path.exists(EMA_STATE_PATH):
        s = init_empty_ema_state(agent_names)
        with open(EMA_STATE_PATH, "w", encoding="utf-8") as f: json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"[EMA] created: {EMA_STATE_PATH}")
        return s
    try:
        with open(EMA_STATE_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
        if not isinstance(s, dict): raise ValueError("bad json")
    except Exception as e:
        print(f"[EMA] load error ({e}), re-init.")
        s = {}
    for n in agent_names:
        if n not in s or not isinstance(s[n], dict):
            s[n] = {}
        for k in ["ema_quality", "ema_latency_ms", "ema_cost", "ema_stability"]:
            s[n].setdefault(k, None)
    return s

def save_ema_state(state: Dict[str, Dict[str, Optional[float]]]):
    with open(EMA_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"[EMA] saved -> {EMA_STATE_PATH}")

def ema_update(old: Optional[float], x: Optional[float], alpha: float) -> Optional[float]:
    if x is None: return old
    if old is None: return float(x)
    return float(alpha * x + (1.0 - alpha) * old)

def calc_call_cost(agent_name: str, ti: Optional[int], to: Optional[int]) -> Optional[float]:
    if ti is None and to is None: return None
    cfg = PRICE_PER_M.get(agent_name);
    if cfg is None: return None
    ti = ti or 0; to = to or 0
    ip = cfg.get("input"); op = cfg.get("output") if cfg.get("output") is not None else cfg.get("input", 0.0)
    if ip is None and op is None: return None
    if ip is None: ip = 0.0
    return float((ti * ip + to * op) / 1e6)

def update_agent_ema(ema_state, agent_name, correct, latency_ms, cost, stability_ok):
    if agent_name not in ema_state: return
    st = ema_state[agent_name]
    if correct is not None:
        st["ema_quality"] = ema_update(st.get("ema_quality"), 1.0 if correct else 0.0, EMA_ALPHA_QUALITY)
    if latency_ms is not None:
        st["ema_latency_ms"] = ema_update(st.get("ema_latency_ms"), float(latency_ms), EMA_ALPHA_LATENCY)
    if cost is not None:
        st["ema_cost"] = ema_update(st.get("ema_cost"), float(cost), EMA_ALPHA_COST)
    if stability_ok is not None:
        st["ema_stability"] = ema_update(st.get("ema_stability"), 1.0 if stability_ok else 0.0, EMA_ALPHA_STABILITY)

# ========== A–J 归一 ==========
A_J = [chr(ord("A") + i) for i in range(10)]
LETTER10_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)

def normalize_to_letter_10(text: str) -> str:
    if not text: return "?"
    m = LETTER10_RE.search(text.strip())
    if m: return m.group(1).upper()
    t = (text or "").strip().upper()
    for L in A_J:
        if re.match(rf"^\s*{L}\b", t): return L
    return "?"

# ========== 通用 Chat ==========
def call_chat_api(base_url: str, api_key: str, model: str,
                  prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, int, Optional[int], Optional[int], int]:
    if not api_key:
        fake = "[ECHO] " + prompt[:120]
        return fake, 1000, None, None, 200
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=40)
        dur_ms = int((time.time() - t0) * 1000)
    except ReadTimeout:
        return "[TIMEOUT]", int((time.time() - t0) * 1000), None, None, 0
    except RequestException as e:
        return f"[REQUEST_ERROR] {e}", int((time.time() - t0) * 1000), None, None, 0

    status = resp.status_code
    ti = to = None
    try:
        data = resp.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        usage = data.get("usage", {})
        ti = usage.get("prompt_tokens"); to = usage.get("completion_tokens")
    except Exception:
        text = f"[HTTP {resp.status_code}] {resp.text[:200]}"
    return text, dur_ms, ti, to, status

# ========== 基线 ==========
def ask_mcq_10choice(provider: str, prompt: str):
    tpl = (
        "You are given a 10-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, D, E, F, G, H, I, or J.\n\n{q}\n"
    )
    q = tpl.format(q=prompt)
    if provider == "openai":
        raw, lat, ti, to, st = call_chat_api(OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL, q, max_tokens=16, temperature=0.0)
    elif provider == "deepseek":
        raw, lat, ti, to, st = call_chat_api(DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, q, max_tokens=16, temperature=0.0)
    elif provider == "xai":
        raw, lat, ti, to, st = call_chat_api(XAI_BASE, XAI_API_KEY, XAI_MODEL, q, max_tokens=16, temperature=0.0)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return normalize_to_letter_10(raw), lat, ti, to, st

# ========== EMA 路由 ==========
def _coerce_float(x, default):
    try:
        if x is None:
            return default
        xf = float(x)
        if math.isfinite(xf):
            return xf
        return default
    except Exception:
        return default

def ema_routing_key(agent_name: str,
                    ema_state: Dict[str, Dict[str, Optional[float]]]) -> Tuple[float, float, float, float]:
    """
    返回用于排序的 key：(-quality, latency, cost, -stability)
    冷启动或坏值时使用安全默认：
      quality=0.5, latency=2000ms, cost=1.0, stability=0.5
    """
    st = ema_state.get(agent_name, {})
    q    = _coerce_float(st.get("ema_quality"),     0.5)
    lat  = _coerce_float(st.get("ema_latency_ms"), 2000.0)
    cost = _coerce_float(st.get("ema_cost"),          1.0)
    stab = _coerce_float(st.get("ema_stability"),     0.5)
    return (-q, lat, cost, -stab)


# ========== 共识 & 自洽 工具 ==========
def _extract_letter(text: str) -> Optional[str]:
    if not text: return None
    m = LETTER10_RE.search(text.strip())
    return m.group(1).upper() if m else None

def _majority(votes: List[str]) -> Optional[str]:
    votes = [v for v in votes if v in A_J]
    if not votes: return None
    top, n = collections.Counter(votes).most_common(1)[0]
    return top if n >= 2 else None

# ========== ORCH：EMA 路由 + 条件自洽（仅合并器） ==========
def orch_answer(prompt: str,
                ema_state: Dict[str, Dict[str, Optional[float]]],
                sc_k: int = 3,
                sc_temp: float = 0.3,
                lcb_z: float = 1.0,
                lcb_gate: float = 0.62) -> Tuple[str, int]:
    # 候选
    candidates = []
    if OPENAI_API_KEY:   candidates.append(("OPENAI",   OPENAI_BASE,   OPENAI_API_KEY,   OPENAI_MODEL))
    if DEEPSEEK_API_KEY: candidates.append(("DEEPSEEK", DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL))
    if XAI_API_KEY:      candidates.append(("XAI",      XAI_BASE,      XAI_API_KEY,      XAI_MODEL))
    if not candidates: return "?", 0

    print("\n[ROUTER] candidate scores (before routing):")
    for name, *_ in candidates:
        st = ema_state.get(name, {})
        key = ema_routing_key(name, ema_state)
        print(f"  - {name:8s} | ema_quality={st.get('ema_quality')} | ema_latency_ms={st.get('ema_latency_ms')} | "
              f"ema_cost={st.get('ema_cost')} | ema_stability={st.get('ema_stability')} | routing_key={key}")

    candidates_sorted = sorted(candidates, key=lambda it: ema_routing_key(it[0], ema_state))
    print("[ROUTER] sorted agents (best first):")
    for r,(n, *_rest) in enumerate(candidates_sorted,1): print(f"  rank {r}: {n}")

    analysis_agents = candidates_sorted[:3]
    merger = candidates_sorted[0]
    print(f"[ROUTER] analysis={ [a[0] for a in analysis_agents] }, merger={merger[0]}")

    # 三段分析（温度0）
    ana_tpl = (
        "请围绕以下十选一多项选择题进行结构化分析，严格控制在300字以内；"
        "按【要点】列出依据与排除逻辑，最后给出'暂定答案(仅A~J中的一个字母)'。\n\n{q}\n"
    )
    ana_prompt = ana_tpl.format(q=prompt)
    analyses, total_lat = [], 0
    for name, base, key, model in analysis_agents:
        txt, lat, *_ = call_chat_api(base, key, model, ana_prompt, max_tokens=512, temperature=0.0)
        analyses.append((name, txt)); total_lat += lat

    # 共识过滤
    draft = [_extract_letter(t) for _,t in analyses]
    maj = _majority([d for d in draft if d])
    if maj: return maj, total_lat

    # LCB 门控
    best_name = merger[0]
    st = ema_state.get(best_name, {}) if ema_state else {}
    q   = st.get("ema_quality", 0.5)
    var = st.get("ema_quality_var", None)
    std = (math.sqrt(var) if (var is not None and var >= 0.0) else 0.25)
    lcb = float(q) - float(lcb_z) * float(std)
    print(f"[ROUTER] merger={best_name} LCB={lcb:.3f} (q={q:.3f}, std={std:.3f}, z={lcb_z})")

    m_name, m_base, m_key, m_model = merger
    bullet = "\n\n".join([f"[{n}] {t}" for n,t in analyses])
    merge_tpl = (
        "你将看到多名分析员对同一十选一题的分析（均≤500字）。"
        "请综合证据，仅输出最终选项字母（A/B/C/D/E/F/G/H/I/J），不得附加解释或其它字符。\n\n"
        f"{prompt}\n\n{bullet}\n\n最终答案："
    )

    if lcb >= lcb_gate:
        # 把握足：单次合并
        raw, lat, *_ = call_chat_api(m_base, m_key, m_model, merge_tpl, max_tokens=16, temperature=0.0)
        total_lat += lat
        return normalize_to_letter_10(raw), total_lat

    # 无共识 & 低把握：对合并器做 k 次采样（轻度随机）
    letters = []
    for _ in range(sc_k):
        raw, lat, *_ = call_chat_api(m_base, m_key, m_model, merge_tpl, max_tokens=16, temperature=float(sc_temp))
        total_lat += lat
        letters.append(normalize_to_letter_10(raw))
    maj2 = _majority(letters)
    if maj2: return maj2, total_lat
    for L in letters:
        if L in A_J: return L, total_lat
    return "?", total_lat

# ========== VOTE ==========
def vote_AJ(open_ans: str, deep_ans: str, xai_ans: str) -> str:
    votes = {"OPENAI": open_ans, "DEEPSEEK": deep_ans, "XAI": xai_ans}
    c = collections.Counter(votes.values())
    if not c: return "?"
    letter, n = c.most_common(1)[0]
    if n >= 2: return letter
    for name in ["OPENAI","DEEPSEEK","XAI"]:
        cand = votes[name]
        if cand in A_J: return cand
    return "?"

# ========== 采样 ==========
def sample_mmlu_pro_10dirs(ds, per_dir: int = 30, seed: int = 42):
    random.seed(seed)
    idx_all = [i for i, opts in enumerate(ds["options"]) if isinstance(opts, list) and len(opts)==10]
    by_cat: Dict[str,List[int]] = {}
    for i in idx_all:
        by_cat.setdefault(ds["category"][i], []).append(i)
    cats_sorted = sorted(by_cat.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(cats_sorted) < 10:
        raise RuntimeError(f"可用 category < 10，仅有：{[c for c,_ in cats_sorted]}")
    chosen_cats = [cats_sorted[i][0] for i in range(10)]
    print("[INFO] 选中的 10 个方向(category)：")
    for c in chosen_cats:
        print(f"  - {c} (samples with 10 options = {len(by_cat[c])})")
    indices: List[int] = []
    for c in chosen_cats:
        pool = by_cat[c][:]
        random.shuffle(pool)
        indices.extend(pool[:per_dir])
    return indices, chosen_cats

# ========== 统计 ==========
def mcnemar_stats(gold: List[str], pred_a: List[str], pred_b: List[str]) -> Dict[str,float]:
    assert len(gold) == len(pred_a) == len(pred_b)
    b = c = 0
    for y, pa, pb in zip(gold, pred_a, pred_b):
        if y not in A_J: continue
        ca = (pa == y); cb = (pb == y)
        if ca and (not cb): b += 1
        elif (not ca) and cb: c += 1
    bc = b + c
    if bc == 0: return {"b":b,"c":c,"b_plus_c":0,"chi2":0.0,"p":1.0}
    chi2 = (b - c) ** 2 / float(bc)
    p = math.erfc(math.sqrt(chi2/2.0))
    return {"b":b,"c":c,"b_plus_c":bc,"chi2":chi2,"p":p}

# ========== 画图（6张）==========
def plot_global_acc(global_acc: Dict[str,float]):
    models = ["ORCH","OPENAI","DEEPSEEK","XAI","VOTE"]
    accs = [global_acc[m] for m in models]
    plt.figure(figsize=(6,4)); plt.bar(models, accs); plt.ylim(0,1.0)
    plt.ylabel("Accuracy"); plt.title("Global Accuracy (MMLU-Pro, 10 dirs × per_dir Q)")
    for i,v in enumerate(accs): plt.text(i, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); out = os.path.join(PLOTS_DIR,"global_acc.png"); plt.savefig(out, dpi=160); plt.close()
    print(f"[PLOT] {out}")

def plot_per_direction_acc(dir_names: List[str], per_dir_acc: Dict[str,Dict[str,float]]):
    models = ["ORCH","OPENAI","DEEPSEEK","XAI","VOTE"]
    x = np.arange(len(dir_names)); width = 0.16
    plt.figure(figsize=(10,5))
    for j,m in enumerate(models):
        ys = [per_dir_acc[d][m] for d in dir_names]
        plt.bar(x+(j-2)*width, ys, width, label=m)
    plt.ylim(0,1.0); plt.xticks(x, dir_names, rotation=35, ha="right")
    plt.ylabel("Accuracy"); plt.title("Per-Direction Accuracy (10 dirs, per_dir Q each)")
    plt.legend(); plt.tight_layout(); out = os.path.join(PLOTS_DIR,"per_direction_acc.png")
    plt.savefig(out, dpi=160); plt.close(); print(f"[PLOT] {out}")

def plot_orch_confusion(golds: List[str], preds_orch: List[str]):
    labels = A_J; cm = confusion_matrix(golds, preds_orch, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig = plt.figure(figsize=(6.5,5)); ax = fig.add_subplot(1,1,1)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); ax.set_title("Confusion Matrix (ORCH, A–J)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,f"{cm_norm[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    out = os.path.join(PLOTS_DIR,"orch_confusion.png"); plt.savefig(out, dpi=160); plt.close()
    print(f"[PLOT] {out}")

def plot_mcnemar_text(best_baseline_name: str, stats: Dict[str,float]):
    b=stats["b"]; c=stats["c"]; bc=stats["b_plus_c"]; chi2=stats["chi2"]; p=stats["p"]
    txt = (f"McNemar Test: ORCH vs {best_baseline_name}\n\n"
           f"b = ORCH correct & {best_baseline_name} wrong = {b}\n"
           f"c = ORCH wrong  & {best_baseline_name} correct = {c}\n"
           f"b + c = {bc}\nchi2 = {chi2:.4f}\np ≈ {p:.6f}\n\n"
           f"Conclusion: {'significant (p < 0.05)' if p < 0.05 else 'not significant (p ≥ 0.05)'}")
    fig, ax = plt.subplots(figsize=(6,4)); ax.axis("off")
    ax.text(0.01,0.99,txt,va="top",ha="left",fontsize=10,family="monospace")
    fig.tight_layout(); out = os.path.join(PLOTS_DIR,"mcnemar_orch_vs_best.png")
    plt.savefig(out, dpi=160); plt.close(); print(f"[PLOT] {out}")

def plot_price_latency_table(avg_latency: Dict[str,float]):
    header = ["Model","In_cached","In_noncached","Out","Avg latency (ms)"]
    rows = [
        ["ORCH (EMA-router)","≈ 3×(O + D + X)","-","-", f"{avg_latency.get('ORCH', float('nan')):.1f}"],
        ["VOTE (3-model)","≈ O + D + X","-","-", f"{avg_latency.get('VOTE', float('nan')):.1f}"],
        ["DEEPSEEK","0.2 元 / 1M","2 元 / 1M","-", f"{avg_latency.get('DEEPSEEK', float('nan')):.1f}"],
        ["GPT-4o-mini","$0.15 / 1M","$0.075 / 1M","$0.60 / 1M", f"{avg_latency.get('OPENAI', float('nan')):.1f}"],
        ["XAI (grok)","$2.00 / 1M","-","$10.00 / 1M", f"{avg_latency.get('XAI', float('nan')):.1f}"],
    ]
    fig, ax = plt.subplots(figsize=(8.5,3.0)); ax.axis("off")
    table = ax.table(cellText=[header]+rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,1.4)
    ax.set_title("Price & Average Latency", pad=10); fig.tight_layout()
    out = os.path.join(PLOTS_DIR,"price_latency_table.png")
    plt.savefig(out, dpi=160); plt.close(); print(f"[PLOT] {out}")

def plot_ema_quality_curve(ema_traj: Dict[str, Dict[str, List[Optional[float]]]]):
    plt.figure(figsize=(7,4))
    for agent_name in EMA_AGENT_NAMES:
        ys = ema_traj.get(agent_name, {}).get("ema_quality", [])
        ys_plot = [np.nan if v is None else v for v in ys]
        xs = list(range(1, len(ys_plot)+1))
        if xs: plt.plot(xs, ys_plot, label=agent_name)
    plt.xlabel("Question Index"); plt.ylabel("EMA Quality (recent accuracy)")
    plt.title("EMA Quality vs Question Index"); plt.ylim(0.0, 1.05); plt.legend()
    plt.tight_layout(); out = os.path.join(PLOTS_DIR,"ema_quality_curve.png")
    plt.savefig(out, dpi=160); plt.close(); print(f"[PLOT] {out}")

# ========== 交互式参数 ==========
def _input_int(prompt_txt: str, default_val: int) -> int:
    s = input(f"{prompt_txt} (默认 {default_val}): ").strip()
    if s == "": return default_val
    try: return int(s)
    except:
        print("  无效输入，使用默认。"); return default_val

def _input_float(prompt_txt: str, default_val: float) -> float:
    s = input(f"{prompt_txt} (默认 {default_val}): ").strip()
    if s == "": return default_val
    try: return float(s)
    except:
        print("  无效输入，使用默认。"); return default_val

# ========== 主流程 ==========
def main():
    print("=== 参数设置（回车采用默认值）===")
    per_dir  = _input_int  ("每个方向抽取题目数 per_dir", 30)
    seed     = _input_int  ("随机种子 seed", 42)
    sc_k     = _input_int  ("合并器自洽采样次数 sc_k", 3)
    sc_temp  = _input_float("合并器自洽温度 sc_temp", 0.3)
    lcb_z    = _input_float("LCB z 系数 lcb_z", 1.0)
    lcb_gate = _input_float("LCB 触发阈值 lcb_gate", 0.62)
    print(f"[CONFIG] per_dir={per_dir}, seed={seed}, sc_k={sc_k}, sc_temp={sc_temp}, lcb_z={lcb_z}, lcb_gate={lcb_gate}")
    print("\n[LOAD] TIGER-Lab/MMLU-Pro (split=test)")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    random.seed(seed); np.random.seed(seed)

    # EMA 状态 & 轨迹
    ema_state = load_ema_state(EMA_AGENT_NAMES)
    ema_traj: Dict[str, Dict[str, List[Optional[float]]]] = {n: {"ema_quality": []} for n in EMA_AGENT_NAMES}

    indices, dir_names = sample_mmlu_pro_10dirs(ds, per_dir=per_dir, seed=seed)
    print(f"[INFO] 总样本数 ≈ {len(indices)} (10 dirs × {per_dir})\n")

    gold_all: List[str]  = []
    open_all: List[str]  = []
    deep_all: List[str]  = []
    xai_all:  List[str]  = []
    vote_all: List[str]  = []
    orch_all: List[str]  = []

    total_lat_open = total_lat_deep = total_lat_xai = 0.0
    total_lat_orch = total_lat_vote = 0.0
    n_calls_open = n_calls_deep = n_calls_xai = 0
    n_calls_orch = n_calls_vote = 0

    per_dir_gold: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_open: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_deep: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_xai:  Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_vote: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_orch: Dict[str, List[str]] = {d: [] for d in dir_names}

    for qi, i in enumerate(indices, 1):
        q = ds["question"][i]; opts = ds["options"][i]
        gold_idx = int(ds["answer_index"][i]); gold_letter = A_J[gold_idx] if 0 <= gold_idx < 10 else "?"
        cat = ds["category"][i]
        prompt = "Question: " + str(q).strip() + "\nOptions:\n" + "\n".join([f"{A_J[k]}. {opts[k]}" for k in range(10)])

        # 三基线
        open_ans, lat_o, ti_o, to_o, st_o = ask_mcq_10choice("openai",  prompt)
        deep_ans, lat_d, ti_d, to_d, st_d = ask_mcq_10choice("deepseek", prompt)
        xai_ans,  lat_x, ti_x, to_x, st_x = ask_mcq_10choice("xai",      prompt)
        vote_ans = vote_AJ(open_ans, deep_ans, xai_ans)

        # ORCH
        orch_ans, lat_orch = orch_answer(prompt, ema_state, sc_k=sc_k, sc_temp=sc_temp, lcb_z=lcb_z, lcb_gate=lcb_gate)

        # 记录
        gold_all.append(gold_letter); open_all.append(open_ans); deep_all.append(deep_ans)
        xai_all.append(xai_ans); vote_all.append(vote_ans); orch_all.append(orch_ans)

        if cat in dir_names:
            per_dir_gold[cat].append(gold_letter)
            per_dir_open[cat].append(open_ans)
            per_dir_deep[cat].append(deep_ans)
            per_dir_xai[cat].append(xai_ans)
            per_dir_vote[cat].append(vote_ans)
            per_dir_orch[cat].append(orch_ans)

        # 延迟
        total_lat_open += lat_o; total_lat_deep += lat_d; total_lat_xai += lat_x
        total_lat_orch += lat_orch; lat_vote = lat_o + lat_d + lat_x; total_lat_vote += lat_vote
        n_calls_open += 1; n_calls_deep += 1; n_calls_xai += 1; n_calls_orch += 1; n_calls_vote += 1

        # EMA 更新
        cost_open = calc_call_cost("OPENAI",   ti_o, to_o)
        cost_deep = calc_call_cost("DEEPSEEK", ti_d, to_d)
        cost_xai  = calc_call_cost("XAI",      ti_x, to_x)
        stab_open = (st_o == 200); stab_deep = (st_d == 200); stab_xai = (st_x == 200)
        correct_open = (open_ans == gold_letter) if (gold_letter in A_J and open_ans in A_J) else False
        correct_deep = (deep_ans == gold_letter) if (gold_letter in A_J and deep_ans in A_J) else False
        correct_xai  = (xai_ans  == gold_letter) if (gold_letter in A_J and xai_ans  in A_J) else False

        update_agent_ema(ema_state, "OPENAI",   correct_open, lat_o, cost_open, stab_open)
        update_agent_ema(ema_state, "DEEPSEEK", correct_deep, lat_d, cost_deep, stab_deep)
        update_agent_ema(ema_state, "XAI",      correct_xai,  lat_x, cost_xai,  stab_xai)

        # 记录轨迹（画第6张图）
        for n in EMA_AGENT_NAMES:
            ema_traj[n]["ema_quality"].append(ema_state.get(n, {}).get("ema_quality"))

        print(f"[Q{qi}/{len(indices)}] GOLD={gold_letter} | ORCH={orch_ans} | OPENAI={open_ans} | "
              f"DEEPSEEK={deep_ans} | XAI={xai_ans} | VOTE={vote_ans}")

    # ACC
    def acc(golds, preds):
        ok = 0; tot = 0
        for y,p in zip(golds, preds):
            if y not in A_J: continue
            tot += 1
            if y == p: ok += 1
        return ok / max(1, tot)

    global_acc = {
        "ORCH": acc(gold_all, orch_all),
        "OPENAI": acc(gold_all, open_all),
        "DEEPSEEK": acc(gold_all, deep_all),
        "XAI": acc(gold_all, xai_all),
        "VOTE": acc(gold_all, vote_all),
    }
    print("\n[GLOBAL ACC]");
    for k,v in global_acc.items(): print(f"  {k:8s} = {v:.3f}")

    per_dir_acc: Dict[str, Dict[str,float]] = {}
    print("\n[PER-DIRECTION ACC]")
    for d in dir_names:
        per_dir_acc[d] = {
            "ORCH": acc(per_dir_gold[d], per_dir_orch[d]),
            "OPENAI": acc(per_dir_gold[d], per_dir_open[d]),
            "DEEPSEEK": acc(per_dir_gold[d], per_dir_deep[d]),
            "XAI": acc(per_dir_gold[d], per_dir_xai[d]),
            "VOTE": acc(per_dir_gold[d], per_dir_vote[d]),
        }
        a = per_dir_acc[d]
        print(f"  {d:30s} ORCH={a['ORCH']:.3f}  OPENAI={a['OPENAI']:.3f}  DEEPSEEK={a['DEEPSEEK']:.3f}  XAI={a['XAI']:.3f}  VOTE={a['VOTE']:.3f}")

    # 混淆矩阵
    print("\n[CONFUSION MATRIX ORCH (raw counts, A–J)]")
    labels = A_J; cm = confusion_matrix(gold_all, orch_all, labels=labels)
    header = "    " + " ".join([f"{L:>3s}" for L in labels]); print(header)
    for i,L in enumerate(labels):
        row = " ".join([f"{cm[i,j]:>3d}" for j in range(len(labels))]); print(f"{L}: {row}")

    # McNemar
    best_base = max(["OPENAI","DEEPSEEK","XAI"], key=lambda m: global_acc[m])
    best_preds = open_all if best_base=="OPENAI" else (deep_all if best_base=="DEEPSEEK" else xai_all)
    stats = mcnemar_stats(gold_all, orch_all, best_preds)
    print(f"\n[MCNEMAR] ORCH vs {best_base}")
    print(f"  b = ORCH correct & {best_base} wrong = {stats['b']}")
    print(f"  c = ORCH wrong  & {best_base} correct = {stats['c']}")
    print(f"  b+c = {stats['b_plus_c']}, chi2 = {stats['chi2']:.4f}, p ≈ {stats['p']:.6g}")
    print("  ==> 差异在 95% 置信水平" + ("下显著 (p < 0.05)" if stats["p"] < 0.05 else "下不显著 (p ≥ 0.05)"))

    # 平均延迟
    avg_latency = {
        "OPENAI":   total_lat_open / max(1, n_calls_open),
        "DEEPSEEK": total_lat_deep / max(1, n_calls_deep),
        "XAI":      total_lat_xai  / max(1, n_calls_xai),
        "ORCH":     total_lat_orch / max(1, n_calls_orch),
        "VOTE":     total_lat_vote / max(1, n_calls_vote),
    }
    print("\n[AVG LATENCY] (ms)")
    for k,v in avg_latency.items(): print(f"  {k:8s} ≈ {v:.1f} ms")

    # EMA 概览
    print("\n[EMA STATS PER AGENT]")
    for n in EMA_AGENT_NAMES:
        st = ema_state.get(n, {})
        print(f"  {n:8s} | ema_quality={st.get('ema_quality')} | ema_latency_ms={st.get('ema_latency_ms')} | "
              f"ema_cost={st.get('ema_cost')} | ema_stability={st.get('ema_stability')}")

    # 保存 EMA
    save_ema_state(ema_state)

    # 6 张图
    plot_global_acc(global_acc)
    plot_per_direction_acc(dir_names, per_dir_acc)
    plot_orch_confusion(gold_all, orch_all)
    plot_mcnemar_text(best_base, stats)
    plot_price_latency_table(avg_latency)
    plot_ema_quality_curve(ema_traj)

    print(f"\n[DONE] 全部图已输出到：{PLOTS_DIR}")

if __name__ == "__main__":
    main()
