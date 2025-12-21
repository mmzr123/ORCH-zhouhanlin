# -*- coding: utf-8 -*-
"""
mmlu_pro_10choice_orch_10dirs_ema_router.py  (minimal EMA upgrade, 2025-11-10)

仅加入三项最有用的小改动：
1) 质量的风险感知：维护 ema_quality_var，并用 q_eff = q - RISK_K*std 排序
2) 动态并发：用 LCB = q - z*std 决定面板规模 1/2/3 家
3) 稳健延迟：更新 ema_latency_ms 前对极端慢值裁剪为 3×当前EMA（至少30ms）

本版按你的要求：
- 使用全新 JSON 文件：ema_state_mmlu_pro_1110_new.json（不复用旧文件）
- 输出图片文件夹：修改EMA-MMUL-PRO-1110
"""

from __future__ import annotations
import os
import time
import math
import random
import re
import argparse
import collections
import json
from typing import List, Dict, Tuple, Optional

# ===== 必要依赖 =====
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

# ===== 输出图片目录（按你的要求）=====
PLOTS_DIR = "修改EMA-MMUL-PRO-1110"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===================== API 配置 =====================
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

# ===================== EMA 配置（使用独立的新 JSON） =====================
EMA_STATE_PATH   = "ema_state_mmlu_pro_1110_new.json"  # 新文件名，避免复用旧历史
EMA_AGENT_NAMES  = ["OPENAI", "DEEPSEEK", "XAI"]

EMA_ALPHA_QUALITY   = 0.2
EMA_ALPHA_LATENCY   = 0.2
EMA_ALPHA_COST      = 0.2
EMA_ALPHA_STABILITY = 0.2

# ===== 轻量路由增强参数 =====
RISK_K = 0.30       # 质量风险惩罚：q_eff = q - RISK_K * std
LCB_Z  = 1.0        # LCB = q - z*std
CONF_LCB_SOLO = 0.80  # LCB >= 0.80 -> 仅 1 家
CONF_LCB_DUO  = 0.65  # 0.65 <= LCB < 0.80 -> 2 家；否则 3 家

# 价格表（按 100 万 tokens），只用于估算每次调用成本的相对大小
PRICE_PER_M = {
    "OPENAI": {"currency": "USD", "input": 0.15, "output": 0.60},
    "DEEPSEEK": {"currency": "CNY", "input": 2.0, "output": None},
    "XAI": {"currency": "USD", "input": 2.0, "output": 10.0},
}

def init_empty_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    state: Dict[str, Dict[str, Optional[float]]] = {}
    for name in agent_names:
        state[name] = {
            "ema_quality": None,
            "ema_quality_var": None,      # 新增：质量的指数方差
            "ema_latency_ms": None,
            "ema_cost": None,
            "ema_stability": None,
        }
    return state

def load_ema_state(agent_names: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    # 使用全新文件名；若不存在则新建，不去加载旧文件名
    if not os.path.exists(EMA_STATE_PATH):
        state = init_empty_ema_state(agent_names)
        with open(EMA_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[EMA] no existing file, created new: {EMA_STATE_PATH}")
        return state

    try:
        with open(EMA_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("bad json format")
    except Exception as e:
        print(f"[EMA] load error ({e}), re-init state.")
        state = {}

    # 补齐缺失 agent / 字段
    for name in agent_names:
        if name not in state or not isinstance(state[name], dict):
            state[name] = {}
        st = state[name]
        for k in ["ema_quality", "ema_quality_var", "ema_latency_ms", "ema_cost", "ema_stability"]:
            if k not in st:
                st[k] = None
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
    """根据 usage 和单价做一个近似成本估计；tokens 不存在时返回 None。"""
    if tokens_in is None and tokens_out is None:
        return None
    cfg = PRICE_PER_M.get(agent_name)
    if cfg is None:
        return None
    ti = tokens_in or 0
    to = tokens_out or 0
    inp_price = cfg.get("input")
    out_price = cfg.get("output")
    if inp_price is None and out_price is None:
        return None
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
    """对单个 Agent 做一次 EMA 更新（仅在原基础上加了两点：质量方差 & 稳健延迟）。"""
    if agent_name not in ema_state:
        return
    st = ema_state[agent_name]

    # ===== quality + variance =====
    if correct is not None:
        q_obs = 1.0 if correct else 0.0
        # 先更新 mean
        new_mean = ema_update(st.get("ema_quality"), q_obs, EMA_ALPHA_QUALITY)
        # 再基于“与新均值的偏差”更新指数方差
        dev2 = (q_obs - (new_mean if new_mean is not None else q_obs)) ** 2
        new_var = ema_update(st.get("ema_quality_var"), dev2, EMA_ALPHA_QUALITY)
        st["ema_quality"] = new_mean
        st["ema_quality_var"] = new_var

    # ===== latency（稳健：裁剪极端慢值） =====
    if latency_ms is not None:
        lat_val = float(latency_ms)
        prev_lat = st.get("ema_latency_ms")
        if prev_lat is not None:
            cap = 3.0 * max(30.0, float(prev_lat))
            lat_val = min(lat_val, cap)
        st["ema_latency_ms"] = ema_update(prev_lat, lat_val, EMA_ALPHA_LATENCY)

    # ===== cost =====
    if cost is not None:
        st["ema_cost"] = ema_update(st.get("ema_cost"), float(cost), EMA_ALPHA_COST)

    # ===== stability =====
    if stability_ok is not None:
        s_obs = 1.0 if stability_ok else 0.0
        st["ema_stability"] = ema_update(st.get("ema_stability"), s_obs, EMA_ALPHA_STABILITY)

# ===================== 工具函数：A–J 归一 =====================
A_J = [chr(ord("A") + i) for i in range(10)]
LETTER10_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)

def normalize_to_letter_10(text: str) -> str:
    if not text:
        return "?"
    m = LETTER10_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    t = (text or "").strip().upper()
    for L in A_J:
        if re.match(rf"^\s*{L}\b", t):
            return L
    return "?"

# ===================== 通用 chat 调用 =====================
def call_chat_api(base_url: str, api_key: str, model: str,
                  prompt: str, max_tokens: int = 64) -> Tuple[str, int, Optional[int], Optional[int], int]:
    """
    统一的 chat 调用，返回 (文本, 延迟 ms, tokens_in, tokens_out, http_status).
    如果 api_key 为空，则返回 echo。
    对超时 / 网络错误做兜底，不让整个评测崩掉。
    """
    if not api_key:
        fake = "[ECHO] " + prompt[:120]
        return fake, 1000, None, None, 200

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=40)
        dur_ms = int((time.time() - t0) * 1000)
    except ReadTimeout:
        dur_ms = int((time.time() - t0) * 1000)
        return "[TIMEOUT]", dur_ms, None, None, 0
    except RequestException as e:
        dur_ms = int((time.time() - t0) * 1000)
        return f"[REQUEST_ERROR] {e}", dur_ms, None, None, 0

    status = resp.status_code
    tokens_in = None
    tokens_out = None

    try:
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            or ""
        ).strip()
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
    except Exception:
        text = f"[HTTP {resp.status_code}] {resp.text[:200]}"

    return text, dur_ms, tokens_in, tokens_out, status

# ===================== 10 选一问答（基线） =====================
def ask_mcq_10choice(provider: str, prompt: str) -> Tuple[str, int, Optional[int], Optional[int], int]:
    """
    provider in {"openai","deepseek","xai"}
    返回 (A-J 或 "?", latency_ms, tokens_in, tokens_out, http_status)
    """
    template = (
        "You are given a 10-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, D, E, F, G, H, I, or J.\n\n"
        "{q}\n"
    )
    q = template.format(q=prompt)

    if provider == "openai":
        raw, lat, ti, to, st = call_chat_api(OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL, q, max_tokens=16)
    elif provider == "deepseek":
        raw, lat, ti, to, st = call_chat_api(DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, q, max_tokens=16)
    elif provider == "xai":
        raw, lat, ti, to, st = call_chat_api(XAI_BASE, XAI_API_KEY, XAI_MODEL, q, max_tokens=16)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    letter = normalize_to_letter_10(raw)
    return letter, lat, ti, to, st

# ===================== ORCH：EMA 路由版（仅小改） =====================
def ema_routing_key(agent_name: str,
                    ema_state: Dict[str, Dict[str, Optional[float]]]) -> Tuple[float, float, float, float]:
    """
    返回用于排序的 key：
      (-q_eff, latency, cost, -stability)
    其中 q_eff = q - RISK_K * std；std = sqrt(ema_quality_var)
    None 用默认值做冷启动：
      q=0.5, std=0.25 -> q_eff≈0.5-0.25*RISK_K；lat=2000ms, cost=1.0, stab=0.5
    """
    st = ema_state.get(agent_name, {})
    q = st.get("ema_quality")
    var = st.get("ema_quality_var")
    lat = st.get("ema_latency_ms")
    cost = st.get("ema_cost")
    stab = st.get("ema_stability")

    if q is None:
        q = 0.5
    std = math.sqrt(var) if (var is not None and var >= 0.0) else 0.25
    q_eff = max(0.0, min(1.0, q - RISK_K * std))

    if lat is None:
        lat = 2000.0
    if cost is None:
        cost = 1.0
    if stab is None:
        stab = 0.5

    return (-float(q_eff), float(lat), float(cost), -float(stab))

def orch_answer(prompt: str,
                ema_state: Dict[str, Dict[str, Optional[float]]]) -> Tuple[str, int]:
    """
    ORCH（带轻量增强的 EMA 路由）：
      1) 按新 routing_key 排序
      2) 用最佳者的 LCB = q - z*std 决定面板规模（1/2/3）
      3) 面板内产出分析，由最佳者做合并
    返回 (A-J 或 "?", total_latency_ms)
    """
    # 所有可用模型
    candidates = []
    if OPENAI_API_KEY:
        candidates.append(("OPENAI",   OPENAI_BASE,   OPENAI_API_KEY,   OPENAI_MODEL))
    if DEEPSEEK_API_KEY:
        candidates.append(("DEEPSEEK", DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL))
    if XAI_API_KEY:
        candidates.append(("XAI",      XAI_BASE,      XAI_API_KEY,      XAI_MODEL))
    if not candidates:
        return "?", 0

    # ===== 打印路由前的指标 =====
    print("\n[ROUTER] candidate indicators:")
    for name, _, _, _ in candidates:
        st = ema_state.get(name, {})
        q = st.get("ema_quality")
        var = st.get("ema_quality_var")
        std = math.sqrt(var) if (var is not None and var >= 0.0) else None
        print(
            f"  - {name:8s} | "
            f"q={None if q is None else f'{q:.3f}'} | "
            f"std={None if std is None else f'{std:.3f}'} | "
            f"lat={st.get('ema_latency_ms')} | "
            f"cost={st.get('ema_cost')} | "
            f"stab={st.get('ema_stability')}"
        )

    # ===== 新 routing_key 排序 =====
    candidates_sorted = sorted(
        candidates,
        key=lambda item: ema_routing_key(item[0], ema_state)
    )

    print("[ROUTER] sorted agents (best first):")
    for rank, (name, _, _, _) in enumerate(candidates_sorted, start=1):
        print(f"  rank {rank}: {name}")

    # ===== 动态并发规模：看最佳者的 LCB =====
    best_name = candidates_sorted[0][0]
    st_best = ema_state.get(best_name, {})

    # 冷启动安全默认：q=0.5, std=0.25（即 var=0.0625）
    q_best = st_best.get("ema_quality")
    if q_best is None:
        q_best = 0.5

    var_best = st_best.get("ema_quality_var")
    std_best = math.sqrt(var_best) if (var_best is not None and var_best >= 0.0) else 0.25

    lcb = float(q_best) - float(LCB_Z) * float(std_best)

    if lcb >= CONF_LCB_SOLO:
        k_panel = 1
    elif lcb >= CONF_LCB_DUO:
        k_panel = 2
    else:
        k_panel = 3

    analysis_agents = candidates_sorted[:k_panel]
    merger = candidates_sorted[0]
    analysis_names = [a[0] for a in analysis_agents]
    merger_name = merger[0]
    print(f"[ROUTER] LCB(best={best_name})={lcb:.3f} -> panel_size={k_panel}; panel={analysis_names}, merger={merger_name}")

    analyses = []
    total_lat = 0

    ana_tpl = (
        "请围绕以下十选一多项选择题进行结构化分析，严格控制在300字以内；"
        "按【要点】列出依据与排除逻辑，最后给出'暂定答案(仅A~J中的一个字母)'。\n\n"
        "{q}\n"
    )
    ana_prompt = ana_tpl.format(q=prompt)

    # step 1: 每个选择出的模型写一段分析
    for name, base, key, model in analysis_agents:
        txt, lat, _, _, _ = call_chat_api(base, key, model, ana_prompt, max_tokens=512)
        total_lat += lat
        analyses.append((name, txt))

    # step 2: 合并 Agent：用路由选出的 merger
    m_name, m_base, m_key, m_model = merger
    bullet = "\n\n".join([f"[{name}] {txt}" for name, txt in analyses])
    merge_tpl = (
        "你将看到多名分析员对同一十选一题的分析（均≤500字）。"
        "请综合证据，仅输出最终选项字母（A/B/C/D/E/F/G/H/I/J），不得附加解释或其它字符。\n\n"
        f"{prompt}\n\n{bullet}\n\n最终答案："
    )
    merge_raw, lat_m, _, _, _ = call_chat_api(m_base, m_key, m_model, merge_tpl, max_tokens=16)
    total_lat += lat_m
    letter = normalize_to_letter_10(merge_raw)
    return letter, total_lat

# ===================== VOTE（A–J） =====================
def vote_AJ(open_ans: str, deep_ans: str, xai_ans: str) -> str:
    """
    3 个模型对 A–J 投票：
      - 多数表决
      - 若三家都不同，优先级 OPENAI > DEEPSEEK > XAI
    """
    votes = {"OPENAI": open_ans, "DEEPSEEK": deep_ans, "XAI": xai_ans}
    counter = collections.Counter(votes.values())
    if not counter:
        return "?"
    letter, cnt = counter.most_common(1)[0]
    if cnt >= 2:
        return letter
    for name in ["OPENAI", "DEEPSEEK", "XAI"]:
        cand = votes[name]
        if cand in A_J:
            return cand
    return "?"

# ===================== 抽样：10 个方向，每个 per_dir 题 =====================
def sample_mmlu_pro_10dirs(ds, per_dir: int = 30, seed: int = 42) -> Tuple[List[int], List[str]]:
    random.seed(seed)
    idx_all = [i for i, opts in enumerate(ds["options"])
               if isinstance(opts, list) and len(opts) == 10]

    by_cat: Dict[str, List[int]] = {}
    for i in idx_all:
        cat = ds["category"][i]
        by_cat.setdefault(cat, []).append(i)

    cats_sorted = sorted(by_cat.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(cats_sorted) < 10:
        raise RuntimeError(f"可用 category 少于 10 个，只找到：{[c for c,_ in cats_sorted]}")

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

# ===================== McNemar 计算 =====================
def mcnemar_stats(gold: List[str],
                  pred_a: List[str],
                  pred_b: List[str]) -> Dict[str, float]:
    assert len(gold) == len(pred_a) == len(pred_b)
    b = c = 0
    for y, pa, pb in zip(gold, pred_a, pred_b):
        if y not in A_J:
            continue
        ca = (pa == y)
        cb = (pb == y)
        if ca and (not cb):
            b += 1
        elif (not ca) and cb:
            c += 1
    b_plus_c = b + c
    if b_plus_c == 0:
        return {"b": b, "c": c, "b_plus_c": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (b - c) ** 2 / float(b_plus_c)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "b_plus_c": b_plus_c, "chi2": chi2, "p": p}

# ===================== 画图函数 =====================
def plot_global_acc(global_acc: Dict[str, float]):
    models = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    accs = [global_acc[m] for m in models]

    plt.figure(figsize=(6,4))
    plt.bar(models, accs)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Global Accuracy (MMLU-Pro, 10 dirs × per_dir Q)")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "global_acc.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_per_direction_acc(dir_names: List[str],
                           per_dir_acc: Dict[str, Dict[str, float]]):
    models = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    x = np.arange(len(dir_names))
    width = 0.16

    plt.figure(figsize=(10,5))
    for j, m in enumerate(models):
        ys = [per_dir_acc[d][m] for d in dir_names]
        plt.bar(x + (j - 2)*width, ys, width, label=m)

    plt.ylim(0, 1.0)
    plt.xticks(x, dir_names, rotation=35, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Per-Direction Accuracy (10 dirs, per_dir Q each)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "per_direction_acc.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_orch_confusion(golds: List[str], preds_orch: List[str]):
    labels = A_J
    cm = confusion_matrix(golds, preds_orch, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix (ORCH, A–J)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "orch_confusion.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_mcnemar_text(best_baseline_name: str, stats: Dict[str, float]):
    b = stats["b"]; c = stats["c"]; bc = stats["b_plus_c"]; chi2 = stats["chi2"]; p = stats["p"]
    txt = (
        f"McNemar Test: ORCH vs {best_baseline_name}\n\n"
        f"b = ORCH correct & {best_baseline_name} wrong = {b}\n"
        f"c = ORCH wrong  & {best_baseline_name} correct = {c}\n"
        f"b + c = {bc}\n"
        f"chi2 = {chi2:.4f}\n"
        f"p ≈ {p:.6f}\n\n"
        f"Conclusion: {'significant (p < 0.05)' if p < 0.05 else 'not significant (p ≥ 0.05)'}"
    )
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axis("off")
    ax.text(0.01, 0.99, txt, va="top", ha="left", fontsize=10, family="monospace")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "mcnemar_orch_vs_best.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_price_latency_table(avg_latency: Dict[str, float]):
    header = ["Model", "In_cached", "In_noncached", "Out", "Avg latency (ms)"]
    rows = [
        ["ORCH (EMA-router)", "≈ 3×(O + D + X)", "-", "-", f"{avg_latency.get('ORCH', float('nan')):.1f}"],
        ["VOTE (3-model)", "≈ O + D + X", "-", "-", f"{avg_latency.get('VOTE', float('nan')):.1f}"],
        ["DEEPSEEK", "0.2 元 / 1M", "2 元 / 1M", "-", f"{avg_latency.get('DEEPSEEK', float('nan')):.1f}"],
        ["GPT-4o-mini", "$0.15 / 1M", "$0.075 / 1M", "$0.60 / 1M", f"{avg_latency.get('OPENAI', float('nan')):.1f}"],
        ["XAI (grok)", "$2.00 / 1M", "-", "$10.00 / 1M", f"{avg_latency.get('XAI', float('nan')):.1f}"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.axis("off")
    table_data = [header] + rows
    table = ax.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Price & Average Latency", pad=10)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "price_latency_table.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_ema_quality_curve(ema_traj: Dict[str, Dict[str, List[Optional[float]]]]):
    """
    只画一张图：ema_quality_curve.png
    横轴题号，纵轴为 EMA quality（近似近期准确率），三条曲线对应三个 agent。
    """
    plt.figure(figsize=(7, 4))
    for agent_name in EMA_AGENT_NAMES:
        ys = ema_traj.get(agent_name, {}).get("ema_quality", [])
        if not ys:
            continue
        ys_plot = [np.nan if v is None else v for v in ys]
        xs = list(range(1, len(ys_plot) + 1))
        plt.plot(xs, ys_plot, label=agent_name)
    plt.xlabel("Question Index")
    plt.ylabel("EMA Quality (recent accuracy)")
    plt.title("EMA Quality vs Question Index")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "ema_quality_curve.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out_path}")

# ===================== 主评测逻辑 =====================
def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Pro 10-choice, 10 dirs × per_dir Q, ORCH(EMA-router) + 3 baselines + VOTE."
    )
    parser.add_argument("--per_dir", type=int, default=30, help="每个方向抽取题目数（默认 30）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ===== EMA 状态加载 / 初始化（使用全新 JSON 文件名）=====
    ema_state = load_ema_state(EMA_AGENT_NAMES)

    # ===== EMA 轨迹，用于画曲线（仍记录 ema_quality） =====
    ema_traj: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for name in EMA_AGENT_NAMES:
        ema_traj[name] = {"ema_quality": [], "ema_latency_ms": [], "ema_cost": [], "ema_stability": []}

    print("[LOAD] TIGER-Lab/MMLU-Pro (split=test)")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    indices, dir_names = sample_mmlu_pro_10dirs(ds, per_dir=args.per_dir, seed=args.seed)
    print(f"[INFO] 总样本数 ≈ {len(indices)} (10 dirs × {args.per_dir})\n")

    gold_all: List[str]  = []
    open_all: List[str]  = []
    deep_all: List[str]  = []
    xai_all:  List[str]  = []
    vote_all: List[str]  = []
    orch_all: List[str]  = []
    dir_all:  List[str]  = []

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
        q = ds["question"][i]
        opts = ds["options"][i]
        gold_idx = int(ds["answer_index"][i])
        gold_letter = A_J[gold_idx] if 0 <= gold_idx < 10 else "?"
        cat = ds["category"][i]

        prompt = "Question: " + str(q).strip() + "\nOptions:\n" + \
                 "\n".join([f"{A_J[k]}. {opts[k]}" for k in range(10)])

        # ===== 单模型基线 =====
        open_ans, lat_o, ti_o, to_o, st_o = ask_mcq_10choice("openai",  prompt)
        deep_ans, lat_d, ti_d, to_d, st_d = ask_mcq_10choice("deepseek", prompt)
        xai_ans,  lat_x, ti_x, to_x, st_x = ask_mcq_10choice("xai",      prompt)

        vote_ans = vote_AJ(open_ans, deep_ans, xai_ans)

        # ===== ORCH (EMA 路由) =====
        orch_ans, lat_orch = orch_answer(prompt, ema_state)

        # ===== 记录全局 =====
        gold_all.append(gold_letter)
        open_all.append(open_ans)
        deep_all.append(deep_ans)
        xai_all.append(xai_ans)
        vote_all.append(vote_ans)
        orch_all.append(orch_ans)
        dir_all.append(cat)

        # ===== 分方向 =====
        if cat in dir_names:
            per_dir_gold[cat].append(gold_letter)
            per_dir_open[cat].append(open_ans)
            per_dir_deep[cat].append(deep_ans)
            per_dir_xai[cat].append(xai_ans)
            per_dir_vote[cat].append(vote_ans)
            per_dir_orch[cat].append(orch_ans)

        # ===== 延迟累积 =====
        total_lat_open += lat_o
        total_lat_deep += lat_d
        total_lat_xai  += lat_x
        total_lat_orch += lat_orch
        lat_vote = lat_o + lat_d + lat_x
        total_lat_vote += lat_vote

        n_calls_open += 1
        n_calls_deep += 1
        n_calls_xai  += 1
        n_calls_orch += 1
        n_calls_vote += 1

        # ===== EMA 更新（基于这一题的对错 + 延迟 + 成本 + 稳定性） =====
        cost_open = calc_call_cost("OPENAI",   ti_o, to_o)
        cost_deep = calc_call_cost("DEEPSEEK", ti_d, to_d)
        cost_xai  = calc_call_cost("XAI",      ti_x, to_x)

        stab_open = (st_o == 200)
        stab_deep = (st_d == 200)
        stab_xai  = (st_x == 200)

        correct_open = (open_ans == gold_letter) if (gold_letter in A_J and open_ans in A_J) else False
        correct_deep = (deep_ans == gold_letter) if (gold_letter in A_J and deep_ans in A_J) else False
        correct_xai  = (xai_ans  == gold_letter) if (gold_letter in A_J and xai_ans  in A_J) else False

        update_agent_ema(ema_state, "OPENAI",   correct_open, lat_o, cost_open, stab_open)
        update_agent_ema(ema_state, "DEEPSEEK", correct_deep, lat_d, cost_deep, stab_deep)
        update_agent_ema(ema_state, "XAI",      correct_xai,  lat_x, cost_xai,  stab_xai)

        # ===== 把当前 EMA 状态写入轨迹，用于画 EMA 曲线 =====
        for name in EMA_AGENT_NAMES:
            st = ema_state.get(name, {})
            for key in ["ema_quality"]:  # 只画质量曲线
                ema_traj[name][key].append(st.get(key))

        print(f"[Q{qi}/{len(indices)}] GOLD={gold_letter} | "
              f"ORCH={orch_ans} | OPENAI={open_ans} | DEEPSEEK={deep_ans} | XAI={xai_ans} | VOTE={vote_ans}")

    # ===== ACC 计算 =====
    def acc(golds, preds):
        ok = 0
        total = 0
        for y, p in zip(golds, preds):
            if y not in A_J:
                continue
            total += 1
            if y == p:
                ok += 1
        return ok / max(1, total)

    global_acc = {
        "ORCH":     acc(gold_all, orch_all),
        "OPENAI":   acc(gold_all, open_all),
        "DEEPSEEK": acc(gold_all, deep_all),
        "XAI":      acc(gold_all, xai_all),
        "VOTE":     acc(gold_all, vote_all),
    }

    print("\n[GLOBAL ACC]")
    for k, v in global_acc.items():
        print(f"  {k:8s} = {v:.3f}")

    per_dir_acc: Dict[str, Dict[str, float]] = {}
    print("\n[PER-DIRECTION ACC]")
    for d in dir_names:
        acc_orch = acc(per_dir_gold[d], per_dir_orch[d])
        acc_open = acc(per_dir_gold[d], per_dir_open[d])
        acc_deep = acc(per_dir_gold[d], per_dir_deep[d])
        acc_xai  = acc(per_dir_gold[d], per_dir_xai[d])
        acc_vote = acc(per_dir_gold[d], per_dir_vote[d])

        per_dir_acc[d] = {"ORCH": acc_orch, "OPENAI": acc_open, "DEEPSEEK": acc_deep, "XAI": acc_xai, "VOTE": acc_vote}
        print(f"  {d:30s} ORCH={acc_orch:.3f}  OPENAI={acc_open:.3f}  DEEPSEEK={acc_deep:.3f}  XAI={acc_xai:.3f}  VOTE={acc_vote:.3f}")

    # ===== Confusion Matrix =====
    print("\n[CONFUSION MATRIX ORCH (raw counts, A–J)]")
    labels = A_J
    cm = confusion_matrix(gold_all, orch_all, labels=labels)
    header = "    " + " ".join([f"{L:>3s}" for L in labels])
    print(header)
    for i, L in enumerate(labels):
        row = " ".join([f"{cm[i, j]:>3d}" for j in range(len(labels))])
        print(f"{L}: {row}")

    # ===== McNemar: ORCH vs 最强单模型基线 =====
    best_base = max(["OPENAI", "DEEPSEEK", "XAI"], key=lambda m: global_acc[m])
    best_preds = open_all if best_base == "OPENAI" else (deep_all if best_base == "DEEPSEEK" else xai_all)

    stats = mcnemar_stats(gold_all, orch_all, best_preds)
    print(f"\n[MCNEMAR] ORCH vs {best_base}")
    print(f"  b = ORCH correct & {best_base} wrong = {stats['b']}")
    print(f"  c = ORCH wrong  & {best_base} correct = {stats['c']}")
    print(f"  b+c = {stats['b_plus_c']}, chi2 = {stats['chi2']:.4f}, p ≈ {stats['p']:.6g}")
    if stats["p"] < 0.05:
        print("  ==> 差异在 95% 置信水平下显著 (p < 0.05)")
    else:
        print("  ==> 差异在 95% 置信水平下不显著 (p ≥ 0.05)")

    # ===== 平均延迟 =====
    avg_latency = {
        "OPENAI":   total_lat_open / max(1, n_calls_open),
        "DEEPSEEK": total_lat_deep / max(1, n_calls_deep),
        "XAI":      total_lat_xai  / max(1, n_calls_xai),
        "ORCH":     total_lat_orch / max(1, n_calls_orch),
        "VOTE":     total_lat_vote / max(1, n_calls_vote),
    }

    print("\n[AVG LATENCY] (ms)")
    for k, v in avg_latency.items():
        print(f"  {k:8s} ≈ {v:.1f} ms")

    # ===== 打印每个 Agent 的最终 EMA 指标 =====
    print("\n[EMA STATS PER AGENT]")
    for name in EMA_AGENT_NAMES:
        st = ema_state.get(name, {})
        # 补充打印 std/LCB 方便观察
        q = st.get('ema_quality')
        var = st.get('ema_quality_var')
        std = math.sqrt(var) if (var is not None and var >= 0.0) else None
        lcb = (q - LCB_Z * std) if (q is not None and std is not None) else None
        print(
            f"  {name:8s} | "
            f"ema_quality={q} | "
            f"std={None if std is None else f'{std:.3f}'} | "
            f"LCB={None if lcb is None else f'{lcb:.3f}'} | "
            f"ema_latency_ms={st.get('ema_latency_ms')} | "
            f"ema_cost={st.get('ema_cost')} | "
            f"ema_stability={st.get('ema_stability')}"
        )

    # ===== 保存 EMA 状态 =====
    save_ema_state(ema_state)

    # ===== 画图 =====
    plot_global_acc(global_acc)
    plot_per_direction_acc(dir_names, per_dir_acc)
    plot_orch_confusion(gold_all, orch_all)
    plot_mcnemar_text(best_base, stats)
    plot_price_latency_table(avg_latency)
    plot_ema_quality_curve(ema_traj)

    print(f"\n[DONE] 所有图已输出到文件夹：{PLOTS_DIR}")

if __name__ == "__main__":
    main()
