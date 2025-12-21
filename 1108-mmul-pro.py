# -*- coding: utf-8 -*-
"""
mmlu_pro_10choice_orch_3dirs.py

MMLU-Pro (10-choice A–J), 3 个方向（category），每个方向 20 题：
- 单模型基线：OPENAI / DEEPSEEK / XAI
- VOTE：3 模型直接答 A–J，多数投票
- ORCH：3 模型各写一段 10 选一分析，由合并 Agent 输出最终 A–J

输出：
  控制台：
    - GLOBAL ACC（ORCH + 三基线 + VOTE）
    - PER-DIRECTION ACC（3 个方向）
    - ORCH 的 10×10 混淆矩阵
    - McNemar (ORCH vs 最强单模型基线)
    - 平均延迟统计（3 provider + ORCH）

  图像（plots_mmlu_pro_orch_3dirs/）：
    - global_acc.png
    - per_direction_acc.png
    - orch_confusion.png
    - mcnemar_orch_vs_best.png
    - price_latency_table.png
"""

from __future__ import annotations
import os
import time
import math
import random
import re
import argparse
import collections
from typing import List, Dict, Tuple

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

PLOTS_DIR = "plots_mmlu_pro_orch_3dirs"
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
                  prompt: str, max_tokens: int = 64) -> Tuple[str, int]:
    """
    统一的 chat 调用，返回 (文本, 延迟 ms).
    如果 api_key 为空，则返回 echo，方便本地调试。
    """
    if not api_key:
        fake = "[ECHO] " + prompt[:120]
        return fake, 1000

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,    # 固定 0，避免随机波动
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    dur_ms = int((time.time() - t0) * 1000)

    try:
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            or ""
        ).strip()
    except Exception:
        text = f"[HTTP {resp.status_code}] {resp.text[:200]}"

    return text, dur_ms

# ===================== 10 选一问答（基线） =====================
def ask_mcq_10choice(provider: str, prompt: str) -> Tuple[str, int]:
    """
    provider in {"openai","deepseek","xai"}
    返回 (A-J 或 "?", latency_ms)
    """
    template = (
        "You are given a 10-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, D, E, F, G, H, I, or J.\n\n"
        "{q}\n"
    )
    q = template.format(q=prompt)

    if provider == "openai":
        raw, lat = call_chat_api(OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL, q, max_tokens=16)
    elif provider == "deepseek":
        raw, lat = call_chat_api(DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, q, max_tokens=16)
    elif provider == "xai":
        raw, lat = call_chat_api(XAI_BASE, XAI_API_KEY, XAI_MODEL, q, max_tokens=16)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    letter = normalize_to_letter_10(raw)
    return letter, lat

# ===================== ORCH：三模型分析 + 合并 =====================
def orch_answer(prompt: str) -> Tuple[str, int]:
    """
    ORCH：
      1) OPENAI / DEEPSEEK / XAI（存在哪个就用哪个）各写一段 10 选一分析（≤300 字）
      2) 合并 Agent（优先 XAI，否则 OPENAI，否则 DEEPSEEK）读取原题 + 3 段分析，只输出 A–J
    返回 (A-J 或 "?", total_latency_ms)
    """
    analysis_agents = []

    if OPENAI_API_KEY:
        analysis_agents.append(("OPENAI", OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL))
    if DEEPSEEK_API_KEY:
        analysis_agents.append(("DEEPSEEK", DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL))
    if XAI_API_KEY:
        analysis_agents.append(("XAI", XAI_BASE, XAI_API_KEY, XAI_MODEL))

    if not analysis_agents:
        # 没有可用模型时，退化为 echo
        return "?", 0

    analyses = []
    total_lat = 0

    ana_tpl = (
        "请围绕以下十选一多项选择题进行结构化分析，严格控制在300字以内；"
        "按【要点】列出依据与排除逻辑，最后给出'暂定答案(仅A~J中的一个字母)'。\n\n"
        "{q}\n"
    )
    ana_prompt = ana_tpl.format(q=prompt)

    # step 1: 每个模型写一段分析
    for name, base, key, model in analysis_agents:
        txt, lat = call_chat_api(base, key, model, ana_prompt, max_tokens=512)
        total_lat += lat
        analyses.append((name, txt))

    # step 2: 合并 Agent：优先 XAI -> OPENAI -> DEEPSEEK -> 第一个分析 Agent
    if XAI_API_KEY:
        merger = ("XAI", XAI_BASE, XAI_API_KEY, XAI_MODEL)
    elif OPENAI_API_KEY:
        merger = ("OPENAI", OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL)
    elif DEEPSEEK_API_KEY:
        merger = ("DEEPSEEK", DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL)
    else:
        merger = analysis_agents[0]

    bullet = "\n\n".join([f"[{name}] {txt}" for name, txt in analyses])
    merge_tpl = (
        "你将看到三名分析员对同一十选一题的分析（均≤500字）。"
        "请综合证据，仅输出最终选项字母（A/B/C/D/E/F/G/H/I/J），不得附加解释或其它字符。\n\n"
        f"{prompt}\n\n{bullet}\n\n最终答案："
    )
    m_name, m_base, m_key, m_model = merger
    merge_raw, lat_m = call_chat_api(m_base, m_key, m_model, merge_tpl, max_tokens=16)
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
    votes = {
        "OPENAI":   open_ans,
        "DEEPSEEK": deep_ans,
        "XAI":      xai_ans,
    }
    counter = collections.Counter(votes.values())
    if not counter:
        return "?"
    letter, cnt = counter.most_common(1)[0]
    if cnt >= 2:
        return letter

    # 三家都不一样时，按优先级挑一个合法字母
    for name in ["OPENAI", "DEEPSEEK", "XAI"]:
        cand = votes[name]
        if cand in A_J:
            return cand
    return "?"

# ===================== 抽样：3 个方向，每个 20 题 =====================
def sample_mmlu_pro_3dirs(ds, per_dir: int = 20, seed: int = 42) -> Tuple[List[int], List[str]]:
    """
    从 MMLU-Pro 测试集中：
      - 只保留 options 长度==10 的样本
      - 按 category 分组
      - 选出 3 个样本数最多的 category
      - 每个 category 抽 per_dir 道题

    返回：
      - indices: 所有被选中的样本索引列表（长度 ≈ 3 * per_dir）
      - dir_names: 3 个方向名称
    """
    random.seed(seed)
    idx_all = [i for i, opts in enumerate(ds["options"])
               if isinstance(opts, list) and len(opts) == 10]

    by_cat: Dict[str, List[int]] = {}
    for i in idx_all:
        cat = ds["category"][i]
        by_cat.setdefault(cat, []).append(i)

    # 按样本量排序，选最多的三个 category 作为“方向”
    cats_sorted = sorted(by_cat.items(), key=lambda kv: len(kv[1]), reverse=True)
    if len(cats_sorted) < 3:
        raise RuntimeError(f"可用 category 少于 3 个，只找到：{[c for c,_ in cats_sorted]}")

    chosen_cats = [cats_sorted[i][0] for i in range(3)]
    print("[INFO] 选中的 3 个方向(category)：")
    for c in chosen_cats:
        print(f"  - {c} (samples with 10 options = {len(by_cat[c])})")

    indices: List[int] = []
    for c in chosen_cats:
        pool = by_cat[c][:]
        random.shuffle(pool)
        indices.extend(pool[:per_dir])

    return indices, chosen_cats

# ===================== McNemar 计算（1 自由度） =====================
def mcnemar_stats(gold: List[str],
                  pred_a: List[str],
                  pred_b: List[str]) -> Dict[str, float]:
    """
    针对同一批题目，比较模型 A 和 B：
      b = A correct & B wrong
      c = A wrong  & B correct

    返回字典：b, c, b_plus_c, chi2, p
    """
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

    # 不带连续性修正的 McNemar 统计量：chi2 = (b - c)^2 / (b + c)
    chi2 = (b - c) ** 2 / float(b_plus_c)
    # 1 自由度卡方的尾概率：p = P(X >= chi2) = erfc( sqrt(chi2/2) )
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
    plt.title("Global Accuracy (MMLU-Pro, 3 dirs × 20 Q)")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "global_acc.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_per_direction_acc(dir_names: List[str],
                           per_dir_acc: Dict[str, Dict[str, float]]):
    """
    per_dir_acc[dir_name][model] = acc
    """
    models = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    x = np.arange(len(dir_names))
    width = 0.16

    plt.figure(figsize=(8,5))
    for j, m in enumerate(models):
        ys = [per_dir_acc[d][m] for d in dir_names]
        plt.bar(x + (j - 2)*width, ys, width, label=m)

    plt.ylim(0, 1.0)
    plt.xticks(x, dir_names, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Per-Direction Accuracy (3 dirs, 20 Q each)")
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
    b = stats["b"]
    c = stats["c"]
    bc = stats["b_plus_c"]
    chi2 = stats["chi2"]
    p = stats["p"]

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
    ax.text(0.01, 0.99, txt, va="top", ha="left",
            fontsize=10, family="monospace")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "mcnemar_orch_vs_best.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_price_latency_table(avg_latency: Dict[str, float]):
    """
    价格按照你给的数字硬编码：
      - Deepseek:
          - In (cache hit): 0.2 元 / 1M tokens
          - In (no cache):  2 元 / 1M tokens
      - GPT-4o-mini:
          - In (cached):    $0.15 / 1M
          - In (no cache):  $0.075 / 1M
          - Out:            $0.60 / 1M
      - XAI:
          - In (cached):    $2.00 / 1M
          - Out:            $10.00 / 1M

    再加上一列平均延迟(ms)。
    """
    header = ["Model", "In_cached", "In_noncached", "Out", "Avg latency (ms)"]

    rows = [
        [
            "DEEPSEEK",
            "0.2 元 / 1M",
            "2 元 / 1M",
            "-",
            f"{avg_latency.get('DEEPSEEK', float('nan')):.1f}",
        ],
        [
            "GPT-4o-mini",
            "$0.15 / 1M",
            "$0.075 / 1M",
            "$0.60 / 1M",
            f"{avg_latency.get('OPENAI', float('nan')):.1f}",
        ],
        [
            "XAI (grok)",
            "$2.00 / 1M",
            "-",
            "$10.00 / 1M",
            f"{avg_latency.get('XAI', float('nan')):.1f}",
        ],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.axis("off")
    table_data = [header] + rows
    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Price & Average Latency", pad=10)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "price_latency_table.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

# ===================== 主评测逻辑 =====================
def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Pro 10-choice, 3 dirs × 20 Q, ORCH + 3 baselines + VOTE."
    )
    parser.add_argument("--per_dir", type=int, default=20,
                        help="每个方向抽取题目数（默认 20）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[LOAD] TIGER-Lab/MMLU-Pro (split=test)")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    indices, dir_names = sample_mmlu_pro_3dirs(ds, per_dir=args.per_dir, seed=args.seed)
    print(f"[INFO] 总样本数 ≈ {len(indices)} (3 dirs × {args.per_dir})\n")

    # 全局结果
    gold_all: List[str]  = []
    open_all: List[str]  = []
    deep_all: List[str]  = []
    xai_all:  List[str]  = []
    vote_all: List[str]  = []
    orch_all: List[str]  = []
    dir_all:  List[str]  = []

    # 延迟统计（provider + ORCH）
    total_lat_open = total_lat_deep = total_lat_xai = total_lat_orch = 0.0
    n_calls_open = n_calls_deep = n_calls_xai = n_calls_orch = 0

    # 分方向统计
    per_dir_gold: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_open: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_deep: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_xai:  Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_vote: Dict[str, List[str]] = {d: [] for d in dir_names}
    per_dir_orch: Dict[str, List[str]] = {d: [] for d in dir_names}

    for qi, i in enumerate(indices, 1):
        q = ds["question"][i]
        opts = ds["options"][i]   # list of 10
        gold_idx = int(ds["answer_index"][i])
        gold_letter = A_J[gold_idx] if 0 <= gold_idx < 10 else "?"
        cat = ds["category"][i]

        prompt = "Question: " + str(q).strip() + "\nOptions:\n" + \
                 "\n".join([f"{A_J[k]}. {opts[k]}" for k in range(10)])

        # 单模型基线
        open_ans, lat_o = ask_mcq_10choice("openai",  prompt)
        deep_ans, lat_d = ask_mcq_10choice("deepseek", prompt)
        xai_ans,  lat_x = ask_mcq_10choice("xai",      prompt)

        vote_ans = vote_AJ(open_ans, deep_ans, xai_ans)

        # ORCH
        orch_ans, lat_orch = orch_answer(prompt)

        gold_all.append(gold_letter)
        open_all.append(open_ans)
        deep_all.append(deep_ans)
        xai_all.append(xai_ans)
        vote_all.append(vote_ans)
        orch_all.append(orch_ans)
        dir_all.append(cat)

        if cat in dir_names:
            per_dir_gold[cat].append(gold_letter)
            per_dir_open[cat].append(open_ans)
            per_dir_deep[cat].append(deep_ans)
            per_dir_xai[cat].append(xai_ans)
            per_dir_vote[cat].append(vote_ans)
            per_dir_orch[cat].append(orch_ans)

        total_lat_open += lat_o
        total_lat_deep += lat_d
        total_lat_xai  += lat_x
        total_lat_orch += lat_orch
        n_calls_open += 1
        n_calls_deep += 1
        n_calls_xai  += 1
        n_calls_orch += 1

        print(f"[Q{qi}/{len(indices)}] GOLD={gold_letter} | "
              f"ORCH={orch_ans} | OPENAI={open_ans} | DEEPSEEK={deep_ans} | XAI={xai_ans} | VOTE={vote_ans}")

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

    # ===== Global ACC =====
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

    # ===== Per-direction ACC =====
    per_dir_acc: Dict[str, Dict[str, float]] = {}
    print("\n[PER-DIRECTION ACC]")
    for d in dir_names:
        acc_orch = acc(per_dir_gold[d], per_dir_orch[d])
        acc_open = acc(per_dir_gold[d], per_dir_open[d])
        acc_deep = acc(per_dir_gold[d], per_dir_deep[d])
        acc_xai  = acc(per_dir_gold[d], per_dir_xai[d])
        acc_vote = acc(per_dir_gold[d], per_dir_vote[d])

        per_dir_acc[d] = {
            "ORCH":     acc_orch,
            "OPENAI":   acc_open,
            "DEEPSEEK": acc_deep,
            "XAI":      acc_xai,
            "VOTE":     acc_vote,
        }
        print(f"  {d:25s} ORCH={acc_orch:.3f}  OPENAI={acc_open:.3f}  "
              f"DEEPSEEK={acc_deep:.3f}  XAI={acc_xai:.3f}  VOTE={acc_vote:.3f}")

    # ===== Confusion Matrix (ORCH) =====
    print("\n[CONFUSION MATRIX ORCH (raw counts, A–J)]")
    labels = A_J
    cm = confusion_matrix(gold_all, orch_all, labels=labels)
    header = "    " + " ".join([f"{L:>3s}" for L in labels])
    print(header)
    for i, L in enumerate(labels):
        row = " ".join([f"{cm[i, j]:>3d}" for j in range(len(labels))])
        print(f"{L}: {row}")

    # ===== McNemar: ORCH vs 最强单模型基线（不包括 VOTE） =====
    best_base = max(["OPENAI", "DEEPSEEK", "XAI"],
                    key=lambda m: global_acc[m])
    if best_base == "OPENAI":
        best_preds = open_all
    elif best_base == "DEEPSEEK":
        best_preds = deep_all
    else:
        best_preds = xai_all

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
    }
    print("\n[AVG LATENCY] (ms)")
    for k, v in avg_latency.items():
        print(f"  {k:8s} ≈ {v:.1f} ms")

    # ===== 画 5 张图 =====
    plot_global_acc(global_acc)
    plot_per_direction_acc(dir_names, per_dir_acc)
    plot_orch_confusion(gold_all, orch_all)
    plot_mcnemar_text(best_base, stats)
    # 价格表只放三家 provider，ORCH 是组合策略不单列价格
    plot_price_latency_table(avg_latency)

    print(f"\n[DONE] 所有图已输出到文件夹：{PLOTS_DIR}")

if __name__ == "__main__":
    main()
