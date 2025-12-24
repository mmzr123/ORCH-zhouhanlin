# -*- coding: utf-8 -*-
"""
gsm8k_orch_5plots.py  (tuned)

GSM8K（数学文字题），评测多模型集成路由器：
- 单模型基线：OPENAI / DEEPSEEK / XAI
- VOTE：3 模型直接答数值，多数投票
- ORCH：3 模型各写一段分析，由合并 Agent 输出最终数值答案

输出：
  控制台：
    - GLOBAL ACC（ORCH + 三基线 + VOTE）
    - 按“题目长度 bucket”（10 个）分组 ACC（B1: 短 -> B10: 长）
    - ORCH vs DEEPSEEK 的 2×2 混淆矩阵（correct / wrong）
    - McNemar (ORCH vs 最强单模型基线)
    - 平均延迟统计（3 provider + ORCH + VOTE）

  图像（plots_gsm8k_orch/）：
    - global_acc.png
    - per_bucket_acc.png
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

PLOTS_DIR = "plots_gsm8k_orch"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===================== API 配置（带默认 key） =====================
OPENAI_API_KEY   = os.getenv(
    "OPENAI_API_KEY",
    ""
)
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE      = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    ""
)
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")

XAI_API_KEY      = os.getenv(
    "XAI_API_KEY",
    ""
)
XAI_MODEL        = os.getenv("XAI_MODEL", "grok-2-latest")
XAI_BASE         = os.getenv("XAI_BASE", "https://api.x.ai/v1")

SYSTEM_PROMPT = (
    "You are a senior reasoning agent.\n"
    "Rules:\n"
    "1) Answer directly.\n"
    "2) Do NOT ask the the user for more info.\n"
    "3) No templates or meta-commentary.\n"
    "4) Be specific, verifiable and concise.\n"
    "5) If info is missing, make the minimum reasonable assumption and state it.\n"
)

random.seed(42)
np.random.seed(42)

# ===================== GSM8K 答案归一：提取最后一个整数 =====================
INT_RE = re.compile(r"[-+]?\d+")

def normalize_gsm8k_answer(text: str) -> str:
    """
    GSM8K 标准答案在原数据中通常是：
      "... We get 42. #### 42"
    这里统一逻辑：
      1) 优先找 '#### number'
      2) 否则取文本中最后一个整数
    """
    if not text:
        return ""
    t = text.strip()
    m = re.search(r"####\s*([-+]?\d+)", t)
    if m:
        return m.group(1).lstrip("+")
    nums = INT_RE.findall(t)
    if not nums:
        return t.strip()
    return nums[-1].lstrip("+")

# ===================== 通用 chat 调用（带重试，防超时崩溃） =====================
def call_chat_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    max_retries: int = 2,
    timeout: int = 40,
) -> Tuple[str, int]:
    """
    统一的 chat 调用，返回 (文本, 延迟 ms).
    如果 api_key 为空，则返回 echo，方便本地调试。
    内置简单重试 + 超时保护，不会因 ReadTimeout 直接崩溃。
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
        "temperature": 0.0,      # 固定 0，避免随机波动
        "max_tokens": max_tokens,
    }

    last_err = None
    start_all = time.time()

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
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
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {e}"
            print(f"[WARN] Timeout calling {model} at {base_url}, attempt {attempt+1}/{max_retries+1}")
        except requests.RequestException as e:
            last_err = f"RequestError: {e}"
            print(f"[WARN] Request error calling {model}: {e} (attempt {attempt+1}/{max_retries+1})")
        # 简单退避
        time.sleep(1.5 * (attempt + 1))

    # 所有重试仍失败，返回错误文本
    dur_ms = int((time.time() - start_all) * 1000)
    return f"[ERROR] {last_err or 'unknown error'}", dur_ms

# ===================== GSM8K 基线问答 =====================
def ask_gsm8k(provider: str, question: str) -> Tuple[str, int]:
    """
    provider in {"openai","deepseek","xai"}
    返回 (预测答案字符串, latency_ms)，答案字符串会被后续 normalize。
    """
    template = (
        "Solve the following math word problem.\n"
        "Return only the final numeric answer, with no units and no explanation.\n\n"
        "{q}\n"
    )
    q = template.format(q=question)

    if provider == "openai":
        raw, lat = call_chat_api(OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL, q, max_tokens=32)
    elif provider == "deepseek":
        raw, lat = call_chat_api(DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, q, max_tokens=32)
    elif provider == "xai":
        raw, lat = call_chat_api(XAI_BASE, XAI_API_KEY, XAI_MODEL, q, max_tokens=32)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return raw, lat

# ===================== ORCH：三模型分析 + 合并 =====================
def orch_answer(question: str) -> Tuple[str, int]:
    """
    ORCH：
      1) OPENAI / DEEPSEEK / XAI（存在哪个就用哪个）各写一段 GSM8K 分析（≤300 字）
      2) 合并 Agent（优先 XAI，否则 OPENAI，否则 DEEPSEEK）读取题目 + 3 段分析，
         只输出最终数值答案（一个整数）

    返回 (预测答案字符串, total_latency_ms)
    """
    analysis_agents = []

    if OPENAI_API_KEY:
        analysis_agents.append(("OPENAI", OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL))
    if DEEPSEEK_API_KEY:
        analysis_agents.append(("DEEPSEEK", DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL))
    if XAI_API_KEY:
        analysis_agents.append(("XAI", XAI_BASE, XAI_API_KEY, XAI_MODEL))

    if not analysis_agents:
        return "", 0

    analyses = []
    total_lat = 0

    ana_tpl = (
        "请围绕下列数学应用题进行结构化分析，严格控制在300字以内；"
        "按【思路】列出解题步骤，最后一行写“暂定答案：<数字>”。\n\n"
        "{q}\n"
    )
    ana_prompt = ana_tpl.format(q=question)

    # step 1: 每个模型写一段分析
    for name, base, key, model in analysis_agents:
        txt, lat = call_chat_api(base, key, model, ana_prompt, max_tokens=256)
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
        "你将看到三名分析员对同一数学题的分析（每条不超过300字）。"
        "请综合证据，只输出最终数值答案，例如 42，不要任何解释或其它字符。\n\n"
        f"{question}\n\n{bullet}\n\n最终答案："
    )
    m_name, m_base, m_key, m_model = merger
    merge_raw, lat_m = call_chat_api(m_base, m_key, m_model, merge_tpl, max_tokens=16)
    total_lat += lat_m
    return merge_raw, total_lat

# ===================== VOTE（三模型直接投票） =====================
def vote_numeric(open_ans: str, deep_ans: str, xai_ans: str) -> str:
    """
    对三个模型各自的答案做多数表决：
      - 先 normalize（提取最终整数）
      - 多数表决；若三家都不同，则优先 OPENAI > DEEPSEEK > XAI
    """
    n_open = normalize_gsm8k_answer(open_ans)
    n_deep = normalize_gsm8k_answer(deep_ans)
    n_xai  = normalize_gsm8k_answer(xai_ans)

    votes = {
        "OPENAI":   n_open,
        "DEEPSEEK": n_deep,
        "XAI":      n_xai,
    }
    counter = collections.Counter(v for v in votes.values() if v)
    if not counter:
        return ""
    val, cnt = counter.most_common(1)[0]
    if cnt >= 2:
        return val

    # 三家都不一样时，按优先级挑一个非空
    for name in ["OPENAI", "DEEPSEEK", "XAI"]:
        cand = votes[name]
        if cand:
            return cand
    return ""

# ===================== 抽样：10 个“bucket”（长度分桶），每个 30 题 =====================
def sample_gsm8k_10buckets(
    ds,
    per_bucket: int = 30,
    seed: int = 42
) -> Tuple[Dict[str, List[int]], List[str], Dict[str, str]]:
    """
    从 GSM8K test 集中：
      - 随机抽取 10 * per_bucket 道题
      - 将这些题按 question token 长度排序
      - 均匀切成 10 个 bucket（近似从易到难）

    返回：
      - bucket_to_indices: dict[key] = [idx...]
      - bucket_keys:       ['B1','B2',...,'B10'] 按难度从短到长
      - bucket_labels:     {'B1': 'B1: 12–24 tok', ...} 便于打印/画图
    """
    random.seed(seed)
    n_total = per_bucket * 10
    all_idx = list(range(len(ds)))
    random.shuffle(all_idx)
    chosen = all_idx[:n_total]

    # 计算长度并排序（从短到长）
    lens_all = [(i, len(ds[i]["question"].split())) for i in chosen]
    lens_all.sort(key=lambda x: x[1])

    bucket_to_indices: Dict[str, List[int]] = {}
    bucket_labels: Dict[str, str] = {}
    bucket_keys: List[str] = []

    for b in range(10):
        key = f"B{b+1}"
        start = b * per_bucket
        end = start + per_bucket
        seg = lens_all[start:end]
        idxs = [idx for idx, L in seg]
        lengths = [L for idx, L in seg]
        lmin, lmax = min(lengths), max(lengths)
        label = f"{key}: {lmin}-{lmax} tok"
        bucket_to_indices[key] = idxs
        bucket_labels[key] = label
        bucket_keys.append(key)

    print("[INFO] GSM8K length buckets:")
    for k in bucket_keys:
        idxs = bucket_to_indices[k]
        print(f"  - {bucket_labels[k]} (n={len(idxs)})")

    return bucket_to_indices, bucket_keys, bucket_labels

# ===================== McNemar 计算（针对“是否答对”） =====================
def mcnemar_stats(gold_norm: List[str], pred_a_norm: List[str], pred_b_norm: List[str]) -> Dict[str, float]:
    """
    gold_norm / pred_*_norm 均为已经 normalize 的字符串。
    b = A correct & B wrong
    c = A wrong  & B correct
    """
    assert len(gold_norm) == len(pred_a_norm) == len(pred_b_norm)
    b = c = 0
    for y, pa, pb in zip(gold_norm, pred_a_norm, pred_b_norm):
        if not y:
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
    p = math.erfc(math.sqrt(chi2 / 2.0))  # 1 自由度卡方的尾概率
    return {"b": b, "c": c, "b_plus_c": b_plus_c, "chi2": chi2, "p": p}

# ===================== 画图函数 =====================
def plot_global_acc(global_acc: Dict[str, float]):
    models = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    accs = [global_acc[m] for m in models]

    plt.figure(figsize=(6,4))
    plt.bar(models, accs)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Global Accuracy (GSM8K, 10 buckets × 30 Q)")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "global_acc.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_per_bucket_acc(
    bucket_keys: List[str],
    bucket_labels: Dict[str, str],
    per_bucket_acc: Dict[str, Dict[str, float]],
):
    models = ["ORCH", "OPENAI", "DEEPSEEK", "XAI", "VOTE"]
    x = np.arange(len(bucket_keys))
    width = 0.16

    plt.figure(figsize=(10,5))
    for j, m in enumerate(models):
        ys = [per_bucket_acc[k][m] for k in bucket_keys]
        plt.bar(x + (j - 2)*width, ys, width, label=m)

    plt.ylim(0, 1.0)
    xlabels = [bucket_labels[k] for k in bucket_keys]
    plt.xticks(x, xlabels, rotation=35, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Per-Bucket Accuracy (GSM8K, by question length)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "per_bucket_acc.png")
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[PLOT] saved -> {out}")

def plot_orch_confusion_correctness(
    gold_norm: List[str],
    orch_norm: List[str],
    deep_norm: List[str],
):
    """
    这里画的是 2×2 混淆矩阵：
      行：ORCH 正确/错误
      列：DEEPSEEK 正确/错误
    这样和 McNemar 的 b/c 是一致的，可视化“谁比谁强”。
    """
    labels = ["correct", "wrong"]
    y_true = []
    y_pred = []
    for y, po, pd in zip(gold_norm, orch_norm, deep_norm):
        if not y:
            continue
        o_ok = (po == y)
        d_ok = (pd == y)
        y_true.append("correct" if o_ok else "wrong")
        y_pred.append("correct" if d_ok else "wrong")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig = plt.figure(figsize=(5.2,4.5))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DEEPSEEK")
    ax.set_ylabel("ORCH")
    ax.set_title("Correctness Confusion (ORCH vs DEEPSEEK)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=8)
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
    固定价格（每 1M tokens），外加 ORCH / VOTE 的相对成本说明：
      - Deepseek:
          - In (cache hit): 0.2 元 / 1M tokens
          - In (no cache):  2 元 / 1M tokens
      - GPT-4o-mini:
          - In (cached):    $0.15 / 1M
          - In (no cache):  $0.075 / 1M
          - Out:            $0.60 / 1M
      - XAI:
          - In:             $2.00 / 1M
          - Out:            $10.00 / 1M
    """
    header = ["Model", "In_cached", "In_noncached", "Out", "Avg latency (ms)"]

    rows = [
        [
            "ORCH (3-agent+merge)",
            "≈ 3×(O + D + X)",
            "-",
            "-",
            f"{avg_latency.get('ORCH', float('nan')):.1f}",
        ],
        [
            "VOTE (3-model)",
            "≈ O + D + X",
            "-",
            "-",
            f"{avg_latency.get('VOTE', float('nan')):.1f}",
        ],
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

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
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
        description="GSM8K: ORCH + 3 baselines + VOTE + 5 plots."
    )
    parser.add_argument("--per_bucket", type=int, default=30,
                        help="每个长度 bucket 抽取题目数（默认 30）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[LOAD] openai/gsm8k (split=test, main)")
    ds = load_dataset("gsm8k", "main", split="test")

    bucket_to_indices, bucket_keys, bucket_labels = sample_gsm8k_10buckets(
        ds, per_bucket=args.per_bucket, seed=args.seed
    )
    total_q = sum(len(v) for v in bucket_to_indices.values())
    print(f"[INFO] 总样本数 ≈ {total_q} (10 buckets × {args.per_bucket})\n")

    # 全局结果（保存“原始预测”和 normalize 之后的）
    gold_raw:  List[str] = []
    gold_norm: List[str] = []

    open_raw:  List[str] = []
    deep_raw:  List[str] = []
    xai_raw:   List[str] = []
    vote_raw:  List[str] = []
    orch_raw:  List[str] = []

    open_norm: List[str] = []
    deep_norm: List[str] = []
    xai_norm:  List[str] = []
    vote_norm: List[str] = []
    orch_norm: List[str] = []

    # 延迟统计
    total_lat_open = total_lat_deep = total_lat_xai = 0.0
    total_lat_orch = total_lat_vote = 0.0
    n_calls_open = n_calls_deep = n_calls_xai = 0
    n_calls_orch = n_calls_vote = 0

    # 分 bucket 统计
    per_bucket_gold: Dict[str, List[str]] = {k: [] for k in bucket_keys}
    per_bucket_orch: Dict[str, List[str]] = {k: [] for k in bucket_keys}
    per_bucket_open: Dict[str, List[str]] = {k: [] for k in bucket_keys}
    per_bucket_deep: Dict[str, List[str]] = {k: [] for k in bucket_keys}
    per_bucket_xai:  Dict[str, List[str]] = {k: [] for k in bucket_keys}
    per_bucket_vote: Dict[str, List[str]] = {k: [] for k in bucket_keys}

    qi = 0
    for key in bucket_keys:
        idxs = bucket_to_indices[key]
        label = bucket_labels[key]
        for idx in idxs:
            qi += 1
            ex = ds[idx]
            q_text = ex["question"]
            gold_ans_raw = ex["answer"]
            gold_ans_norm = normalize_gsm8k_answer(gold_ans_raw)

            # 基线
            open_ans_raw, lat_o = ask_gsm8k("openai",  q_text)
            deep_ans_raw, lat_d = ask_gsm8k("deepseek", q_text)
            xai_ans_raw,  lat_x = ask_gsm8k("xai",      q_text)

            open_ans_norm = normalize_gsm8k_answer(open_ans_raw)
            deep_ans_norm = normalize_gsm8k_answer(deep_ans_raw)
            xai_ans_norm  = normalize_gsm8k_answer(xai_ans_raw)

            # VOTE（直接用 normalize 之后的）
            vote_ans_norm = vote_numeric(open_ans_raw, deep_ans_raw, xai_ans_raw)
            vote_ans_raw  = vote_ans_norm  # 对于数值任务，raw 就用归一后的数值

            # ORCH
            orch_ans_raw, lat_orch = orch_answer(q_text)
            orch_ans_norm = normalize_gsm8k_answer(orch_ans_raw)

            gold_raw.append(gold_ans_raw)
            gold_norm.append(gold_ans_norm)

            open_raw.append(open_ans_raw)
            deep_raw.append(deep_ans_raw)
            xai_raw.append(xai_ans_raw)
            vote_raw.append(vote_ans_raw)
            orch_raw.append(orch_ans_raw)

            open_norm.append(open_ans_norm)
            deep_norm.append(deep_ans_norm)
            xai_norm.append(xai_ans_norm)
            vote_norm.append(vote_ans_norm)
            orch_norm.append(orch_ans_norm)

            per_bucket_gold[key].append(gold_ans_norm)
            per_bucket_open[key].append(open_ans_norm)
            per_bucket_deep[key].append(deep_ans_norm)
            per_bucket_xai[key].append(xai_ans_norm)
            per_bucket_vote[key].append(vote_ans_norm)
            per_bucket_orch[key].append(orch_ans_norm)

            total_lat_open += lat_o
            total_lat_deep += lat_d
            total_lat_xai  += lat_x
            total_lat_orch += lat_orch
            total_lat_vote += (lat_o + lat_d + lat_x)

            n_calls_open += 1
            n_calls_deep += 1
            n_calls_xai  += 1
            n_calls_orch += 1
            n_calls_vote += 1

            print(
                f"[Q{qi}/{total_q}] {label} | "
                f"GOLD={gold_ans_norm} | "
                f"ORCH={orch_ans_norm} | OPENAI={open_ans_norm} | "
                f"DEEPSEEK={deep_ans_norm} | XAI={xai_ans_norm} | "
                f"VOTE={vote_ans_norm}"
            )

    def acc(golds_norm, preds_norm):
        ok = 0
        tot = 0
        for y, p in zip(golds_norm, preds_norm):
            if not y:
                continue
            tot += 1
            if y == p:
                ok += 1
        return ok / max(1, tot)

    # ===== Global ACC =====
    global_acc = {
        "ORCH":     acc(gold_norm, orch_norm),
        "OPENAI":   acc(gold_norm, open_norm),
        "DEEPSEEK": acc(gold_norm, deep_norm),
        "XAI":      acc(gold_norm, xai_norm),
        "VOTE":     acc(gold_norm, vote_norm),
    }

    print("\n[GLOBAL ACC]")
    for k, v in global_acc.items():
        print(f"  {k:8s} = {v:.3f}")

    # ===== Per-bucket ACC =====
    per_bucket_acc: Dict[str, Dict[str, float]] = {}
    print("\n[PER-BUCKET ACC]")
    for key in bucket_keys:
        acc_orch = acc(per_bucket_gold[key], per_bucket_orch[key])
        acc_open = acc(per_bucket_gold[key], per_bucket_open[key])
        acc_deep = acc(per_bucket_gold[key], per_bucket_deep[key])
        acc_xai  = acc(per_bucket_gold[key], per_bucket_xai[key])
        acc_vote = acc(per_bucket_gold[key], per_bucket_vote[key])

        per_bucket_acc[key] = {
            "ORCH":     acc_orch,
            "OPENAI":   acc_open,
            "DEEPSEEK": acc_deep,
            "XAI":      acc_xai,
            "VOTE":     acc_vote,
        }
        label = bucket_labels[key]
        print(
            f"  {label:18s} ORCH={acc_orch:.3f}  OPENAI={acc_open:.3f}  "
            f"DEEPSEEK={acc_deep:.3f}  XAI={acc_xai:.3f}  VOTE={acc_vote:.3f}"
        )

    # ===== McNemar: ORCH vs 最强单模型基线（不包括 VOTE） =====
    best_base = max(["OPENAI", "DEEPSEEK", "XAI"],
                    key=lambda m: global_acc[m])
    if best_base == "OPENAI":
        best_norm = open_norm
    elif best_base == "DEEPSEEK":
        best_norm = deep_norm
    else:
        best_norm = xai_norm

    stats = mcnemar_stats(gold_norm, orch_norm, best_norm)
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

    # ===== 画 5 张图 =====
    plot_global_acc(global_acc)
    plot_per_bucket_acc(bucket_keys, bucket_labels, per_bucket_acc)
    plot_orch_confusion_correctness(gold_norm, orch_norm, deep_norm)
    plot_mcnemar_text(best_base, stats)
    plot_price_latency_table(avg_latency)

    print(f"\n[DONE] 所有图已输出到文件夹：{PLOTS_DIR}")

if __name__ == "__main__":
    main()

