# -*- coding: utf-8 -*-
"""
mmlu_4choice_vote_only.py

- 10 个 MMLU 学科 × 每科 N 题（4 选一）
- 三条基线：OPENAI / DEEPSEEK / XAI
- VOTE：三模型各自只输出 A/B/C/D，多数投票

输出：
  - 控制台：全局 ACC、分学科 ACC
  - 图像（plots_vote/ 下 3 张 png）：
      1) global_acc.png
      2) per_subject_acc.png
      3) vote_confusion.png
"""

from __future__ import annotations
import os, time, random, re, argparse, collections
from typing import List, Dict, Tuple, Any

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import requests

PLOTS_DIR = "plots_vote"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===================== API 配置 =====================
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE      = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE    = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")

XAI_API_KEY      = os.getenv("XAI_API_KEY", "")
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

def normalize_to_letter(text: str) -> str:
    """从模型输出里抽取 A/B/C/D 单个字母."""
    if not text:
        return "?"
    LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    t = (text or "").strip().upper()
    for L in ["A","B","C","D"]:
        if re.match(rf"^\s*{L}\b", t):
            return L
    return "?"

def call_chat_api(base_url: str, api_key: str, model: str, prompt: str) -> Tuple[str,int]:
    """统一的 chat 调用，返回文本和延迟(ms). key 为空时本地 echo."""
    if not api_key:
        # 没 key 时做一个 deterministic echo，方便本地调试
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
        "temperature": 0.0,          # 固定 0，避免采样波动
        "max_tokens": 64,            # 只要一个字母，给少一点
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    dur_ms = int((time.time() - t0) * 1000)
    try:
        data = resp.json()
        text = (
            data.get("choices",[{}])[0]
                .get("message",{})
                .get("content","")
            or ""
        ).strip()
    except Exception:
        text = f"[HTTP {resp.status_code}] {resp.text[:200]}"
    return text, dur_ms

def ask_mcq_4choice(provider: str, prompt: str) -> Tuple[str,int]:
    """
    provider in {"openai","deepseek","xai"}
    返回 (A/B/C/D/?, latency_ms)
    """
    template = (
        "You are given a 4-choice multiple-choice question.\n"
        "Pick the single correct option.\n"
        "Return only the letter A, B, C, or D.\n\n"
        "{q}\n"
    )
    q = template.format(q=prompt)

    if provider == "openai":
        raw, lat = call_chat_api(OPENAI_BASE, OPENAI_API_KEY, OPENAI_MODEL, q)
    elif provider == "deepseek":
        raw, lat = call_chat_api(DEEPSEEK_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, q)
    elif provider == "xai":
        raw, lat = call_chat_api(XAI_BASE, XAI_API_KEY, XAI_MODEL, q)
    else:
        raise ValueError(provider)

    letter = normalize_to_letter(raw)
    return letter, lat

def vote_abcd(open_ans: str, deep_ans: str, xai_ans: str) -> str:
    """
    用三个答案做 ABCD 投票。
    - 多数表决
    - 三家都不一样时，优先级：OPENAI > DEEPSEEK > XAI
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

    # 极端情况：三家都不一样
    prefer = ["XAI","OPENAI","DEEPSEEK"]
    for name in prefer:
        L = votes.get(name, "?")
        if L in ["A","B","C","D"]:
            return L
    return letter

SUBJECTS = [
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

def sample_indices(ds, n: int) -> List[int]:
    idxs = [i for i, ex in enumerate(ds) if len(ex["choices"]) >= 4]
    random.shuffle(idxs)
    return idxs[:min(n, len(idxs))]

def acc(ys: List[str], ps: List[str]) -> float:
    return sum(1 for y,p in zip(ys,ps) if y==p) / max(1, len(ys))

def mmlu_eval_vote_only(per_subject_n: int = 30, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    gold_all:  List[str] = []
    open_all:  List[str] = []
    deep_all:  List[str] = []
    xai_all:   List[str] = []
    vote_all:  List[str] = []

    lat_open = lat_deep = lat_xai = 0.0
    total_q = 0

    per_subject_stats: Dict[str, Dict[str,List[str]]] = {
        s: {"gold":[], "open":[], "deep":[], "xai":[], "vote":[]} for s in SUBJECTS
    }

    for subj in SUBJECTS:
        print(f"\n[LOAD] cais/mmlu, subject={subj}")
        ds = load_dataset("cais/mmlu", subj, split="test")
        idxs = sample_indices(ds, per_subject_n)

        for local_i, i in enumerate(idxs, 1):
            ex = ds[i]
            q = ex["question"]
            choices = ex["choices"]
            gold_idx = int(ex["answer"])
            if len(choices) < 4:
                continue

            prompt = (
                f"Question: {q}\n"
                f"Options:\nA. {choices[0]}\nB. {choices[1]}\n"
                f"C. {choices[2]}\nD. {choices[3]}\n"
            )

            open_ans, lat_o = ask_mcq_4choice("openai",  prompt)
            deep_ans, lat_d = ask_mcq_4choice("deepseek",prompt)
            xai_ans,  lat_x = ask_mcq_4choice("xai",     prompt)
            vote_ans        = vote_abcd(open_ans, deep_ans, xai_ans)

            gold_letter = ["A","B","C","D"][gold_idx]

            gold_all.append(gold_letter)
            open_all.append(open_ans)
            deep_all.append(deep_ans)
            xai_all.append(xai_ans)
            vote_all.append(vote_ans)

            s = per_subject_stats[subj]
            s["gold"].append(gold_letter)
            s["open"].append(open_ans)
            s["deep"].append(deep_ans)
            s["xai"].append(xai_ans)
            s["vote"].append(vote_ans)

            lat_open += lat_o
            lat_deep += lat_d
            lat_xai  += lat_x
            total_q  += 1

            print(f"[{subj} Q{local_i}/{len(idxs)}] GOLD={gold_letter} | "
                  f"OPENAI={open_ans} | DEEPSEEK={deep_ans} | XAI={xai_ans} | "
                  f"VOTE={vote_ans}")

    # ===== 全局 ACC =====
    acc_open = acc(gold_all, open_all)
    acc_deep = acc(gold_all, deep_all)
    acc_xai  = acc(gold_all, xai_all)
    acc_vote = acc(gold_all, vote_all)

    print("\n[GLOBAL ACC]")
    print(f"  OPENAI   = {acc_open:.3f}")
    print(f"  DEEPSEEK = {acc_deep:.3f}")
    print(f"  XAI      = {acc_xai:.3f}")
    print(f"  VOTE     = {acc_vote:.3f}")

    # ===== 分学科 ACC =====
    print("\n[PER-SUBJECT ACC]")
    per_subject_acc: Dict[str, Dict[str,float]] = {}
    for subj in SUBJECTS:
        st = per_subject_stats[subj]
        a_open = acc(st["gold"], st["open"])
        a_deep = acc(st["gold"], st["deep"])
        a_xai  = acc(st["gold"], st["xai"])
        a_vote = acc(st["gold"], st["vote"])
        per_subject_acc[subj] = {
            "open": a_open,
            "deep": a_deep,
            "xai":  a_xai,
            "vote": a_vote,
        }
        print(f"  {subj:26s} OPENAI={a_open:.3f}  DEEPSEEK={a_deep:.3f}  "
              f"XAI={a_xai:.3f}  VOTE={a_vote:.3f}")

    # ===== VOTE 混淆矩阵 =====
    labels = ["A","B","C","D"]
    cm_vote = confusion_matrix(gold_all, vote_all, labels=labels)
    print("\n[CONFUSION MATRIX VOTE (raw counts)]")
    print("     A   B   C   D")
    for i, L in enumerate(labels):
        row = " ".join(f"{cm_vote[i,j]:3d}" for j in range(len(labels)))
        print(f"{L}:  {row}")

    # ===== 平均延迟 =====
    if total_q > 0:
        avg_open = lat_open / total_q
        avg_deep = lat_deep / total_q
        avg_xai  = lat_xai  / total_q
    else:
        avg_open = avg_deep = avg_xai = 0.0

    print("\n[AVG LATENCY] (ms / question)")
    print(f"  OPENAI   ≈ {avg_open:.0f} ms")
    print(f"  DEEPSEEK ≈ {avg_deep:.0f} ms")
    print(f"  XAI      ≈ {avg_xai:.0f} ms")

    # ================== 画图 ==================
    # 1) 全局 ACC
    fig1 = plt.figure(figsize=(5,4))
    ax1 = fig1.add_subplot(1,1,1)
    models = ["OPENAI","DEEPSEEK","XAI","VOTE"]
    accs   = [acc_open, acc_deep, acc_xai, acc_vote]
    ax1.bar(models, accs)
    ax1.set_ylim(0,1.0)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Global Accuracy (4-choice MMLU, vote vs baselines)")
    for i,v in enumerate(accs):
        ax1.text(i, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig1.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "global_acc.png")
    fig1.savefig(path1, dpi=160)
    print(f"[PLOT] saved -> {path1}")

    # 2) 分学科 ACC
    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.add_subplot(1,1,1)
    idx = np.arange(len(SUBJECTS))
    width = 0.2
    open_vals = [per_subject_acc[s]["open"] for s in SUBJECTS]
    deep_vals = [per_subject_acc[s]["deep"] for s in SUBJECTS]
    xai_vals  = [per_subject_acc[s]["xai"]  for s in SUBJECTS]
    vote_vals = [per_subject_acc[s]["vote"] for s in SUBJECTS]

    ax2.bar(idx - 1.5*width, open_vals, width, label="OPENAI")
    ax2.bar(idx - 0.5*width, deep_vals, width, label="DEEPSEEK")
    ax2.bar(idx + 0.5*width, xai_vals,  width, label="XAI")
    ax2.bar(idx + 1.5*width, vote_vals, width, label="VOTE")

    ax2.set_xticks(idx)
    ax2.set_xticklabels(SUBJECTS, rotation=45, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0,1.0)
    ax2.set_title("Per-subject Accuracy (vote vs baselines)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "per_subject_acc.png")
    fig2.savefig(path2, dpi=160)
    print(f"[PLOT] saved -> {path2}")

    # 3) VOTE 混淆矩阵
    fig3 = plt.figure(figsize=(4.5,4))
    ax3 = fig3.add_subplot(1,1,1)
    im = ax3.imshow(cm_vote.astype(float), interpolation="nearest")
    ax3.set_xticks(range(len(labels)))
    ax3.set_yticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("Predicted (VOTE)")
    ax3.set_ylabel("Gold")
    ax3.set_title("Confusion Matrix (VOTE)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax3.text(j, i, f"{cm_vote[i,j]:d}", ha="center", va="center", fontsize=8)
    fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()
    path3 = os.path.join(PLOTS_DIR, "vote_confusion.png")
    fig3.savefig(path3, dpi=160)
    print(f"[PLOT] saved -> {path3}")

def main():
    ap = argparse.ArgumentParser(description="4-choice MMLU vote-only (3 baselines + VOTE)")
    ap.add_argument("--per_subject_n", type=int, default=30,
                    help="每个学科抽题数，默认 30")
    args = ap.parse_args()
    mmlu_eval_vote_only(per_subject_n=args.per_subject_n, seed=42)

if __name__ == "__main__":
    main()

