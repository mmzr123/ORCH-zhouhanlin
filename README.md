好的，这是为你整理的纯文本格式 README 内容。你可以直接选中并复制到你的 `README.md` 文件中。

---

# ORCH: Multi-Agent Orchestrator for Discrete-Choice Reasoning

### Project Overview

This repository contains runnable scripts for evaluating ORCH-style multi-agent routing and aggregation. The framework is designed to optimize discrete-choice reasoning across several benchmarks:

* MMLU: 4-option multiple-choice evaluation.
* MMLU-Pro: 10-option multiple-choice (A–J), offering higher challenge levels.
* GSM8K: Math word problem solving with structured reasoning.

Core Architecture (ORCH):

1. Question Decomposition: A dispatcher agent rewrites a single original item into a guided set of sub-questions.
2. Parallel Analysis: Multiple LLM agents solve different sub-questions in parallel, producing independent reasoning traces.
3. Merge: A dedicated merge agent consolidates these traces into a final single choice.
4. Baselines: Supports single-shot runs for providers like OpenAI, DeepSeek, xAI, and Groq for performance comparison.

---

### Requirements

Python 3.9+ is recommended. Install the necessary dependencies via pip:

pip install -U datasets numpy matplotlib scikit-learn requests python-dotenv

Note: The datasets library will automatically download benchmark data from Hugging Face upon execution.

---

### API Keys and Security

IMPORTANT: Do NOT hard-code API keys in the source code. This repository uses environment variables to manage credentials securely.

1. Create a .env file in the root directory (this file is ignored by Git):

OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
XAI_API_KEY=your_xai_key
GROQ_API_KEY=your_groq_key

2. Ensure your .gitignore includes the following to prevent accidental leaks:

.env
*.key
*.pem
**pycache**/

3. In your Python scripts, always retrieve keys using os.getenv("KEY_NAME").

---

### Scripts Description

The following scripts implement different evaluation protocols and ablation studies:

MMLU Series:

* 1108-mmlu-10学科.py: Runs MMLU on a fixed 10-subject subset for reproducible reporting.
* 1109-MMLU+EMA.py: MMLU evaluation with EMA-based online scoring (reliability/latency/cost) for routing.
* 1110-mmlu-ema-k=2.py: EMA routing variant with a fixed panel size of 2 parallel agents.

MMLU-Pro Series:

* 1108-mmml-pro.py: Standard MMLU-Pro evaluation (10-option format).
* 1109-MMLU-PRO-EMA.py: MMLU-Pro evaluation using EMA-style routing.
* 1109-mmlu-pro-10x30.py: Implementation of the 10x30 batching and sampling protocol.
* 1110-mmlu-pro-ema-k=2.py: MMLU-Pro EMA routing with 2 parallel agents.

Other Tasks & Variants:

* 1109-gsm8k.py: Math word problem evaluation.
* 1108-vote+三基线.py: Comparison between three single-shot baselines and a simple majority voting ensemble.
* 1109-openai+dp(orch-1).py: Ablation study focusing on the OpenAI decomposition pipeline (ORCH-1).

---

### Quick Start

Linux / macOS:
export OPENAI_API_KEY="your_key_here"
python 1108-mmlu-10学科.py

Windows PowerShell:
$env:OPENAI_API_KEY="your_key_here"
python "1108-mmlu-10学科.py"

Outputs:

* Per-item logs comparing Gold answers vs ORCH vs Baselines.
* Aggregate accuracy and confusion matrices.
* Performance plots saved as .png files.

---

### Reproducibility

To ensure exact replication of results:

* A global seed (default SEED=42) is used for all sampling and shuffling.
* Sampling is performed deterministically per subject.
* All sampled indices are stored during the run to allow for audit and re-evaluation.

---

### Security: Secret Recovery

If you have accidentally committed API keys to the Git history, you must:

1. Revoke the keys immediately at the provider's dashboard.
2. Purge the history using git-filter-repo:

git-filter-repo --invert-paths --path PATH/TO/FILE_WITH_KEYS
git push --force --mirror origin

Collaborators must re-clone the repository after a history rewrite to avoid re-introducing the deleted secrets.
