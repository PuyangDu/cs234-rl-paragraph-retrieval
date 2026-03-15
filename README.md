# RL for Multi-Hop QA Paragraph Retrieval

CS234 Final Project — Learning sequential paragraph retrieval policies for multi-hop question answering via Behavioral Cloning (BC), Proximal Policy Optimization (PPO), and Direct Preference Optimization (DPO).

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Method](#method)
4. [Training](#training)
5. [Results](#results)
6. [LLM Evaluation](#llm-evaluation)
7. [Zero-Shot Transfer](#zero-shot-transfer)
8. [File Structure](#file-structure)
9. [How to Run](#how-to-run)

---

## Overview

Multi-hop question answering requires reasoning over multiple documents. Given a question and 10 candidate paragraphs (a mix of gold supporting paragraphs and distractors), our system learns a **sequential retrieval policy** that decides which paragraphs to read and when to stop, before handing selected context to an LLM for answer generation.

We compare three RL approaches:
- **Behavioral Cloning (BC)**: Supervised imitation of oracle demonstrations
- **BC + PPO**: On-policy fine-tuning with dense retrieval rewards
- **SFT + DPO**: Offline preference optimization from ranked trajectory pairs

All training uses retrieval-only signals (no LLM calls). Final evaluation uses Qwen3-8B via Ollama to measure end-to-end answer accuracy.

### Dataset

We use **2WikiMultiHopQA** (compositional subset) in **blind mode** — all paragraph titles are anonymized to `Para_0` ... `Para_9` to prevent title-based shortcuts. Each question has 2 or 4 gold supporting paragraphs out of 10 candidates.

| Split | Size | Purpose |
|---|---|---|
| BC Train | 2,249 | Behavioral cloning supervision |
| BC Dev | 449 | BC early stopping |
| PPO/DPO Train | 4,499 | On-policy rollouts / preference pairs |
| Eval | 1,499 | Held-out retrieval evaluation |

---

## Model Architecture

![Model Architecture](model_architecture.png)

**RetrievalSelector** — a dual-path MLP (~200K parameters) that outputs a distribution over 11 actions (`read_0` ... `read_9`, `answer`):

- **Input** $\phi(s_t) \in \mathbb{R}^{486}$: 384-dim sentence embedding (all-MiniLM-L6-v2) + 10 cosine similarities + 32 structured features + 60 per-paragraph features
- **Path A** (Per-Paragraph Scorer): Per-paragraph features $(B, 10, 10) \to$ MLP $\to$ `para_scores` $\in \mathbb{R}^{10}$. Frozen during PPO.
- **Path B** (Context Pathway): Global features $(B, 396) \to$ Linear $\to$ LayerNorm $\to$ ReLU $\to$ ResBlock $\to$ Dropout(0.15) $\to$ three heads: `ctx_to_para` $(128 \to 10)$, `answer` $(128 \to 1)$, `value` $(128 \to 1)$
- **Combined**: $\text{logits} = [\alpha \cdot \text{pathA} + (1-\alpha) \cdot \text{pathB},\; \text{answer\_logit}]$ where $\alpha = \sigma(\text{learnable})$, init $\approx 0.62$
- **Output**: $\pi_\theta(a|s) = \text{softmax}(\text{logits})$ with already-read paragraphs masked to $-\infty$; $\hat{V}(s_t)$ for GAE baseline

---

## Method

### Behavioral Cloning (BC)

Supervised cross-entropy loss on oracle demonstrations. At each state, the oracle selects the gold paragraph with the highest per-paragraph score (or `answer` when all golds are read). Early stopping on BC Dev F1 (patience = 5).

### PPO Fine-Tuning

Starting from BC weights, fine-tune with PPO using dense per-step rewards:

| Reward Component | Value |
|---|---|
| Read gold paragraph | +1.0 |
| Read distractor | −0.2 |
| Step cost | −0.08 |
| Order bonus (gold in dataset order) | +0.1 |
| Bridge entity bonus | +0.15 |
| Completion bonus (all golds found) | +1.5 |

Additional details: adaptive KL coefficient (0.03 initial, auto-tuned to keep KL in [0.1, 0.4]), potential-based reward shaping ramping from 0 to 0.2 after iteration 3, Path A frozen during PPO, 4 epochs per update, clip $\epsilon = 0.2$.

### SFT + DPO (Direct Preference Optimization)

Starting from BC weights, train with DPO ($\beta = 0.1$) on preference pairs constructed from the PPO training set:
- **Read preferences**: At each trajectory step, the oracle's chosen paragraph is preferred over each non-chosen paragraph
- **Stop preferences**: At terminal states (all golds read), `answer` is preferred over each remaining distractor

DPO avoids reward engineering but has a fundamental limitation in this multi-step setting: the shared `answer_head` cannot distinguish "read 1 of 2 golds" from "read 2 of 2 golds," causing the stop signal to over-generalize. The resulting policy reads exactly 1 paragraph then stops (P=90.8%, R=37.3%, F1=52.9%). This illustrates that offline preference methods struggle with sequential decision-making where the value of stopping depends on trajectory history.

---

## Training

### Hyperparameters

| Parameter | Value |
|---|---|
| Sentence encoder | all-MiniLM-L6-v2 (384-dim) |
| Feature dimension | 486 |
| Learning rate (BC / PPO / DPO) | 3 × 10⁻⁵ |
| PPO iterations | 15 (patience = 6) |
| PPO epochs per update | 4 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.2 |
| Entropy coefficient | 0.02 |
| DPO $\beta$ | 0.1 |
| Budget K | 5 |
| Seed | 42 |

---

## Results

### Retrieval Performance (1,499 Eval Questions, Blind Mode)

| Strategy | Precision | Recall | F1 | Avg Reads |
|---|---|---|---|---|
| Best Greedy (K=3) | 41.4% | 51.0% | 45.7% | 3.0 |
| SFT + DPO | 90.8% | 37.3% | 52.9% | 1.0 |
| BC | 76.1% | 76.9% | 76.5% | 2.5 |
| **BC + PPO** | **77.1%** | **77.2%** | **77.1%** | **2.4** |

### ROC Analysis (Step-0 Paragraph Ranking)

| Model | AUC |
|---|---|
| Random | 0.506 |
| Greedy (BoW) | 0.631 |
| SFT + DPO | 0.844 |
| BC | 0.881 |
| BC + PPO | 0.885 |

![ROC Curve](checkpoints_blind/roc_curve.png)

### Statistical Significance (Paired Permutation, 10K Permutations)

| Comparison | F1 (A) | F1 (B) | p-value |
|---|---|---|---|
| BC+PPO vs BC | 77.1% | 76.5% | 0.018 \* |
| SFT+DPO vs BC | 52.9% | 76.5% | 1.000 |

### PPO Training Curve

![PPO Training Curve](checkpoints_blind/ppo_training_curve.png)

### Adaptive Reading

The learned policies dynamically adjust read count based on question difficulty:

| Strategy | Gold=2 Reads | Gold=4 Reads | Adaptive? |
|---|---|---|---|
| Greedy (K) | K | K | No |
| BC | 2.03 | 4.00 | Yes |
| **BC + PPO** | **2.01** | **3.97** | **Yes** |

![Adaptive Reads](checkpoints_blind/adaptive_reads.png)

### Additional Plots

| | |
|:---:|:---:|
| ![Precision-Recall](checkpoints_blind/precision_recall.png) | ![PR by Gold](checkpoints_blind/precision_recall_by_gold.png) |
| ![F1 by Gold](checkpoints_blind/f1_by_gold_count.png) | ![ROC by Gold](checkpoints_blind/roc_curve_by_gold.png) |

---

## LLM Evaluation

End-to-end evaluation using Qwen3-8B (via Ollama) on **60 hard questions** pre-filtered to exclude questions the LLM can answer without context. Answer accuracy scored via cascaded exact-match, substring containment, and LLM-as-judge.

### LLM Downstream Performance (2WikiMultiHopQA)

| Strategy | Answer Acc | Retrieval F1 | Reads |
|---|---|---|---|
| Best Greedy | 65% | 38% | 3.0 |
| SFT + DPO | 65% | 54.7% | 3.8 |
| BC | 65% | 78% | 2.1 |
| **BC + PPO** | **75%** | **78%** | **2.1** |

BC + PPO achieves the highest answer accuracy (+10% over all other strategies) while reading the fewest paragraphs on average.

---

## Zero-Shot Transfer

To test generalization, we apply the models trained on 2WikiMultiHopQA directly to **HotpotQA** without any fine-tuning, evaluating on 60 pre-filtered hard questions.

### Zero-Shot Transfer (2WikiMultiHopQA → HotpotQA)

| Strategy | HotpotQA F1 | 2Wiki F1 | Δ |
|---|---|---|---|
| Best Greedy | 44.7% | 45.7% | −1.0% |
| SFT + DPO | 40.1% | 54.7% | −14.6% |
| BC | 60.7% | 76.5% | −15.8% |
| **BC + PPO** | **62.7%** | **77.1%** | **−14.4%** |

Key findings:
- **BC + PPO retains the highest absolute F1 on HotpotQA** (62.7%), confirming generalization
- All learned policies degrade on transfer (Δ ≈ −14–16%), while Best Greedy's BoW heuristic is dataset-agnostic (Δ = −1.0%)
- DPO's over-stopping pathology (always reading 1 paragraph) worsens on the new domain

---

## File Structure

```
.
├── train_only.py            # BC + PPO training (no LLM required)
├── train_dpo.py             # SFT + DPO training + ROC curves + significance tests
├── eval_llm.py              # Unified LLM evaluation (2Wiki + HotpotQA transfer)
├── ppo_finetuner.py         # RetrievalSelector, PPOFineTuner, TaskScorer, rewards
├── hotpot_pipeline.py       # Data loading, blind mode, pre-filtering, baselines
├── multi_agent_baseline.py  # RetrievalAgent (LLM interface), trajectory structs
├── plot_results.py          # Generate all visualization plots from train_metrics.json
├── plot_architecture.py     # Generate model architecture figure
├── run_modal.py             # Modal cloud deployment (optional, GPU)
├── requirements.txt         # Python dependencies
├── model_architecture.png   # Architecture diagram
├── checkpoints_blind/       # Saved models & training artifacts
│   ├── split.json           #   Data splits
│   ├── bc_model.pt          #   BC model weights
│   ├── ppo_best.pt          #   Best PPO model weights
│   ├── sft_dpo_model.pt     #   SFT+DPO model weights
│   ├── train_metrics.json   #   All training metrics, ROC data, significance tests
│   └── *.png / *.csv        #   Training curves and plots
└── results/                 # LLM evaluation outputs
    ├── llm_eval_report.txt  #   Formatted evaluation tables
    └── llm_eval_results.json #  Machine-readable results
```

---

## How to Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with `qwen3:8b` (for LLM evaluation only)

### Setup

```bash
pip install -r requirements.txt

# For LLM evaluation only:
ollama pull qwen3:8b
```

### Training

```bash
# Train BC + PPO (no LLM needed)
python train_only.py              # full run (~30 min CPU)
python train_only.py --small      # quick test (~2 min)

# Train SFT + DPO + generate ROC curves
python train_dpo.py

# Regenerate plots from saved metrics
python plot_results.py
```

### LLM Evaluation

```bash
# Full evaluation: 60 hard questions x 2 datasets (requires Ollama)
python eval_llm.py

# Quick test: 20 questions x 2 datasets
python eval_llm.py --small
```

### Cloud Deployment (Optional)

```bash
pip install modal && modal setup

# Run LLM eval on Modal (GPU: A10G)
modal run run_modal.py::run_eval
modal run run_modal.py::run_eval --small
```
