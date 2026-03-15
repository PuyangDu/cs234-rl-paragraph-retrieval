"""
Unified LLM Evaluation: 2WikiMultiHopQA + Zero-Shot HotpotQA Transfer.

Evaluates 4 strategies (Best Greedy, SFT+DPO, BC, BC+PPO) on
60 hard pre-filtered questions with LLM answer scoring (qwen3:8b).

Group 1: 2WikiMultiHopQA (in-domain)
Group 2: HotpotQA (zero-shot transfer)

Outputs:
  results/llm_eval_results.json   — full metrics for both datasets
  results/llm_eval_report.txt     — formatted tables for paper

Requires:
  - Ollama running with qwen3:8b
  - checkpoints_blind/bc_model.pt, ppo_best.pt, sft_dpo_model.pt
  - checkpoints_blind/split.json (for 2Wiki eval set)

Usage:
    python eval_llm.py                 # full eval (60 questions × 2 datasets)
    python eval_llm.py --small         # quick test (20 questions × 2 datasets)
"""

import ast
import json
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter

from hotpot_pipeline import (
    setup_scorer, run_baseline, load_hotpot_data,
    _anonymize_titles, _box, _json_default,
)
from ppo_finetuner import PPOFineTuner, TaskScorer
from multi_agent_baseline import RetrievalAgent

# ======================================================================
#  Constants
# ======================================================================

CKPT_DIR = "checkpoints_blind"
OUT_DIR = "results"
K_BUDGET = 5


# ======================================================================
#  Helpers
# ======================================================================

def _fix_supporting_titles(examples):
    """Normalise supporting_titles from JSON (list/str → set)."""
    for ex in examples.values():
        st = ex["supporting_titles"]
        if isinstance(st, str):
            try:
                st = ast.literal_eval(st)
            except (ValueError, SyntaxError):
                st = set()
        ex["supporting_titles"] = set(st) if not isinstance(st, set) else st


def prefilter_hard(examples, target_count, cache_path):
    """Keep only questions the LLM cannot answer without context.

    Uses a dataset-specific cache file so 2Wiki and HotpotQA caches
    do not collide.
    """
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    scorer = setup_scorer(examples)
    agent = RetrievalAgent(agent_id=0, model="qwen3:8b")

    failed = {}
    passed = 0
    for q_id, ex in examples.items():
        if q_id in cache:
            llm_knows = cache[q_id]
        else:
            traj = agent.solve(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], strategy="no_context",
            )
            score = scorer.score_answer(q_id, traj.final_answer or "")
            llm_knows = score > 0.8
            cache[q_id] = llm_knows

        if llm_knows:
            passed += 1
        else:
            failed[q_id] = ex
        if len(failed) >= target_count:
            break

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"    Kept {len(failed)} hard questions, skipped {passed} easy ones")
    return failed


def find_best_greedy_k(examples, scorer):
    """Find the best K for greedy strategy by retrieval F1 (K=1..7)."""
    best_m, best_k = None, 3
    for k in range(1, 8):
        m, _ = run_baseline(examples, scorer, "greedy",
                            max_reads=k, label=f"Greedy ({k})")
        if best_m is None or m["f1"] > best_m["f1"]:
            best_m, best_k = m, k
    print(f"    Best greedy: K={best_k}, F1={best_m['f1']:.1%}, "
          f"Acc={best_m['accuracy']:.1%}")
    return best_k, best_m


def eval_one_dataset(dataset_label, examples, models, n_target, cache_path):
    """Run full LLM eval on one dataset.

    Returns dict  { strategy_name: metrics_dict }
    """
    _box(f"LLM Eval — {dataset_label}")

    # Pre-filter
    print(f"\n  Pre-filtering to {n_target} hard questions...")
    hard = prefilter_hard(examples, n_target, cache_path)
    print(f"  Final: {len(hard)} questions for LLM eval")

    gc = Counter(len(ex["supporting_titles"]) for ex in hard.values())
    print(f"  Gold distribution: "
          + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc.items())))

    scorer = setup_scorer(hard)
    results = {}

    # 1) Best Greedy
    print(f"\n  [{dataset_label}] Best Greedy (searching K=1..7)...")
    best_k, greedy_m = find_best_greedy_k(hard, scorer)
    greedy_m["strategy"] = f"Best Greedy (K={best_k})"
    results["Best Greedy"] = greedy_m

    # 2) SFT+DPO
    print(f"\n  [{dataset_label}] SFT+DPO...")
    dpo_m, _ = models["dpo"].eval_policy(hard, scorer, K_BUDGET,
                                          label="SFT+DPO")
    results["SFT+DPO"] = dpo_m

    # 3) BC
    print(f"\n  [{dataset_label}] BC...")
    bc_m, _ = models["bc"].eval_policy(hard, scorer, K_BUDGET, label="BC")
    results["BC"] = bc_m

    # 4) BC+PPO
    print(f"\n  [{dataset_label}] BC+PPO...")
    ppo_m, _ = models["ppo"].eval_policy(hard, scorer, K_BUDGET,
                                          label="BC+PPO")
    results["BC+PPO"] = ppo_m

    return results


# ======================================================================
#  Report generation
# ======================================================================

def format_report(wiki2_results, hotpot_results, n_target, strategies):
    """Build list of report lines."""
    lines = []
    lines.append("=" * 70)
    lines.append("  LLM Evaluation Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Questions per dataset: {n_target} (hard, pre-filtered)")
    lines.append(f"  LLM: qwen3:8b  |  Budget: K={K_BUDGET}")
    lines.append("=" * 70)
    lines.append("")

    # --- Table 1: LLM Downstream Performance (2Wiki) ---
    lines.append("  Table 1 — LLM Downstream Performance (2WikiMultiHopQA)")
    lines.append(f"  {'Strategy':<18} {'Answer Acc':>10} {'Retrieval F1':>13} "
                 f"{'Reads':>6}")
    lines.append(f"  {'─'*18} {'─'*10} {'─'*13} {'─'*6}")
    for s in strategies:
        m = wiki2_results[s]
        lines.append(f"  {s:<18} {m['accuracy']:>9.0%} "
                     f"{m['f1']:>12.1%} {m['avg_reads']:>5.1f}")
    lines.append("")

    # --- Table 2: Zero-Shot Transfer ---
    lines.append("  Table 2 — Zero-Shot Transfer "
                 "(2WikiMultiHopQA → HotpotQA)")
    lines.append(f"  {'Strategy':<18} {'HotpotQA F1':>12} {'2Wiki F1':>9} "
                 f"{'Δ':>7}")
    lines.append(f"  {'─'*18} {'─'*12} {'─'*9} {'─'*7}")
    for s in strategies:
        h_f1 = hotpot_results[s]["f1"]
        w_f1 = wiki2_results[s]["f1"]
        delta = h_f1 - w_f1
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {s:<18} {h_f1:>11.1%} {w_f1:>8.1%} "
                     f"{sign}{delta:>5.1%}")
    lines.append("")

    # --- Detailed per-strategy ---
    for ds_label, res in [("2WikiMultiHopQA", wiki2_results),
                          ("HotpotQA (transfer)", hotpot_results)]:
        lines.append(f"  {ds_label} — Detailed Metrics")
        lines.append(f"  {'Strategy':<18} {'Acc':>5} {'Prec':>6} {'Rec':>6} "
                     f"{'F1':>6} {'Reads':>6}")
        lines.append(f"  {'─'*18} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for s in strategies:
            m = res[s]
            lines.append(
                f"  {s:<18} {m['accuracy']:>4.0%} "
                f"{m.get('precision',0):>5.1%} "
                f"{m.get('recall',0):>5.1%} "
                f"{m['f1']:>5.1%} "
                f"{m['avg_reads']:>5.1f}")
        lines.append("")

    return lines


# ======================================================================
#  Main
# ======================================================================

def main(small=False):
    _box("Unified LLM Evaluation (2Wiki + HotpotQA Transfer)")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    Path(OUT_DIR).mkdir(exist_ok=True)

    N_TARGET = 20 if small else 60

    # ------------------------------------------------------------------
    # [1/5] Load trained models
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading trained models...")

    bc_path  = os.path.join(CKPT_DIR, "bc_model.pt")
    ppo_path = os.path.join(CKPT_DIR, "ppo_best.pt")
    dpo_path = os.path.join(CKPT_DIR, "sft_dpo_model.pt")

    for p in [bc_path, ppo_path, dpo_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} not found. Run train_only.py first.")

    scorer_dummy = TaskScorer()  # needed for PPOFineTuner init

    bc_tuner  = PPOFineTuner(scorer_dummy, device="cpu", blind=True)
    bc_tuner.load_model(bc_path)
    print(f"  BC       ← {bc_path}")

    ppo_tuner = PPOFineTuner(scorer_dummy, device="cpu", blind=True)
    ppo_tuner.load_model(ppo_path)
    print(f"  BC+PPO   ← {ppo_path}")

    dpo_tuner = PPOFineTuner(scorer_dummy, device="cpu", blind=True)
    dpo_tuner.load_model(dpo_path)
    print(f"  SFT+DPO  ← {dpo_path}")

    models = {"bc": bc_tuner, "ppo": ppo_tuner, "dpo": dpo_tuner}

    # ------------------------------------------------------------------
    # [2/5] Load 2WikiMultiHopQA eval set
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading 2WikiMultiHopQA eval set...")
    split_path = os.path.join(CKPT_DIR, "split.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. Run train_only.py first.")

    with open(split_path) as f:
        saved = json.load(f)
    wiki2_examples = saved["eval"]
    _fix_supporting_titles(wiki2_examples)
    wiki2_examples = _anonymize_titles(wiki2_examples)
    print(f"  2Wiki eval pool: {len(wiki2_examples)} examples")

    # ------------------------------------------------------------------
    # [3/5] Load HotpotQA (for transfer)
    # ------------------------------------------------------------------
    print(f"\n[3/5] Loading HotpotQA data...")
    n_hotpot_pool = 500 if small else 2000
    hotpot_examples = load_hotpot_data("train", max_examples=n_hotpot_pool)
    hotpot_examples = _anonymize_titles(hotpot_examples)
    print(f"  HotpotQA pool: {len(hotpot_examples)} examples")

    # ------------------------------------------------------------------
    # [4/5] Run LLM eval on both datasets
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running LLM eval "
          f"(target={N_TARGET} hard Qs per dataset)...")

    wiki2_results = eval_one_dataset(
        "2WikiMultiHopQA", wiki2_examples, models, N_TARGET,
        os.path.join(OUT_DIR, "prefilter_cache_2wiki.json"),
    )

    hotpot_results = eval_one_dataset(
        "HotpotQA", hotpot_examples, models, N_TARGET,
        os.path.join(OUT_DIR, "prefilter_cache_hotpot.json"),
    )

    # ------------------------------------------------------------------
    # [5/5] Generate report and save
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating report...")

    strategies = ["Best Greedy", "SFT+DPO", "BC", "BC+PPO"]

    report_lines = format_report(wiki2_results, hotpot_results,
                                 N_TARGET, strategies)
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_path = os.path.join(OUT_DIR, "llm_eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"  Saved report  → {report_path}")

    # Save JSON metrics
    all_metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_target": N_TARGET,
            "budget": K_BUDGET,
            "llm": "qwen3:8b",
            "blind": True,
        },
        "wiki2": {s: wiki2_results[s] for s in strategies},
        "hotpot": {s: hotpot_results[s] for s in strategies},
    }
    metrics_path = os.path.join(OUT_DIR, "llm_eval_results.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    print(f"  Saved metrics → {metrics_path}")

    print("\nDone!")


if __name__ == "__main__":
    main(small="--small" in sys.argv)
