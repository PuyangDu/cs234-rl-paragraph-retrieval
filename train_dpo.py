"""
Train BC+DPO and compute ROC data for all models.

Loads the existing BC model (bc_model.pt), applies DPO fine-tuning,
evaluates all models, computes per-paragraph ROC scores, and updates
train_metrics.json + report.txt.

Does NOT retrain BC or PPO.

Usage:
    python train_dpo.py
    python train_dpo.py --small   # quick test with fewer DPO epochs
"""

import json
import os
import sys
import random
import csv
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

# Monkey-patch: allow RetrievalAgent to work without Ollama
import multi_agent_baseline
multi_agent_baseline.RetrievalAgent._verify_connection = lambda self: None

from multi_agent_baseline import NUM_PARAGRAPHS, RetrievalAgent, _STOP_WORDS
from ppo_finetuner import PPOFineTuner
from hotpot_pipeline import _compute_retrieval_metrics, _box, _json_default

CKPT_DIR = "checkpoints_blind"
METRICS_PATH = os.path.join(CKPT_DIR, "train_metrics.json")
BC_MODEL_PATH = os.path.join(CKPT_DIR, "bc_model.pt")
PPO_MODEL_PATH = os.path.join(CKPT_DIR, "ppo_best.pt")
DPO_MODEL_PATH = os.path.join(CKPT_DIR, "sft_dpo_model.pt")
SPLIT_PATH = os.path.join(CKPT_DIR, "split.json")
K_BUDGET = 5


def _per_q_f1(supp_read, total_reads, n_gold):
    if n_gold == 0:
        return 1.0 if total_reads == 0 else 0.0
    rec = supp_read / n_gold
    prec = supp_read / max(1, total_reads)
    return 2 * prec * rec / max(1e-9, prec + rec)


def eval_policy_retrieval(fine_tuner, examples, max_steps, label):
    """Eval a policy model with per-gold-count breakdown (deterministic argmax)."""
    fine_tuner.model.eval()
    total = total_reads = total_supp = total_gold = 0
    n_correct = 0
    per_q_f1s = []
    by_gc = defaultdict(lambda: {"total": 0, "reads": 0, "supp": 0, "gold": 0, "f1s": []})
    for q_id, ex in examples.items():
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        traj = agent.solve_with_policy(
            q_id, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], policy=fine_tuner, max_steps=max_steps,
            training=True,
        )
        total += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read
        n_gold = len(ex["supporting_titles"])
        total_gold += n_gold
        if n_gold > 0 and traj.num_supporting_read / n_gold >= 0.5:
            n_correct += 1
        qf1 = _per_q_f1(traj.num_supporting_read, traj.total_reads, n_gold)
        per_q_f1s.append(qf1)
        by_gc[n_gold]["total"] += 1
        by_gc[n_gold]["reads"] += traj.total_reads
        by_gc[n_gold]["supp"] += traj.num_supporting_read
        by_gc[n_gold]["gold"] += n_gold
        by_gc[n_gold]["f1s"].append(qf1)

    recall = total_supp / max(1, total_gold)
    prec = total_supp / max(1, total_reads)
    f1 = 2 * prec * recall / max(1e-9, prec + recall)
    m = {
        "strategy": label,
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "retrieval_acc": n_correct / max(1, total),
        "avg_reads": total_reads / max(1, total),
        "avg_supporting_found": total_supp / max(1, total),
        "total": total,
        "per_q_f1": per_q_f1s,
    }
    for gc, acc in sorted(by_gc.items()):
        sub = _compute_retrieval_metrics(acc["total"], acc["reads"], acc["supp"], acc["gold"])
        sub["total"] = acc["total"]
        sub["per_q_f1"] = acc["f1s"]
        m[f"gold_{gc}"] = sub
    fine_tuner.model.train()
    return m


def compute_policy_roc_scores(fine_tuner, examples):
    """Compute per-paragraph relevance scores from a policy model.

    For each (question, paragraph_i), uses the model's step-0 softmax
    probability P(read_i | initial_state) as the relevance score.

    Returns dict with:
      - "scores": list of floats (one per paragraph across all questions)
      - "labels": list of 0/1 (gold=1, distractor=0)
      - "gold_2": {"scores": [...], "labels": [...]}
      - "gold_4": {"scores": [...], "labels": [...]}
    """
    fine_tuner.model.eval()
    all_scores = []
    all_labels = []
    by_gc = defaultdict(lambda: {"scores": [], "labels": []})

    for q_id, ex in examples.items():
        paragraphs = ex["paragraphs"][:NUM_PARAGRAPHS]
        supp_titles = ex["supporting_titles"]
        n = len(paragraphs)
        n_gold = len(supp_titles)

        # Build step-0 state (nothing read)
        context = (
            f"Task: {ex['question']}\n"
            f"Titles: {' | '.join(p[0] for p in paragraphs)}\n"
            f"Read: Nothing yet\n"
            f"Step: 0"
        )
        feat = fine_tuner.trainer.extract_features(
            context, question=ex["question"], paragraphs=paragraphs)

        with torch.no_grad():
            logits, _ = fine_tuner.model(feat.unsqueeze(0))

        # Softmax over read actions only (exclude answer logit)
        read_logits = logits[0, :n]
        scores = F.softmax(read_logits, dim=0).cpu().numpy()

        for i in range(n):
            is_gold = 1 if paragraphs[i][0] in supp_titles else 0
            s = float(scores[i])
            all_scores.append(s)
            all_labels.append(is_gold)
            by_gc[n_gold]["scores"].append(s)
            by_gc[n_gold]["labels"].append(is_gold)

    fine_tuner.model.train()
    result = {"scores": all_scores, "labels": all_labels}
    for gc, data in sorted(by_gc.items()):
        result[f"gold_{gc}"] = data
    return result


def compute_greedy_roc_scores(examples):
    """Compute word-overlap scores for ROC (greedy baseline)."""
    all_scores = []
    all_labels = []
    by_gc = defaultdict(lambda: {"scores": [], "labels": []})

    for q_id, ex in examples.items():
        paragraphs = ex["paragraphs"][:NUM_PARAGRAPHS]
        supp_titles = ex["supporting_titles"]
        n_gold = len(supp_titles)
        q_words = set(ex["question"].lower().split()) - _STOP_WORDS

        for i in range(len(paragraphs)):
            text = " ".join(paragraphs[i][1])
            p_words = set(text.lower().split()) - _STOP_WORDS
            overlap = len(q_words & p_words) / max(1, len(q_words))
            is_gold = 1 if paragraphs[i][0] in supp_titles else 0
            all_scores.append(overlap)
            all_labels.append(is_gold)
            by_gc[n_gold]["scores"].append(overlap)
            by_gc[n_gold]["labels"].append(is_gold)

    result = {"scores": all_scores, "labels": all_labels}
    for gc, data in sorted(by_gc.items()):
        result[f"gold_{gc}"] = data
    return result


def compute_random_roc_scores(examples):
    """Compute random scores for ROC baseline."""
    rng = np.random.RandomState(42)
    all_scores = []
    all_labels = []
    by_gc = defaultdict(lambda: {"scores": [], "labels": []})

    for q_id, ex in examples.items():
        paragraphs = ex["paragraphs"][:NUM_PARAGRAPHS]
        supp_titles = ex["supporting_titles"]
        n_gold = len(supp_titles)

        for i in range(len(paragraphs)):
            s = float(rng.rand())
            is_gold = 1 if paragraphs[i][0] in supp_titles else 0
            all_scores.append(s)
            all_labels.append(is_gold)
            by_gc[n_gold]["scores"].append(s)
            by_gc[n_gold]["labels"].append(is_gold)

    result = {"scores": all_scores, "labels": all_labels}
    for gc, data in sorted(by_gc.items()):
        result[f"gold_{gc}"] = data
    return result


def _strip_per_q(d):
    out = {}
    for k, v in d.items():
        if k == "per_q_f1":
            continue
        if isinstance(v, dict):
            out[k] = _strip_per_q(v)
        else:
            out[k] = v
    return out


def compute_auc(scores, labels):
    """Compute AUC from scores and labels without sklearn."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = 0.0
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def main(small=False):
    _box("Train BC+DPO & Compute ROC Data")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # [1/5] Load data splits
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data splits...")
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    bc_examples = splits["bc"]
    bc_dev_examples = splits["bc_dev"]
    eval_examples = splits["eval"]
    print(f"  BC train: {len(bc_examples)}")
    print(f"  BC dev:   {len(bc_dev_examples)}")
    print(f"  Eval:     {len(eval_examples)}")

    # ------------------------------------------------------------------
    # [2/5] Train DPO: load BC model → DPO fine-tune
    # ------------------------------------------------------------------
    print(f"\n[2/5] Training BC+DPO...")
    print(f"  Loading BC model from {BC_MODEL_PATH}...")
    dpo_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=True,
        lr=5e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    dpo_tuner.load_model(BC_MODEL_PATH)

    dpo_epochs = 5 if small else 15
    dpo_patience = 3 if small else 5
    dpo_history = dpo_tuner.dpo_train(
        bc_examples,
        max_steps=K_BUDGET,
        dpo_epochs=dpo_epochs,
        batch_size=32,
        lr=5e-5,
        beta=0.1,
        dev_examples=bc_dev_examples,
        patience=dpo_patience,
    )

    dpo_tuner.save_model(DPO_MODEL_PATH)
    print(f"  Saved DPO model → {DPO_MODEL_PATH}")

    # ------------------------------------------------------------------
    # [3/6] Eval all models on eval set (retrieval-only, no LLM)
    # ------------------------------------------------------------------
    print(f"\n[3/6] Evaluating all models on eval set ({len(eval_examples)} examples)...")

    # Load BC model
    bc_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=True,
        lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    bc_tuner.load_model(BC_MODEL_PATH)

    # Load PPO model
    ppo_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=True,
        lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    ppo_tuner.load_model(PPO_MODEL_PATH)

    # Evaluate all three models (collects per_q_f1 for significance tests)
    bc_ret = eval_policy_retrieval(bc_tuner, eval_examples, K_BUDGET, "BC-only")
    print(f"  BC:       P={bc_ret['precision']:.1%}  "
          f"R={bc_ret['recall']:.1%}  F1={bc_ret['f1']:.1%}  "
          f"Reads={bc_ret['avg_reads']:.2f}")

    ppo_ret = eval_policy_retrieval(ppo_tuner, eval_examples, K_BUDGET, "PPO (ours)")
    print(f"  BC+PPO:   P={ppo_ret['precision']:.1%}  "
          f"R={ppo_ret['recall']:.1%}  F1={ppo_ret['f1']:.1%}  "
          f"Reads={ppo_ret['avg_reads']:.2f}")

    # DPO eval
    dpo_ret = eval_policy_retrieval(dpo_tuner, eval_examples, K_BUDGET, "SFT+DPO")
    print(f"  SFT+DPO:  P={dpo_ret['precision']:.1%}  "
          f"R={dpo_ret['recall']:.1%}  F1={dpo_ret['f1']:.1%}  "
          f"Reads={dpo_ret['avg_reads']:.2f}")

    # ------------------------------------------------------------------
    # [4/6] Compute ROC scores for all models
    # ------------------------------------------------------------------
    print(f"\n[4/6] Computing ROC scores for all models...")

    print("  Computing BC ROC scores...")
    bc_roc = compute_policy_roc_scores(bc_tuner, eval_examples)
    bc_auc = compute_auc(bc_roc["scores"], bc_roc["labels"])
    print(f"    BC AUC = {bc_auc:.4f}")

    print("  Computing BC+PPO ROC scores...")
    ppo_roc = compute_policy_roc_scores(ppo_tuner, eval_examples)
    ppo_auc = compute_auc(ppo_roc["scores"], ppo_roc["labels"])
    print(f"    BC+PPO AUC = {ppo_auc:.4f}")

    print("  Computing BC+DPO ROC scores...")
    dpo_roc = compute_policy_roc_scores(dpo_tuner, eval_examples)
    dpo_auc = compute_auc(dpo_roc["scores"], dpo_roc["labels"])
    print(f"    BC+DPO AUC = {dpo_auc:.4f}")

    print("  Computing Greedy ROC scores...")
    greedy_roc = compute_greedy_roc_scores(eval_examples)
    greedy_auc = compute_auc(greedy_roc["scores"], greedy_roc["labels"])
    print(f"    Greedy AUC = {greedy_auc:.4f}")

    print("  Computing Random ROC scores...")
    random_roc = compute_random_roc_scores(eval_examples)
    random_auc = compute_auc(random_roc["scores"], random_roc["labels"])
    print(f"    Random AUC = {random_auc:.4f}")

    # Per-gold-count AUC
    roc_summary = {}
    for name, roc_data in [("BC", bc_roc), ("BC+PPO", ppo_roc),
                           ("SFT+DPO", dpo_roc), ("Greedy", greedy_roc),
                           ("Random", random_roc)]:
        entry = {"auc": compute_auc(roc_data["scores"], roc_data["labels"])}
        for key in roc_data:
            if key.startswith("gold_"):
                gc_data = roc_data[key]
                entry[key] = {
                    "auc": compute_auc(gc_data["scores"], gc_data["labels"]),
                }
        roc_summary[name] = entry

    # ------------------------------------------------------------------
    # [5/6] Pairwise significance tests (BC vs PPO vs DPO)
    # ------------------------------------------------------------------
    print(f"\n[5/6] Pairwise significance tests...")

    def paired_permutation_test(f1s_a, f1s_b, n_perm=10000, seed=42):
        """One-sided paired permutation test. H1: mean(a) > mean(b)."""
        rng = np.random.RandomState(seed)
        a, b = np.array(f1s_a, dtype=float), np.array(f1s_b, dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = a - b
        obs_diff = np.mean(d)
        count = sum(1 for _ in range(n_perm)
                    if np.mean(d * rng.choice([-1, 1], size=n)) >= obs_diff)
        return count / n_perm

    def _sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    # All pairwise comparisons
    sig_pairs = [
        ("BC+PPO", "BC-only",  ppo_ret, bc_ret),
        ("SFT+DPO", "BC-only", dpo_ret, bc_ret),
        ("SFT+DPO", "BC+PPO",  dpo_ret, ppo_ret),
    ]
    sig_results = {}
    for label_a, label_b, ret_a, ret_b in sig_pairs:
        # Overall
        p_val = paired_permutation_test(ret_a["per_q_f1"], ret_b["per_q_f1"])
        key = f"{label_a} vs {label_b}"
        sig_results[key] = {
            "f1_a": ret_a["f1"], "f1_b": ret_b["f1"],
            "p_value": p_val, "significant": p_val < 0.05,
        }
        star = _sig_star(p_val)
        print(f"  {key}: p={p_val:.4f}  {star}")

        # Per gold-count
        gold_counts = sorted({int(k.split('_')[1]) for k in ret_a if k.startswith('gold_')})
        for gc in gold_counts:
            gk = f"gold_{gc}"
            sub_a = ret_a.get(gk, {})
            sub_b = ret_b.get(gk, {})
            if "per_q_f1" in sub_a and "per_q_f1" in sub_b:
                p_gc = paired_permutation_test(sub_a["per_q_f1"], sub_b["per_q_f1"])
                gc_key = f"{label_a} vs {label_b} (gold={gc})"
                sig_results[gc_key] = {
                    "f1_a": sub_a["f1"], "f1_b": sub_b["f1"],
                    "p_value": p_gc, "significant": p_gc < 0.05,
                }
                star = _sig_star(p_gc)
                print(f"    gold={gc}: p={p_gc:.4f}  {star}")

    # ------------------------------------------------------------------
    # [6/6] Update train_metrics.json and report.txt
    # ------------------------------------------------------------------
    print(f"\n[6/6] Updating metrics and report...")

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    # Update retrieval metrics (keep per_q_f1 for reproducibility)
    metrics["bc_retrieval"] = bc_ret
    metrics["ppo_retrieval"] = ppo_ret
    metrics["sft_dpo_retrieval"] = dpo_ret
    metrics["dpo_loss_history"] = dpo_history
    metrics["significance_tests_pairwise"] = sig_results

    # Save ROC data (scores + labels for each model)
    metrics["roc_data"] = {
        "bc": {"scores": bc_roc["scores"], "labels": bc_roc["labels"],
               "auc": bc_auc},
        "ppo": {"scores": ppo_roc["scores"], "labels": ppo_roc["labels"],
                "auc": ppo_auc},
        "dpo": {"scores": dpo_roc["scores"], "labels": dpo_roc["labels"],
                "auc": dpo_auc},
        "greedy": {"scores": greedy_roc["scores"], "labels": greedy_roc["labels"],
                   "auc": greedy_auc},
        "random": {"scores": random_roc["scores"], "labels": random_roc["labels"],
                   "auc": random_auc},
    }
    # Per-gold-count ROC data
    for model_key, roc_data in [("bc", bc_roc), ("ppo", ppo_roc),
                                ("dpo", dpo_roc), ("greedy", greedy_roc),
                                ("random", random_roc)]:
        for gc_key in roc_data:
            if gc_key.startswith("gold_"):
                metrics["roc_data"][model_key][gc_key] = roc_data[gc_key]

    metrics["roc_summary"] = roc_summary

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"  Saved updated metrics → {METRICS_PATH}")

    # Save DPO loss curve CSV
    dpo_csv = os.path.join(CKPT_DIR, "dpo_loss_curve.csv")
    with open(dpo_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "dev_loss"])
        for ep in dpo_history:
            w.writerow([ep["epoch"], f"{ep['train_loss']:.6f}",
                        f"{ep.get('dev_loss', ''):.6f}" if "dev_loss" in ep else ""])
    print(f"  Saved DPO loss curve → {dpo_csv}")

    # Update report.txt — replace DPO section if exists, else append
    report_path = os.path.join(CKPT_DIR, "report.txt")
    with open(report_path, "r") as f:
        existing_report = f.read()

    # Strip old DPO section if present
    marker = "  BC+DPO Training & ROC Analysis"
    if marker in existing_report:
        idx = existing_report.index(marker)
        # Find the preceding ====== line
        prev_eq = existing_report.rfind("=" * 70, 0, idx)
        if prev_eq > 0:
            existing_report = existing_report[:prev_eq].rstrip()

    # Build DPO + ROC section
    dpo_section = []
    dpo_section.append("\n\n" + "=" * 70)
    dpo_section.append("  BC+DPO Training & ROC Analysis")
    dpo_section.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    dpo_section.append("=" * 70)
    dpo_section.append("")
    dpo_section.append("DPO Configuration:")
    dpo_section.append(f"  Base model: BC (bc_model.pt)")
    dpo_section.append(f"  β = 0.1, lr = 5e-5, epochs = {dpo_epochs}, patience = {dpo_patience}")
    dpo_section.append(f"  Preference pairs: oracle gold vs distractor at each decision state")
    dpo_section.append("")

    # DPO loss curve
    dpo_section.append("DPO Loss Curve:")
    for ep in dpo_history:
        dev_s = f"  dev={ep['dev_loss']:.4f}" if "dev_loss" in ep else ""
        dpo_section.append(f"  epoch {ep['epoch']:>2}  train={ep['train_loss']:.4f}{dev_s}")
    dpo_section.append("")

    # SFT+DPO retrieval results
    dpo_section.append("SFT+DPO Retrieval Results:")
    dpo_section.append(f"  P={dpo_ret['precision']:.1%}  R={dpo_ret['recall']:.1%}  "
                       f"F1={dpo_ret['f1']:.1%}  Reads={dpo_ret['avg_reads']:.2f}")
    for gc in sorted(int(k.split('_')[1]) for k in dpo_ret if k.startswith('gold_')):
        sub = dpo_ret[f"gold_{gc}"]
        dpo_section.append(f"  Gold={gc}: P={sub['precision']:.1%}  R={sub['recall']:.1%}  "
                           f"F1={sub['f1']:.1%}  Reads={sub.get('avg_reads', 0):.2f}  n={sub['total']}")
    dpo_section.append("")

    # ROC/AUC summary
    dpo_section.append("ROC AUC Summary:")
    dpo_section.append(f"  {'Strategy':<15} {'Overall':>8} " +
                       " ".join(f"{'Gold=' + k.split('_')[1]:>8}" for k in sorted(roc_summary["BC"].keys()) if k.startswith("gold_")))
    dpo_section.append(f"  {'─'*15} {'─'*8} " +
                       " ".join("─" * 8 for k in sorted(roc_summary["BC"].keys()) if k.startswith("gold_")))
    for name in ["Random", "Greedy", "BC", "BC+PPO", "SFT+DPO"]:
        entry = roc_summary[name]
        line = f"  {name:<15} {entry['auc']:>7.4f} "
        for k in sorted(entry.keys()):
            if k.startswith("gold_"):
                line += f" {entry[k]['auc']:>7.4f} "
        dpo_section.append(line)

    # Pairwise significance tests
    dpo_section.append("")
    dpo_section.append("Pairwise Significance Tests (paired permutation, one-sided, n=10000):")
    for test_key, test_val in sig_results.items():
        star = _sig_star(test_val["p_value"])
        dpo_section.append(f"  {test_key}:  p={test_val['p_value']:.4f}  {star}")

    with open(report_path, "w") as f:
        f.write(existing_report.rstrip() + "\n" + "\n".join(dpo_section) + "\n")
    print(f"  Updated report → {report_path}")

    print("\nDone! Now run: python plot_results.py  (to generate ROC curves and all plots)")


if __name__ == "__main__":
    main(small="--small" in sys.argv)
