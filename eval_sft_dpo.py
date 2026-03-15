"""Evaluate SFT+DPO model and add its metrics to train_metrics.json."""
import json
import os
import sys
import numpy as np
from collections import defaultdict

# Monkey-patch to avoid Ollama connection
import multi_agent_baseline
multi_agent_baseline.RetrievalAgent._verify_connection = lambda self: None

from multi_agent_baseline import RetrievalAgent
from ppo_finetuner import PPOFineTuner
from hotpot_pipeline import _compute_retrieval_metrics

CKPT_DIR = "checkpoints_blind"
METRICS_PATH = os.path.join(CKPT_DIR, "train_metrics.json")
MODEL_PATH = os.path.join(CKPT_DIR, "sft_dpo_model.pt")
SPLIT_PATH = os.path.join(CKPT_DIR, "split.json")
K_BUDGET = 5


def _per_q_f1(supp_read, total_reads, n_gold):
    if n_gold == 0:
        return 1.0 if total_reads == 0 else 0.0
    rec = supp_read / n_gold
    prec = supp_read / max(1, total_reads)
    return 2 * prec * rec / max(1e-9, prec + rec)


def eval_policy_retrieval(fine_tuner, examples, max_steps, label):
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
    return m


def main():
    print("Loading eval split...")
    with open(SPLIT_PATH) as f:
        splits = json.load(f)
    eval_examples = splits["eval"]
    print(f"  Eval set: {len(eval_examples)} examples")

    print(f"Loading SFT+DPO model from {MODEL_PATH}...")
    fine_tuner = PPOFineTuner(scorer=None, device="cpu", blind=True,
                              lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03)
    import torch
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    fine_tuner.model.load_state_dict(state_dict)

    print("Evaluating SFT+DPO...")
    sft_dpo_ret = eval_policy_retrieval(fine_tuner, eval_examples, K_BUDGET, "SFT+DPO")
    print(f"  SFT+DPO:  P={sft_dpo_ret['precision']:.1%}  "
          f"R={sft_dpo_ret['recall']:.1%}  F1={sft_dpo_ret['f1']:.1%}  "
          f"Reads={sft_dpo_ret['avg_reads']:.2f}")

    # Strip per_q_f1 for JSON storage
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

    # Load existing metrics and add sft_dpo
    print("Updating train_metrics.json...")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    metrics["sft_dpo_retrieval"] = _strip_per_q(sft_dpo_ret)

    # Significance tests: BC vs BC+PPO
    bc_per_q = None
    ppo_per_q = None
    # We need per_q_f1 for BC and PPO; re-evaluate if needed, or load from model
    # For now, add significance between SFT+DPO vs BC and SFT+DPO vs PPO
    # Since we have sft_dpo per_q_f1 we can store it for later use
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved updated metrics → {METRICS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
