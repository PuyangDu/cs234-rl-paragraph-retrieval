"""Generate extra result plots from train_metrics.json.

Usage:
    python plot_results.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = 'checkpoints_blind/train_metrics.json'
OUT_DIR = 'checkpoints_blind'


def load():
    with open(METRICS) as f:
        return json.load(f)


def _get_src(m, strategy):
    """Return the metrics dict for a given strategy name."""
    bc = m['bc_retrieval']
    ppo = m['ppo_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval')
    bl = {b['strategy']: b for b in m['baselines_retrieval']}
    if strategy == 'BC':
        return bc
    elif strategy == 'BC+PPO':
        return ppo
    elif strategy == 'SFT+DPO':
        return sft_dpo
    else:
        return bl[strategy]


def plot_f1_by_gold(m):
    """Grouped bar chart: retrieval F1 broken down by gold=2 vs gold=4.
    Only Random(7), Greedy(3), BC, BC+PPO, SFT+DPO."""
    strategies = ['Random (7)', 'Greedy (3)', 'BC', 'BC+PPO', 'SFT+DPO']

    gold2_f1, gold4_f1, overall_f1 = [], [], []
    for s in strategies:
        src = _get_src(m, s)
        gold2_f1.append(src['gold_2']['f1'] * 100)
        gold4_f1.append(src['gold_4']['f1'] * 100)
        overall_f1.append(src['f1'] * 100)

    x = np.arange(len(strategies))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w, gold2_f1, w, label='Gold=2', color='#5B9BD5',
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x, gold4_f1, w, label='Gold=4', color='#ED7D31',
                edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w, overall_f1, w, label='Overall', color='#70AD47',
                edgecolor='white', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    # Annotation box: BC, BC+PPO, SFT+DPO gaps (gold2 vs gold4)
    bc_idx = strategies.index('BC')
    ppo_idx = strategies.index('BC+PPO')
    sft_idx = strategies.index('SFT+DPO')
    bc_gap = abs(gold2_f1[bc_idx] - gold4_f1[bc_idx])
    ppo_gap = abs(gold2_f1[ppo_idx] - gold4_f1[ppo_idx])
    sft_gap = abs(gold2_f1[sft_idx] - gold4_f1[sft_idx])
    ax.annotate(
        f'BC gap: {bc_gap:.1f}%  |  BC+PPO gap: {ppo_gap:.1f}%\n'
        f'SFT+DPO gap: {sft_gap:.1f}%',
        xy=(0.98, 0.95), xycoords='axes fraction',
        fontsize=9, ha='right', va='top', color='#C00000',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2CC',
                  edgecolor='#C00000', alpha=0.8))

    ax.set_ylabel('Retrieval F1 (%)', fontsize=12)
    ax.set_title('Retrieval F1 by Gold Paragraph Count (Gold=2 vs Gold=4)',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = f'{OUT_DIR}/f1_by_gold_count.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_adaptive_reads(m):
    """Bar chart showing BC, BC+PPO, SFT+DPO adapt read count to question difficulty."""
    strategies = ['BC', 'BC+PPO', 'SFT+DPO']

    gold2_reads, gold4_reads = [], []
    for s in strategies:
        src = _get_src(m, s)
        gold2_reads.append(src['gold_2']['avg_reads'])
        gold4_reads.append(src['gold_4']['avg_reads'])

    x = np.arange(len(strategies))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - w / 2, gold2_reads, w, label='Gold=2 questions',
                color='#5B9BD5', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w / 2, gold4_reads, w, label='Gold=4 questions',
                color='#ED7D31', edgecolor='white', linewidth=0.5)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold')

    ax.annotate('Adaptive reads\n(learned policy)',
                xy=(1, max(max(gold2_reads), max(gold4_reads)) + 0.8),
                fontsize=9, ha='center', color='#2E7D32',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                          edgecolor='#2E7D32', alpha=0.8))

    # Delta arrows for each strategy
    for i, s in enumerate(strategies):
        gap = gold4_reads[i] - gold2_reads[i]
        ax.annotate('', xy=(i + 0.22, gold4_reads[i]),
                    xytext=(i - 0.22, gold2_reads[i]),
                    arrowprops=dict(arrowstyle='<->', color='#C00000', lw=1.8))
        ax.text(i,
                (gold2_reads[i] + gold4_reads[i]) / 2 + 0.15,
                f'\u0394 = {gap:.2f}', ha='center', fontsize=9,
                color='#C00000', fontweight='bold')

    ax.set_ylabel('Average Number of Reads', fontsize=12)
    ax.set_title(
        'Adaptive Read Count: Learned Policies Read More for Harder Questions',
        fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, max(max(gold2_reads), max(gold4_reads)) + 2.0)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=2, color='#5B9BD5', linestyle=':', alpha=0.4, linewidth=1)
    ax.axhline(y=4, color='#ED7D31', linestyle=':', alpha=0.4, linewidth=1)
    ax.text(-0.3, 2.05, 'ideal for gold=2', fontsize=7,
            color='#5B9BD5', alpha=0.7)
    ax.text(-0.3, 4.05, 'ideal for gold=4', fontsize=7,
            color='#ED7D31', alpha=0.7)
    plt.tight_layout()
    path = f'{OUT_DIR}/adaptive_reads.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved {path}')


def plot_precision_recall(m):
    """Overall Precision-Recall tradeoff with SFT+DPO added."""
    bc = m['bc_retrieval']
    ppo = m['ppo_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval')
    baselines = m['baselines_retrieval']

    random_pts = [(b['recall'], b['precision'], b['strategy'])
                  for b in baselines if 'Random' in b['strategy']]
    greedy_pts = [(b['recall'], b['precision'], b['strategy'])
                  for b in baselines if 'Greedy' in b['strategy']]

    fig, ax = plt.subplots(figsize=(8, 6))
    rr = [p[0] for p in random_pts]
    rp = [p[1] for p in random_pts]
    ax.plot(rr, rp, 'o-', color='#999999', label='Random(K)',
            markersize=5, linewidth=1.5)
    for r, p, lbl in random_pts:
        k = lbl.split('(')[1].rstrip(')')
        ax.annotate(f'K={k}', (r, p), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color='#999999')
    gr = [p[0] for p in greedy_pts]
    gp = [p[1] for p in greedy_pts]
    ax.plot(gr, gp, 's-', color='#2196F3', label='Greedy(K)',
            markersize=5, linewidth=1.5)
    for r, p, lbl in greedy_pts:
        k = lbl.split('(')[1].rstrip(')')
        ax.annotate(f'K={k}', (r, p), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color='#2196F3')
    for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6]:
        r_range = np.linspace(0.01, 0.99, 200)
        p_iso = f1_val * r_range / (2 * r_range - f1_val)
        mask = (p_iso > 0) & (p_iso <= 1)
        ax.plot(r_range[mask], p_iso[mask], '--', color='#E0E0E0',
                linewidth=0.8, alpha=0.7)
        idx = np.argmin(np.abs(r_range - f1_val))
        if mask[idx]:
            ax.annotate(f'F1={f1_val}', (r_range[idx], p_iso[idx]),
                        fontsize=6, color='#BDBDBD')
    ax.scatter(bc['recall'], bc['precision'], marker='D',
               s=120, color='#FF9800', zorder=5, edgecolors='black',
               linewidths=0.8, label=f"BC (F1={bc['f1']:.1%})")
    ax.scatter(ppo['recall'], ppo['precision'], marker='*',
               s=250, color='#F44336', zorder=5, edgecolors='black',
               linewidths=0.8, label=f"BC+PPO (F1={ppo['f1']:.1%})")
    if sft_dpo:
        ax.scatter(sft_dpo['recall'], sft_dpo['precision'], marker='^',
                   s=150, color='#9C27B0', zorder=5, edgecolors='black',
                   linewidths=0.8, label=f"SFT+DPO (F1={sft_dpo['f1']:.1%})")
    ax.annotate('', xy=(ppo['recall'], ppo['precision']),
                xytext=(bc['recall'], bc['precision']),
                arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                lw=1.5, ls='--'))
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Tradeoff: Paragraph Retrieval\n'
                 '(eval set, no LLM)', fontsize=13)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    path = f'{OUT_DIR}/precision_recall.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path}')


def _compute_roc_curve(scores, labels):
    """Compute ROC curve (FPR, TPR) from scores and binary labels."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1]
    fpr_list = [0.0]
    tpr_list = [0.0]
    tp = 0
    fp = 0
    prev_score = None
    for s, l in pairs:
        if prev_score is not None and s != prev_score:
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)
        if l == 1:
            tp += 1
        else:
            fp += 1
        prev_score = s
    fpr_list.append(fp / n_neg)
    tpr_list.append(tp / n_pos)
    return fpr_list, tpr_list


def _compute_auc(scores, labels):
    """Compute AUC from scores and labels."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = 0.0
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def plot_precision_recall_by_gold(m):
    """Per-gold-count P-R subplots with SFT+DPO."""
    bc = m['bc_retrieval']
    ppo = m['ppo_retrieval']
    sft_dpo = m.get('sft_dpo_retrieval')
    baselines = m['baselines_retrieval']

    gold_counts = sorted({int(k.split('_')[1]) for k in bc if k.startswith('gold_')})
    n_gc = len(gold_counts)
    if n_gc == 0:
        return

    fig, axes = plt.subplots(1, n_gc, figsize=(7 * n_gc, 6), squeeze=False)
    for col, gc in enumerate(sorted(gold_counts)):
        ax = axes[0, col]
        key = f"gold_{gc}"

        # Greedy curve
        gc_greedy_r, gc_greedy_p = [], []
        for b in baselines:
            if "Greedy" in b["strategy"]:
                sub = b.get(key)
                if sub:
                    gc_greedy_r.append(sub["recall"])
                    gc_greedy_p.append(sub["precision"])
        if gc_greedy_r:
            ax.plot(gc_greedy_r, gc_greedy_p, 's-', color='#2196F3',
                    label='Greedy(K)', markersize=5, linewidth=1.5)
            for b in baselines:
                if "Greedy" in b["strategy"]:
                    sub = b.get(key)
                    if sub:
                        k = b["strategy"].split('(')[1].rstrip(')')
                        ax.annotate(f'K={k}', (sub["recall"], sub["precision"]),
                                    textcoords="offset points", xytext=(5, 5),
                                    fontsize=7, color='#2196F3')

        # Random curve
        gc_rand_r, gc_rand_p = [], []
        for b in baselines:
            if "Random" in b["strategy"]:
                sub = b.get(key)
                if sub:
                    gc_rand_r.append(sub["recall"])
                    gc_rand_p.append(sub["precision"])
        if gc_rand_r:
            ax.plot(gc_rand_r, gc_rand_p, 'o-', color='#999999',
                    label='Random(K)', markersize=4, linewidth=1)

        # F1 iso-curves
        for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6]:
            r_range = np.linspace(0.01, 0.99, 200)
            p_iso = f1_val * r_range / (2 * r_range - f1_val)
            mask_iso = (p_iso > 0) & (p_iso <= 1)
            ax.plot(r_range[mask_iso], p_iso[mask_iso], '--', color='#E0E0E0',
                    linewidth=0.6, alpha=0.5)

        # BC, BC+PPO, SFT+DPO points
        bc_sub = bc.get(key)
        ppo_sub = ppo.get(key)
        sft_sub = sft_dpo.get(key) if sft_dpo else None

        if bc_sub:
            ax.scatter(bc_sub['recall'], bc_sub['precision'], marker='D',
                       s=100, color='#FF9800', zorder=5, edgecolors='black',
                       linewidths=0.8, label=f"BC (F1={bc_sub['f1']:.1%})")
        if ppo_sub:
            ax.scatter(ppo_sub['recall'], ppo_sub['precision'], marker='*',
                       s=200, color='#F44336', zorder=5, edgecolors='black',
                       linewidths=0.8, label=f"BC+PPO (F1={ppo_sub['f1']:.1%})")
        if sft_sub:
            ax.scatter(sft_sub['recall'], sft_sub['precision'], marker='^',
                       s=150, color='#9C27B0', zorder=5, edgecolors='black',
                       linewidths=0.8, label=f"SFT+DPO (F1={sft_sub['f1']:.1%})")
        if bc_sub and ppo_sub:
            ax.annotate('', xy=(ppo_sub['recall'], ppo_sub['precision']),
                        xytext=(bc_sub['recall'], bc_sub['precision']),
                        arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                        lw=1.5, ls='--'))

        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Gold={gc} Subset (n={ppo_sub.get("total", "?") if ppo_sub else "?"})',
                     fontsize=12)
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

    fig.suptitle('Per-Gold-Count Precision-Recall Tradeoff', fontsize=14, y=1.02)
    fig.tight_layout()
    path = f'{OUT_DIR}/precision_recall_by_gold.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── ROC Curve Plots ──────────────────────────────────────────────────

def plot_roc_curve(m):
    """Overall ROC curves for all methods."""
    roc_data = m.get('roc_data')
    if not roc_data:
        print('  (no roc_data in metrics, skipping ROC plot)')
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], '--', color='#E0E0E0', linewidth=0.8)  # reference diagonal

    styles = [
        ('random', '#999999', '--', 1.5, 'Random'),
        ('greedy', '#2196F3', '-',  2,   'Greedy BoW'),
        ('bc',     '#FF9800', '-',  2,   'BC'),
        ('ppo',    '#F44336', '-',  2.5, 'BC+PPO'),
        ('dpo',    '#9C27B0', '-',  2,   'SFT+DPO'),
    ]
    for key, color, ls, lw, label in styles:
        d = roc_data.get(key)
        if not d or 'scores' not in d:
            continue
        fpr, tpr = _compute_roc_curve(d['scores'], d['labels'])
        auc_val = d.get('auc', _compute_auc(d['scores'], d['labels']))
        ax.plot(fpr, tpr, color=color, ls=ls, lw=lw,
                label=f'{label} (AUC={auc_val:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Paragraph Retrieval', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    fig.tight_layout()
    path = f'{OUT_DIR}/roc_curve.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_roc_curve_by_gold(m):
    """Per-gold-count ROC subplots."""
    roc_data = m.get('roc_data')
    if not roc_data:
        print('  (no roc_data in metrics, skipping per-gold ROC plot)')
        return

    bc_data = roc_data.get('bc', {})
    gold_keys = sorted(k for k in bc_data if k.startswith('gold_'))
    if not gold_keys:
        return

    n_gc = len(gold_keys)
    fig, axes = plt.subplots(1, n_gc, figsize=(7 * n_gc, 6.5), squeeze=False)

    styles = [
        ('random', '#999999', '--', 1.5, 'Random'),
        ('greedy', '#2196F3', '-',  2,   'Greedy BoW'),
        ('bc',     '#FF9800', '-',  2,   'BC'),
        ('ppo',    '#F44336', '-',  2.5, 'BC+PPO'),
        ('dpo',    '#9C27B0', '-',  2,   'SFT+DPO'),
    ]

    for col, gc_key in enumerate(gold_keys):
        ax = axes[0, col]
        gc = gc_key.split('_')[1]
        ax.plot([0, 1], [0, 1], '--', color='#E0E0E0', linewidth=0.8)  # reference diagonal

        n_samples = 0
        for key, color, ls, lw, label in styles:
            model_data = roc_data.get(key, {})
            gc_data = model_data.get(gc_key)
            if not gc_data or 'scores' not in gc_data:
                continue
            fpr, tpr = _compute_roc_curve(gc_data['scores'], gc_data['labels'])
            auc_val = gc_data.get('auc', _compute_auc(gc_data['scores'], gc_data['labels']))
            ax.plot(fpr, tpr, color=color, ls=ls, lw=lw,
                    label=f'{label} (AUC={auc_val:.3f})')
            if key == 'bc':
                n_samples = len(gc_data['labels']) // 10

        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'Gold={gc} Subset (n≈{n_samples})', fontsize=12)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')

    fig.suptitle('Per-Gold-Count ROC Curves', fontsize=14, y=1.02)
    fig.tight_layout()
    path = f'{OUT_DIR}/roc_curve_by_gold.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


if __name__ == '__main__':
    print('Generating extra plots...')
    m = load()
    plot_f1_by_gold(m)
    plot_adaptive_reads(m)
    plot_precision_recall(m)
    plot_precision_recall_by_gold(m)
    plot_roc_curve(m)
    plot_roc_curve_by_gold(m)
    print('Done.')
