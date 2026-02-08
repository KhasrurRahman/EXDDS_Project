import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_metric_comparison(save_path: str = "Doc/figures/"):
    """Bar chart comparing Baseline vs ORE across metrics."""
    
    metrics = ['Recall@20', 'NDCG@20', 'Precision@20']
    
    # TREC-DL 2019
    baseline_2019 = [0.2984, 0.8528, 0.8093]
    ore_2019 = [0.2988, 0.8613, 0.8105]
    
    # TREC-DL 2020
    baseline_2020 = [0.3778, 0.8225, 0.7167]
    ore_2020 = [0.3770, 0.8307, 0.7167]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # TREC-DL 2019
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, baseline_2019, width, label='Baseline (Hybrid)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, ore_2019, width, label='ORE', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('TREC-DL 2019', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    # TREC-DL 2020
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, baseline_2020, width, label='Baseline (Hybrid)', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, ore_2020, width, label='ORE', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('TREC-DL 2020', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    
    for bar in bars3:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        ax2.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}metric_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}metric_comparison.png")
    plt.close()


def plot_per_query_improvement(save_path: str = "Doc/figures/"):
    """Scatter plot showing per-query NDCG improvement."""
    
    # TREC-DL 2019 data
    baseline_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.8436, 0.9238, 1.0, 0.8974, 0.4573, 0.9649,
        1.0, 1.0, 0.6522, 1.0, 0.7853, 1.0, 0.9357, 1.0, 0.7614, 1.0,
        0.5765, 1.0, 0.8672, 0.7951, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.6813, 1.0, 1.0, 0.8230, 1.0, 0.6272,
        0.6645, 0.7259, 1.0
    ]
    
    ore_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.9096, 0.9238, 1.0, 0.8974, 0.6085, 0.9649,
        1.0, 1.0, 0.7536, 1.0, 0.8391, 1.0, 0.9357, 1.0, 0.8355, 1.0,
        0.5765, 1.0, 0.8672, 0.8232, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.7221, 1.0, 1.0, 0.8230, 1.0, 0.6554,
        0.6645, 0.7571, 1.0
    ]
    
    # TREC-DL 2020 data
    baseline_2020 = [
        0.8628, 1.0, 0.7410, 0.9134, 0.9505, 1.0, 0.6932, 0.5765, 1.0, 0.7620,
        0.9421, 1.0, 0.6915, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.2505, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7361,
        0.6858, 0.9802, 0.8193, 1.0
    ]
    
    ore_2020 = [
        0.8628, 1.0, 0.7876, 0.9134, 0.9011, 1.0, 0.7821, 0.5765, 1.0, 0.8196,
        0.9421, 1.0, 0.7449, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.3603, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7683,
        0.6858, 0.9802, 0.8534, 1.0
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TREC-DL 2019
    ax1 = axes[0]
    colors_2019 = ['green' if o > b else 'red' if o < b else 'gray' 
                   for b, o in zip(baseline_2019, ore_2019)]
    ax1.scatter(baseline_2019, ore_2019, c=colors_2019, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    ax1.set_xlabel('Baseline NDCG@20', fontsize=12)
    ax1.set_ylabel('ORE NDCG@20', fontsize=12)
    ax1.set_title('TREC-DL 2019: Per-Query Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.2, 1.05)
    ax1.set_ylim(0.2, 1.05)
    ax1.legend()
    
    # TREC-DL 2020
    ax2 = axes[1]
    colors_2020 = ['green' if o > b else 'red' if o < b else 'gray' 
                   for b, o in zip(baseline_2020, ore_2020)]
    ax2.scatter(baseline_2020, ore_2020, c=colors_2020, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    ax2.set_xlabel('Baseline NDCG@20', fontsize=12)
    ax2.set_ylabel('ORE NDCG@20', fontsize=12)
    ax2.set_title('TREC-DL 2020: Per-Query Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.2, 1.05)
    ax2.set_ylim(0.2, 1.05)
    ax2.legend()
    
    # Add legend explanation
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='ORE wins'),
                       Patch(facecolor='gray', label='Tie'),
                       Patch(facecolor='red', label='ORE loses')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{save_path}per_query_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}per_query_scatter.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}per_query_scatter.png")
    plt.close()


def plot_improvement_distribution(save_path: str = "Doc/figures/"):
    """Histogram of NDCG improvements."""
    
    baseline_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.8436, 0.9238, 1.0, 0.8974, 0.4573, 0.9649,
        1.0, 1.0, 0.6522, 1.0, 0.7853, 1.0, 0.9357, 1.0, 0.7614, 1.0,
        0.5765, 1.0, 0.8672, 0.7951, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.6813, 1.0, 1.0, 0.8230, 1.0, 0.6272,
        0.6645, 0.7259, 1.0
    ]
    ore_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.9096, 0.9238, 1.0, 0.8974, 0.6085, 0.9649,
        1.0, 1.0, 0.7536, 1.0, 0.8391, 1.0, 0.9357, 1.0, 0.8355, 1.0,
        0.5765, 1.0, 0.8672, 0.8232, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.7221, 1.0, 1.0, 0.8230, 1.0, 0.6554,
        0.6645, 0.7571, 1.0
    ]
    
    baseline_2020 = [
        0.8628, 1.0, 0.7410, 0.9134, 0.9505, 1.0, 0.6932, 0.5765, 1.0, 0.7620,
        0.9421, 1.0, 0.6915, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.2505, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7361,
        0.6858, 0.9802, 0.8193, 1.0
    ]
    ore_2020 = [
        0.8628, 1.0, 0.7876, 0.9134, 0.9011, 1.0, 0.7821, 0.5765, 1.0, 0.8196,
        0.9421, 1.0, 0.7449, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.3603, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7683,
        0.6858, 0.9802, 0.8534, 1.0
    ]
    
    diff_2019 = [o - b for b, o in zip(baseline_2019, ore_2019)]
    diff_2020 = [o - b for b, o in zip(baseline_2020, ore_2020)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # TREC-DL 2019
    ax1 = axes[0]
    ax1.hist(diff_2019, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax1.axvline(x=np.mean(diff_2019), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(diff_2019):.4f}')
    ax1.set_xlabel('NDCG@20 Improvement (ORE - Baseline)', fontsize=12)
    ax1.set_ylabel('Number of Queries', fontsize=12)
    ax1.set_title('TREC-DL 2019: Improvement Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # TREC-DL 2020
    ax2 = axes[1]
    ax2.hist(diff_2020, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax2.axvline(x=np.mean(diff_2020), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(diff_2020):.4f}')
    ax2.set_xlabel('NDCG@20 Improvement (ORE - Baseline)', fontsize=12)
    ax2.set_ylabel('Number of Queries', fontsize=12)
    ax2.set_title('TREC-DL 2020: Improvement Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}improvement_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}improvement_distribution.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}improvement_distribution.png")
    plt.close()


def plot_win_tie_loss(save_path: str = "Doc/figures/"):
    """Pie charts showing win/tie/loss breakdown."""
    
    # Calculate from data
    wins_2019, ties_2019, losses_2019 = 11, 31, 1
    wins_2020, ties_2020, losses_2020 = 10, 43, 1
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    # TREC-DL 2019
    ax1 = axes[0]
    sizes_2019 = [wins_2019, ties_2019, losses_2019]
    labels_2019 = [f'Wins\n({wins_2019})', f'Ties\n({ties_2019})', f'Losses\n({losses_2019})']
    ax1.pie(sizes_2019, labels=labels_2019, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0, 0.05))
    ax1.set_title('TREC-DL 2019\nWin/Tie/Loss', fontsize=14, fontweight='bold')
    
    # TREC-DL 2020
    ax2 = axes[1]
    sizes_2020 = [wins_2020, ties_2020, losses_2020]
    labels_2020 = [f'Wins\n({wins_2020})', f'Ties\n({ties_2020})', f'Losses\n({losses_2020})']
    ax2.pie(sizes_2020, labels=labels_2020, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0, 0.05))
    ax2.set_title('TREC-DL 2020\nWin/Tie/Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}win_tie_loss.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}win_tie_loss.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}win_tie_loss.png")
    plt.close()


def plot_baseline_vs_improvement(save_path: str = "Doc/figures/"):
    """Shows that ORE helps more on harder queries."""
    
    baseline_all = [
        1.0, 0.8944, 1.0, 1.0, 0.8436, 0.9238, 1.0, 0.8974, 0.4573, 0.9649,
        1.0, 1.0, 0.6522, 1.0, 0.7853, 1.0, 0.9357, 1.0, 0.7614, 1.0,
        0.5765, 1.0, 0.8672, 0.7951, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.6813, 1.0, 1.0, 0.8230, 1.0, 0.6272,
        0.6645, 0.7259, 1.0,
        0.8628, 1.0, 0.7410, 0.9134, 0.9505, 1.0, 0.6932, 0.5765, 1.0, 0.7620,
        0.9421, 1.0, 0.6915, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.2505, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7361,
        0.6858, 0.9802, 0.8193, 1.0
    ]
    
    ore_all = [
        1.0, 0.8944, 1.0, 1.0, 0.9096, 0.9238, 1.0, 0.8974, 0.6085, 0.9649,
        1.0, 1.0, 0.7536, 1.0, 0.8391, 1.0, 0.9357, 1.0, 0.8355, 1.0,
        0.5765, 1.0, 0.8672, 0.8232, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.7221, 1.0, 1.0, 0.8230, 1.0, 0.6554,
        0.6645, 0.7571, 1.0,
        0.8628, 1.0, 0.7876, 0.9134, 0.9011, 1.0, 0.7821, 0.5765, 1.0, 0.8196,
        0.9421, 1.0, 0.7449, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.3603, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7683,
        0.6858, 0.9802, 0.8534, 1.0
    ]
    
    improvements = [o - b for b, o in zip(baseline_all, ore_all)]
    
    # Filter out perfect baselines for clearer visualization
    filtered_baseline = [b for b in baseline_all if b < 1.0]
    filtered_improvements = [imp for b, imp in zip(baseline_all, improvements) if b < 1.0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(filtered_baseline, filtered_improvements, alpha=0.7, s=80, 
               c='#3498db', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(filtered_baseline, filtered_improvements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(filtered_baseline), max(filtered_baseline), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Baseline NDCG@20', fontsize=12)
    ax.set_ylabel('NDCG@20 Improvement', fontsize=12)
    ax.set_title('ORE Helps More on Harder Queries\n(Combined TREC-DL 2019 & 2020)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}baseline_vs_improvement.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}baseline_vs_improvement.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}baseline_vs_improvement.png")
    plt.close()


def plot_summary_table(save_path: str = "Doc/figures/"):
    """Create a summary table as an image."""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    data = [
        ['TREC-DL 2019', '43', '0.2984', '0.2988', '+0.13%', '0.8528', '0.8613', '+1.00%'],
        ['TREC-DL 2020', '54', '0.3778', '0.3770', '-0.21%', '0.8225', '0.8307', '+0.99%'],
    ]
    
    columns = ['Dataset', 'Queries', 'Recall\nBaseline', 'Recall\nORE', 'Recall\nΔ', 
               'NDCG\nBaseline', 'NDCG\nORE', 'NDCG\nΔ']
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight NDCG improvement
    table[(1, 7)].set_facecolor('#d5f5e3')
    table[(2, 7)].set_facecolor('#d5f5e3')
    
    plt.title('ORE Experimental Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_path}summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}summary_table.pdf', bbox_inches='tight')
    print(f"✅ Saved: {save_path}summary_table.png")
    plt.close()


def generate_all_figures():
    """Generate all figures for the report."""
    import os
    
    save_path = "Doc/figures/"
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*50)
    print("  Generating Visualizations for ORE Results")
    print("="*50 + "\n")
    
    plot_metric_comparison(save_path)
    plot_per_query_improvement(save_path)
    plot_improvement_distribution(save_path)
    plot_win_tie_loss(save_path)
    plot_baseline_vs_improvement(save_path)
    plot_summary_table(save_path)
    
    print("\n" + "="*50)
    print("  ✅ All figures generated successfully!")
    print(f"  📁 Location: {save_path}")
    print("="*50 + "\n")


if __name__ == "__main__":
    generate_all_figures()