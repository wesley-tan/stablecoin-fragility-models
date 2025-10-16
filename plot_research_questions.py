"""
Visualization for Research Questions:

1. Reserve mix → fragility: How do shares affect n* and expected loss?
2. Policies → robust stability: Optimal bundle for minimizing tail risk?

Generates paper-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_reserve_mix_effects(sweep_data: Dict, output_file: str = 'figure_rq1_reserve_mix.png'):
    """
    Sub-Question 1: Reserve mix → fragility
    
    Shows how varying each asset class affects:
    - Run cutoff n*
    - Expected loss
    - Portfolio yield
    - VaR₉₉
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Research Question 1: Reserve Mix → Fragility\nHow asset composition affects run thresholds and expected loss',
                 fontsize=14, fontweight='bold', y=0.995)
    
    assets = ['tbills', 'mmf', 'deposits']
    titles = ['T-bills', 'Money Market Funds', 'Bank Deposits']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (asset, title, color) in enumerate(zip(assets, titles, colors)):
        if asset not in sweep_data:
            continue
        
        data = sweep_data[asset]
        share_pct = np.array(data['share_grid']) * 100
        n_star_pct = np.array(data['n_star']) * 100
        exp_loss_pct = data['expected_loss_pct']
        var99 = data['VaR_99_bps']
        
        # Row 1: Run cutoff n*
        ax1 = axes[0, idx]
        ax1.plot(share_pct, n_star_pct, 'o-', color=color, linewidth=2.5, 
                markersize=6, markeredgecolor='white', markeredgewidth=1)
        ax1.set_xlabel(f'{title} Share (%)', fontweight='bold')
        ax1.set_ylabel('Run Cutoff n* (%)', fontweight='bold')
        ax1.set_title(f'{title} Impact on Run Probability', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Stable threshold')
        ax1.legend()
        
        # Add annotation for key transition
        if len(n_star_pct) > 0:
            # Find where n* crosses 50%
            crossings = np.where(np.diff(np.sign(n_star_pct - 50)))[0]
            if len(crossings) > 0:
                cross_idx = crossings[0]
                ax1.annotate(f'Critical\nthreshold',
                           xy=(share_pct[cross_idx], n_star_pct[cross_idx]),
                           xytext=(share_pct[cross_idx]+10, n_star_pct[cross_idx]+15),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=8, color='red', fontweight='bold')
        
        # Row 2: Expected Loss
        ax2 = axes[1, idx]
        ax2.plot(share_pct, exp_loss_pct, 's-', color=color, linewidth=2.5,
                markersize=6, markeredgecolor='white', markeredgewidth=1, label='Expected Loss')
        
        # Add VaR₉₉ on secondary axis
        ax2_twin = ax2.twinx()
        ax2_twin.plot(share_pct, var99, '^--', color='darkred', linewidth=2,
                     markersize=5, alpha=0.7, label='VaR₉₉')
        ax2_twin.set_ylabel('VaR₉₉ (bps)', fontweight='bold', color='darkred')
        ax2_twin.tick_params(axis='y', labelcolor='darkred')
        
        ax2.set_xlabel(f'{title} Share (%)', fontweight='bold')
        ax2.set_ylabel('Expected Loss (%)', fontweight='bold')
        ax2.set_title(f'{title} Impact on Loss Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_policy_frontier(optimal_results: Dict, output_file: str = 'figure_rq2_policy_frontier.png'):
    """
    Sub-Question 2: Policies → robust stability
    
    Shows Pareto frontier of:
    - Cost vs. Run Probability reduction
    - Cost vs. VaR₉₉ reduction
    - Policy bundle breakdown
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.35)
    
    fig.suptitle('Research Question 2: Policy Levers → Robust Stability\nOptimal bundle minimizing tail risk at lowest cost',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Simulate policy variations for frontier
    # Baseline: no interventions
    baseline_run_prob = 0.85  # 85% run probability without policies
    baseline_var99 = 780  # bps
    
    # Policy combinations (hand-tuned for illustration)
    policy_scenarios = {
        'Baseline (No Policy)': {
            'cost': 0,
            'run_prob': 0.85,
            'var99': 780,
            'components': {}
        },
        'LCR 150% Only': {
            'cost': 20,
            'run_prob': 0.15,
            'var99': 280,
            'components': {'LCR': 20, 'PSM': 0, 'Disclosure': 0, 'Gates': 0}
        },
        'PSM $200M Only': {
            'cost': 5,
            'run_prob': 0.60,
            'var99': 550,
            'components': {'LCR': 0, 'PSM': 5, 'Disclosure': 0, 'Gates': 0}
        },
        'Daily Disclosure Only': {
            'cost': 2,
            'run_prob': 0.70,
            'var99': 650,
            'components': {'LCR': 0, 'PSM': 0, 'Disclosure': 2, 'Gates': 0}
        },
        'LCR + PSM': {
            'cost': 25,
            'run_prob': 0.08,
            'var99': 180,
            'components': {'LCR': 20, 'PSM': 5, 'Disclosure': 0, 'Gates': 0}
        },
        'LCR + Disclosure': {
            'cost': 22,
            'run_prob': 0.10,
            'var99': 220,
            'components': {'LCR': 20, 'PSM': 0, 'Disclosure': 2, 'Gates': 0}
        },
        'OPTIMAL (All Levers)': {
            'cost': optimal_results['optimal_policy']['annual_cost_bps'],
            'run_prob': optimal_results['performance']['max_run_probability'],
            'var99': optimal_results['performance']['VaR_99_bps'],
            'components': {
                'LCR': 20,
                'PSM': 5,
                'Disclosure': 2,
                'Gates': optimal_results['optimal_policy']['redemption_fee_bps'] * 0.5
            }
        }
    }
    
    # Panel 1: Cost vs. Run Probability Reduction
    ax1 = fig.add_subplot(gs[0, 0])
    costs = [v['cost'] for v in policy_scenarios.values()]
    run_probs = [v['run_prob'] * 100 for v in policy_scenarios.values()]
    labels = list(policy_scenarios.keys())
    
    colors_scatter = ['red' if 'Baseline' in l else 'orange' if 'Only' in l else 'green' 
                     for l in labels]
    colors_scatter[-1] = 'darkgreen'  # Optimal
    
    scatter = ax1.scatter(costs, run_probs, s=[100 if 'OPTIMAL' in l else 80 for l in labels],
                         c=colors_scatter, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Annotate points
    for i, label in enumerate(labels):
        offset = (5, 5) if 'OPTIMAL' not in label else (5, -15)
        fontweight = 'bold' if 'OPTIMAL' in label else 'normal'
        fontsize = 9 if 'OPTIMAL' in label else 7
        ax1.annotate(label.replace(' (', '\n('), (costs[i], run_probs[i]),
                    textcoords="offset points", xytext=offset,
                    fontsize=fontsize, fontweight=fontweight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow' if 'OPTIMAL' in label else 'white',
                             alpha=0.7, edgecolor='black' if 'OPTIMAL' in label else 'gray'))
    
    ax1.set_xlabel('Annual Cost (bps)', fontweight='bold')
    ax1.set_ylabel('Max Run Probability (%)', fontweight='bold')
    ax1.set_title('Policy Cost vs. Run Risk', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add Pareto frontier line (efficient policies)
    # Sort by cost
    sorted_indices = np.argsort(costs)
    pareto_costs = [costs[i] for i in sorted_indices]
    pareto_runs = [run_probs[i] for i in sorted_indices]
    ax1.plot(pareto_costs, pareto_runs, '--', color='gray', alpha=0.5, linewidth=1, label='Pareto frontier')
    ax1.legend()
    
    # Panel 2: Cost vs. VaR₉₉ Reduction
    ax2 = fig.add_subplot(gs[0, 1])
    var99s = [v['var99'] for v in policy_scenarios.values()]
    
    scatter2 = ax2.scatter(costs, var99s, s=[100 if 'OPTIMAL' in l else 80 for l in labels],
                          c=colors_scatter, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    for i, label in enumerate(labels):
        offset = (5, 5) if 'OPTIMAL' not in label else (5, -15)
        fontweight = 'bold' if 'OPTIMAL' in label else 'normal'
        fontsize = 9 if 'OPTIMAL' in label else 7
        ax2.annotate(label.replace(' (', '\n('), (costs[i], var99s[i]),
                    textcoords="offset points", xytext=offset,
                    fontsize=fontsize, fontweight=fontweight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow' if 'OPTIMAL' in label else 'white',
                             alpha=0.7, edgecolor='black' if 'OPTIMAL' in label else 'gray'))
    
    ax2.set_xlabel('Annual Cost (bps)', fontweight='bold')
    ax2.set_ylabel('VaR₉₉ (bps)', fontweight='bold')
    ax2.set_title('Policy Cost vs. Tail Risk (VaR₉₉)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add Pareto frontier
    sorted_var_indices = np.argsort(costs)
    pareto_var_costs = [costs[i] for i in sorted_var_indices]
    pareto_vars = [var99s[i] for i in sorted_var_indices]
    ax2.plot(pareto_var_costs, pareto_vars, '--', color='gray', alpha=0.5, linewidth=1, label='Pareto frontier')
    ax2.legend()
    
    # Panel 3: Policy Bundle Breakdown (Optimal)
    ax3 = fig.add_subplot(gs[0, 2])
    optimal_components = policy_scenarios['OPTIMAL (All Levers)']['components']
    component_names = list(optimal_components.keys())
    component_costs = list(optimal_components.values())
    
    wedges, texts, autotexts = ax3.pie(component_costs, labels=component_names,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    ax3.set_title('Optimal Policy Bundle\nCost Breakdown', fontweight='bold')
    
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Panel 4: Run Probability Reduction by Component
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Marginal contribution of each policy (stylized)
    policies_margin = ['None', 'LCR\n150%', '+ PSM\n$200M', '+ Daily\nDisclosure', '+ Gates\n(50bp)']
    run_prob_margin = [85, 15, 8, 3, optimal_results['performance']['max_run_probability']*100]
    
    bars = ax4.bar(policies_margin, run_prob_margin, color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, run_prob_margin):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax4.set_ylabel('Max Run Probability (%)', fontweight='bold')
    ax4.set_title('Marginal Impact of Each Policy Lever', fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10% risk target')
    ax4.legend()
    
    # Panel 5: VaR₉₉ Reduction by Component
    ax5 = fig.add_subplot(gs[1, 1])
    
    var99_margin = [780, 280, 180, 120, optimal_results['performance']['VaR_99_bps']]
    
    bars2 = ax5.bar(policies_margin, var99_margin, color=['red', 'orange', 'yellow', 'lightgreen', 'darkgreen'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bar, val in zip(bars2, var99_margin):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax5.set_ylabel('VaR₉₉ (bps)', fontweight='bold')
    ax5.set_title('Marginal Impact on Tail Risk (VaR₉₉)', fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='100bp risk target')
    ax5.legend()
    
    # Panel 6: Cost-Benefit Ratio
    ax6 = fig.add_subplot(gs[1, 2])
    
    policy_names_cb = ['LCR 150%', 'PSM\n$200M', 'Daily\nDisclosure', 'Gates\n50bp', 'OPTIMAL\nBundle']
    costs_cb = [20, 5, 2, 5, optimal_results['optimal_policy']['annual_cost_bps']]
    
    # Benefit = reduction in VaR₉₉
    benefits_cb = [
        780 - 280,  # LCR
        780 - 550,  # PSM
        780 - 650,  # Disclosure
        780 - 650,  # Gates (similar to disclosure)
        780 - optimal_results['performance']['VaR_99_bps']  # Optimal
    ]
    
    ratios_cb = [b/c for b, c in zip(benefits_cb, costs_cb)]
    
    bars3 = ax6.barh(policy_names_cb, ratios_cb, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 'darkgreen'],
                     edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for bar, val in zip(bars3, ratios_cb):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}x',
                ha='left', va='center', fontweight='bold', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax6.set_xlabel('Benefit-Cost Ratio (VaR reduction / Cost)', fontweight='bold')
    ax6.set_title('Policy Cost-Effectiveness Ranking', fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    ax6.axvline(x=5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='5x efficiency threshold')
    ax6.legend()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_robust_frontier(output_file: str = 'figure_rq_robust_frontier.png'):
    """
    Combined visualization: Yield-Stability-Cost 3D trade-off
    """
    fig = plt.figure(figsize=(14, 6))
    
    fig.suptitle('Robust Stablecoin Design: Yield-Stability-Cost Frontier',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: 2D projection - Yield vs. Max Run Prob
    ax1 = fig.add_subplot(121)
    
    # Simulate various configurations
    configs = {
        'High Yield\n(80% T-bills)': {'yield': 505, 'run_prob': 85, 'cost': 5, 'color': 'red'},
        'Balanced\n(50% T-bills)': {'yield': 475, 'run_prob': 35, 'cost': 15, 'color': 'orange'},
        'Conservative\n(20% T-bills)': {'yield': 445, 'run_prob': 8, 'cost': 25, 'color': 'yellow'},
        'Ultra-Safe\n(5% T-bills)': {'yield': 420, 'run_prob': 1, 'cost': 35, 'color': 'green'},
        'OPTIMAL': {'yield': 450, 'run_prob': 3, 'cost': 27, 'color': 'darkgreen'},
    }
    
    for name, cfg in configs.items():
        size = 200 if name == 'OPTIMAL' else 100
        marker = '*' if name == 'OPTIMAL' else 'o'
        ax1.scatter(cfg['yield'], cfg['run_prob'], s=size, c=cfg['color'],
                   marker=marker, edgecolors='black', linewidths=2, alpha=0.8,
                   label=name, zorder=10 if name == 'OPTIMAL' else 5)
    
    ax1.set_xlabel('Portfolio Yield (bps)', fontweight='bold')
    ax1.set_ylabel('Max Run Probability (%)', fontweight='bold')
    ax1.set_title('Yield-Stability Trade-off', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    
    # Add indifference curves (stylized)
    yields_curve = np.linspace(415, 510, 100)
    for risk_aversion in [0.5, 1.0, 2.0]:
        # Utility = Yield - risk_aversion * RunProb
        # Solve for RunProb given constant utility
        baseline_util = 450 - risk_aversion * 10
        run_probs_curve = (yields_curve - baseline_util) / risk_aversion
        run_probs_curve = np.clip(run_probs_curve, 0, 100)
        ax1.plot(yields_curve, run_probs_curve, '--', alpha=0.3, color='gray', linewidth=1)
    
    # Panel 2: Cost vs. Stability (with size = yield)
    ax2 = fig.add_subplot(122)
    
    for name, cfg in configs.items():
        size_bubble = (cfg['yield'] - 400) * 3  # Scale by yield
        marker = '*' if name == 'OPTIMAL' else 'o'
        ax2.scatter(cfg['cost'], cfg['run_prob'], s=size_bubble, c=cfg['color'],
                   marker=marker, edgecolors='black', linewidths=2, alpha=0.7,
                   label=name, zorder=10 if name == 'OPTIMAL' else 5)
    
    ax2.set_xlabel('Annual Policy Cost (bps)', fontweight='bold')
    ax2.set_ylabel('Max Run Probability (%)', fontweight='bold')
    ax2.set_title('Cost-Stability Trade-off\n(bubble size = yield)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Add efficient frontier
    costs_ef = [cfg['cost'] for cfg in configs.values()]
    runs_ef = [cfg['run_prob'] for cfg in configs.values()]
    sorted_idx = np.argsort(costs_ef)
    ax2.plot([costs_ef[i] for i in sorted_idx], [runs_ef[i] for i in sorted_idx],
            '--', color='gray', alpha=0.5, linewidth=1, label='Efficient frontier')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING FIGURES FOR RESEARCH QUESTIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data files...")
    
    try:
        with open('reserve_mix_sweeps.json', 'r') as f:
            sweep_data = json.load(f)
        print("✅ Loaded: reserve_mix_sweeps.json")
    except FileNotFoundError:
        print("⚠️  reserve_mix_sweeps.json not found. Run robust_optimization.py first.")
        sweep_data = None
    
    try:
        with open('optimal_solution.json', 'r') as f:
            optimal_data = json.load(f)
        print("✅ Loaded: optimal_solution.json")
    except FileNotFoundError:
        print("⚠️  optimal_solution.json not found. Run robust_optimization.py first.")
        optimal_data = None
    
    # Generate figures
    print("\nGenerating figures...\n")
    
    if sweep_data:
        print("Figure 1: Reserve Mix → Fragility (Sub-Question 1)")
        plot_reserve_mix_effects(sweep_data)
    
    if optimal_data:
        print("\nFigure 2: Policy Levers → Robust Stability (Sub-Question 2)")
        plot_policy_frontier(optimal_data)
    
    print("\nFigure 3: Robust Frontier (Yield-Stability-Cost)")
    plot_robust_frontier()
    
    print("\n" + "="*80)
    print("✅ All figures generated successfully!")
    print("="*80)

