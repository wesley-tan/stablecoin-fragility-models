"""
CORRECTED Plots: Run Cutoff with PSM Overlay
============================================

Creates corrected figures with:
1. Right direction (more T-bills → safer)
2. PSM/backstop overlay
3. κ sensitivity grid
"""

import numpy as np
import matplotlib.pyplot as plt
from run_cutoff_corrected import grid_run_cutoff_corrected, psm_overlay


def plot_run_cutoff_with_psm(pi: float = 0.10, kappa: float = 0.08,
                             save_path: str = 'figure_corrected_psm.png'):
    """
    Plot run cutoff with PSM overlay.
    
    Shows:
    - Baseline (no PSM)
    - PSM 10% buffer
    - PSM 20% buffer
    """
    # Baseline
    result_base = grid_run_cutoff_corrected(pi, kappa)
    
    # PSM overlays
    psm_results = psm_overlay(pi, kappa, psm_sizes=[0.0, 0.10, 0.20])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot baseline
    tbill_pct = result_base['tbill_share_grid'] * 100
    n_pct = np.array(result_base['n_star']) * 100
    
    ax.plot(tbill_pct, n_pct, 'b-', lw=2.5, label='Baseline (no PSM)', zorder=3)
    
    # Plot PSM overlays
    colors = ['g', 'orange']
    psm_labels = ['PSM_10pct', 'PSM_20pct']
    psm_names = ['10% PSM buffer', '20% PSM buffer']
    
    for i, (label, name) in enumerate(zip(psm_labels, psm_names)):
        psm_data = psm_results[label]
        lam_grid = psm_data['lam_grid']
        tbill_grid = (1 - lam_grid) * 100
        n_stars = np.array(psm_data['n_star']) * 100
        
        ax.plot(tbill_grid, n_stars, color=colors[i], ls='--', lw=2, 
               label=name, zorder=2)
    
    # Mark thresholds
    thresh_base = (1 - result_base['lam_cutoff']) * 100
    ax.axvline(thresh_base, color='red', ls=':', lw=1.5, alpha=0.7,
              label=f'Threshold: {thresh_base:.0f}% T-bills')
    
    # Shade vulnerable region
    vuln_mask = n_pct > 1.0
    if np.any(vuln_mask):
        ax.fill_between(tbill_pct, 0, n_pct, where=vuln_mask, 
                       alpha=0.2, color='red', label='Vulnerable (run)')
    
    # Labels
    ax.set_xlabel('T-bill Share (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Run Cutoff vs Reserve Mix (CORRECTED)\n' +
                f'π={pi*100:.0f}%, κ={kappa*100:.0f}% | More T-bills → SAFER',
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(40, 100)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add insight box
    insight = (
        "CORRECTED: Higher T-bill share → less forced selling → safer.\n"
        "PSM buffer shifts threshold left (lower liquid requirement)."
    )
    ax.text(0.98, 0.02, insight, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_kappa_sensitivity(pi: float = 0.10,
                          save_path: str = 'figure_corrected_kappa.png'):
    """
    Plot sensitivity to κ (market impact).
    """
    kappa_vals = [0.02, 0.05, 0.08, 0.10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(kappa_vals)))
    
    for i, kappa in enumerate(kappa_vals):
        result = grid_run_cutoff_corrected(pi, kappa)
        
        tbill_pct = result['tbill_share_grid'] * 100
        n_pct = np.array(result['n_star']) * 100
        
        ax.plot(tbill_pct, n_pct, color=colors[i], lw=2,
               label=f'κ={kappa*100:.0f}% (threshold: {(1-result["lam_cutoff"])*100:.0f}%)')
    
    ax.set_xlabel('T-bill Share (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sensitivity to Market Impact κ\n' +
                f'π={pi*100:.0f}% | Deeper markets → lower κ → lower threshold',
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(40, 100)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig, ax


def plot_pi_sensitivity(kappa: float = 0.08,
                       save_path: str = 'figure_corrected_pi.png'):
    """
    Plot sensitivity to π (stress level).
    """
    pi_vals = [0.05, 0.08, 0.10, 0.15]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pi_vals)))
    
    for i, pi in enumerate(pi_vals):
        result = grid_run_cutoff_corrected(pi, kappa)
        
        tbill_pct = result['tbill_share_grid'] * 100
        n_pct = np.array(result['n_star']) * 100
        
        ax.plot(tbill_pct, n_pct, color=colors[i], lw=2,
               label=f'π={pi*100:.0f}% (threshold: {(1-result["lam_cutoff"])*100:.0f}%)')
    
    ax.set_xlabel('T-bill Share (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Sensitivity to Redemption Stress π\n' +
                f'κ={kappa*100:.0f}% | Higher stress → threshold shifts right',
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(40, 100)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING CORRECTED FIGURES")
    print("=" * 70)
    
    print("\n1. Main figure with PSM overlay...")
    plot_run_cutoff_with_psm(pi=0.10, kappa=0.08)
    
    print("\n2. κ sensitivity...")
    plot_kappa_sensitivity(pi=0.10)
    
    print("\n3. π sensitivity...")
    plot_pi_sensitivity(kappa=0.08)
    
    print("\n" + "=" * 70)
    print("✅ All corrected figures generated!")
    print("=" * 70)


