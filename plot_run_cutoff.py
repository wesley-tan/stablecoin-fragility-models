"""
Plot Run Cutoff vs Reserve Mix
================================

Clean implementation of two-tier model with plotting.
Generates Figure 1 for paper: "Run Cutoff vs T-bill Share"
"""

import numpy as np
import matplotlib.pyplot as plt
import json


def p_of_R(R, lam, kappa):
    """Fire-sale price"""
    if R <= lam: 
        return 1.0
    return 1.0 - kappa * (R - lam) / max(1e-12, (1 - lam))


def f_of_n(n, lam, pi, kappa):
    """
    Patient indifference condition: f(n*) = 0
    
    f(n) = λ + p(R(n))·(R(n) - λ) - R(n)
    """
    R = pi + n * (1 - pi)
    p = p_of_R(R, lam, kappa)
    return lam + p * (R - lam) - R


def solve_n_star(lam, pi, kappa, tol=1e-10):
    """Solve for equilibrium run fraction n*"""
    # If no fire sale needed
    if pi <= lam: 
        return 0.0
    
    lo, hi = 0.0, 1.0
    flo = f_of_n(lo, lam, pi, kappa)
    fhi = f_of_n(hi, lam, pi, kappa)
    
    # If f(1) > 0, no positive root -> no profitable run
    if fhi > 0: 
        return 0.0
    
    # Bisection
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = f_of_n(mid, lam, pi, kappa)
        if abs(fm) < tol: 
            return mid
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    
    return mid


def grid_run_cutoff(pi=0.08, kappa=0.10, lam_grid=None):
    """Compute run cutoff curve over λ grid"""
    if lam_grid is None:
        lam_grid = np.linspace(0.00, 0.60, 121)
    
    nstars = np.array([solve_n_star(lam, pi, kappa) for lam in lam_grid])
    
    # Cutoff = smallest lam with n*≈0
    zero_idx = np.where(nstars <= 1e-8)[0]
    lam_hat = lam_grid[zero_idx[0]] if zero_idx.size > 0 else lam_grid[-1]
    
    return lam_grid, 1 - lam_grid, nstars, lam_hat


def plot_paper_figure(pi=0.115, kappa=0.10, 
                     save_path='figure1_run_cutoff.png',
                     show=True):
    """
    Generate paper-ready figure: Run cutoff vs reserve mix.
    
    Args:
        pi: Impatient fraction (11.5% = USDC/SVB peak)
        kappa: Market impact coefficient (10% = moderate)
        save_path: Where to save figure
        show: Whether to display figure
    """
    # Compute curve
    lam_grid, tbill_share, nstars, lam_hat = grid_run_cutoff(pi, kappa)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main plot: n* vs T-bill share
    tbill_pct = tbill_share * 100
    n_pct = nstars * 100
    
    # Color by vulnerability
    vulnerable = nstars > 1e-4
    
    # Plot stable region (n*=0)
    stable_mask = ~vulnerable
    if np.any(stable_mask):
        ax.plot(tbill_pct[stable_mask], n_pct[stable_mask], 
               'g-', lw=3, label='Stable (n*=0)', zorder=3)
    
    # Plot vulnerable region (n*>0)
    if np.any(vulnerable):
        ax.plot(tbill_pct[vulnerable], n_pct[vulnerable], 
               'r-', lw=3, label='Vulnerable (run)', zorder=3)
    
    # Mark the cutoff
    tbill_hat = (1 - lam_hat) * 100
    ax.axvline(tbill_hat, color='orange', ls='--', lw=2, 
              label=f'Run cutoff: {tbill_hat:.1f}% T-bills', zorder=2)
    
    # Shade vulnerable region
    if np.any(vulnerable):
        ax.fill_between(tbill_pct, 0, n_pct, 
                       where=vulnerable, alpha=0.2, color='red', 
                       label='Vulnerable region')
    
    # Mark key reference points
    # USDC pre-SVB: ~80% T-bills, ~20% cash/MMF+deposits
    ax.axvline(80, color='blue', ls=':', lw=1.5, alpha=0.7,
              label='USDC pre-SVB (~80% T-bills)')
    
    # Annotations
    ax.text(tbill_hat + 2, 50, f'λ̂ = {lam_hat*100:.0f}% liquid\n({tbill_hat:.0f}% T-bills)',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Labels and formatting
    ax.set_xlabel('T-bill Share (%, Tier-2)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Run Equilibrium vs Reserve Composition\n' + 
                f'(π={pi*100:.1f}% impatient, κ={kappa*100:.0f}% impact)',
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(40, 100)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, ls='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add text box with key insight
    insight = (
        "Key Finding: Sharp threshold at ~89% T-bills.\n"
        "Reserve mix shifts from 10%→15% liquid eliminates run risk entirely.\n"
        "Mechanism: Higher T-bill share → fire-sale cascade → coordination on runs."
    )
    ax.text(0.98, 0.02, insight, 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_dual_axis_version(pi=0.115, kappa=0.10, 
                          save_path='figure1_dual_axis.png',
                          show=True):
    """
    Alternative version with dual y-axis (as in user's code).
    """
    lam_grid, tbill_share, nstars, lam_hat = grid_run_cutoff(pi, kappa)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Left axis: Remaining patients after run
    remaining_pct = (1 - nstars) * (1 - pi) * 100
    ax.plot(tbill_share * 100, remaining_pct, 'b-', lw=2.5, label='Remaining patients (%)')
    ax.set_xlabel('T-bill Share (%, Tier-2)', fontsize=12)
    ax.set_ylabel('Remaining Patient Investors (%)', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Right axis: n* (run fraction)
    ax2 = ax.twinx()
    ax2.plot(tbill_share * 100, nstars * 100, 'r--', lw=2, label='n* (run fraction %)')
    ax2.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Cutoff line
    tbill_hat = (1 - lam_hat) * 100
    ax.axvline(tbill_hat, color='orange', ls='--', lw=2, alpha=0.7,
              label=f'Cutoff: {tbill_hat:.1f}%')
    
    ax.set_title(f'Run Cutoff vs Reserve Mix (π={pi*100:.1f}%, κ={kappa*100:.0f}%)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, (ax, ax2)


def sensitivity_comparison(pi_vals=[0.05, 0.115, 0.20], 
                          kappa=0.10,
                          save_path='figure2_sensitivity.png',
                          show=True):
    """
    Compare run cutoffs under different stress scenarios (varying π).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green', 'blue', 'red']
    styles = ['-', '--', ':']
    
    for i, pi in enumerate(pi_vals):
        lam_grid, tbill_share, nstars, lam_hat = grid_run_cutoff(pi, kappa)
        
        label = f'π={pi*100:.1f}% (λ̂={lam_hat*100:.0f}%)'
        ax.plot(tbill_share * 100, nstars * 100, 
               color=colors[i], ls=styles[i], lw=2.5, label=label)
    
    ax.set_xlabel('T-bill Share (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Equilibrium Run Fraction n* (%)', fontsize=12, fontweight='bold')
    ax.set_title('Run Cutoff Sensitivity to Redemption Stress (κ=10%)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(40, 100)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def generate_all_figures():
    """Generate all paper figures"""
    print("=" * 70)
    print("GENERATING PAPER FIGURES")
    print("=" * 70)
    
    # Main figure (USDC/SVB calibration)
    print("\n📊 Figure 1: Run Cutoff vs Reserve Mix (main)")
    plot_paper_figure(pi=0.115, kappa=0.10, 
                     save_path='figure1_run_cutoff.png',
                     show=False)
    
    # Dual-axis version
    print("\n📊 Figure 1 (alternative): Dual-axis version")
    plot_dual_axis_version(pi=0.115, kappa=0.10,
                          save_path='figure1_dual_axis.png',
                          show=False)
    
    # Sensitivity to stress scenarios
    print("\n📊 Figure 2: Sensitivity to redemption stress")
    sensitivity_comparison(pi_vals=[0.05, 0.115, 0.20],
                         kappa=0.10,
                         save_path='figure2_sensitivity.png',
                         show=False)
    
    # Export data for external plotting (e.g., in R or Stata)
    print("\n💾 Exporting data for external tools...")
    lam_grid, tbill_share, nstars, lam_hat = grid_run_cutoff(pi=0.115, kappa=0.10)
    
    data_export = {
        'lambda_tier1': lam_grid.tolist(),
        'tbill_share_pct': (tbill_share * 100).tolist(),
        'n_star_pct': (nstars * 100).tolist(),
        'lambda_cutoff': lam_hat,
        'parameters': {'pi': 0.115, 'kappa': 0.10}
    }
    
    with open('figure_data.json', 'w') as f:
        json.dump(data_export, f, indent=2)
    
    print("✓ Saved plot data to: figure_data.json")
    
    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED")
    print("=" * 70)
    print("\nFiles created:")
    print("  - figure1_run_cutoff.png (main figure)")
    print("  - figure1_dual_axis.png (alternative)")
    print("  - figure2_sensitivity.png (robustness)")
    print("  - figure_data.json (data export)")


if __name__ == "__main__":
    # Generate all figures for paper
    generate_all_figures()
    
    # Or run individual plots:
    # plot_paper_figure(pi=0.115, kappa=0.10, show=True)


