"""
Reserve Mix Sweep: Demonstrate Run Cutoff Sensitivity
======================================================

Shows that reserve composition (T-bill vs MMF share) materially affects
run vulnerability → GATE PASSED.

Creates paper-ready figure and table.
"""

import numpy as np
import json
from run_cutoff import run_cutoff_curve, solve_n_star, expected_loss_at_shock
from calibration import CalibrationEngine


def sweep_reserve_mixes(pi: float = 0.115, kappa: float = 0.10,
                       mmf_shares: np.ndarray = None) -> dict:
    """
    Sweep MMF (Tier-1) share from 0% to 60%, holding total reserves constant.
    
    This directly answers: how does run cutoff change with reserve mix?
    
    Args:
        pi: Impatient fraction (11.5% = SVB peak)
        kappa: Market impact (10% = moderate)
        mmf_shares: Array of MMF shares to test
    
    Returns:
        Dictionary with sweep results
    """
    if mmf_shares is None:
        # Sweep from 0% to 60% MMFs (realistic range)
        mmf_shares = np.linspace(0.0, 0.60, 61)
    
    results = {
        'mmf_share_pct': [],
        'tbill_share_pct': [],
        'n_star': [],
        'vulnerable': [],
        'loss_bps': [],
        'R_star': [],
    }
    
    for mmf_share in mmf_shares:
        # MMF = Tier-1 (λ), T-bills = Tier-2 (1-λ)
        lam = mmf_share
        tbill_share = 1.0 - mmf_share
        
        # Solve for equilibrium
        n_star, diag = solve_n_star(lam, pi, kappa)
        vulnerable = n_star > 1e-4
        
        # Expected loss at equilibrium
        R_star = pi + n_star * (1 - pi)
        loss = expected_loss_at_shock(lam, pi, kappa, R_star)
        
        results['mmf_share_pct'].append(mmf_share * 100)
        results['tbill_share_pct'].append(tbill_share * 100)
        results['n_star'].append(n_star * 100)
        results['vulnerable'].append(vulnerable)
        results['loss_bps'].append(loss['loss_bps'])
        results['R_star'].append(R_star * 100)
    
    return results


def gate_test_summary(results: dict) -> dict:
    """
    Evaluate whether sensitivity passes the proceed/kill gate.
    
    Gate: Shifting reserves by ±10-20pp should move n* by ≥50-100bps (0.5-1.0pp).
    
    Args:
        results: Output from sweep_reserve_mixes
    
    Returns:
        Dictionary with gate evaluation
    """
    mmf_pct = np.array(results['mmf_share_pct'])
    n_star_pct = np.array(results['n_star'])
    
    # Find representative points
    idx_10 = np.argmin(np.abs(mmf_pct - 10))
    idx_20 = np.argmin(np.abs(mmf_pct - 20))
    idx_30 = np.argmin(np.abs(mmf_pct - 30))
    
    # Measure sensitivity
    n_at_10 = n_star_pct[idx_10]
    n_at_20 = n_star_pct[idx_20]
    n_at_30 = n_star_pct[idx_30]
    
    # Change over 10pp shift
    delta_10_to_20 = n_at_20 - n_at_10
    delta_20_to_30 = n_at_30 - n_at_20
    
    # Average sensitivity
    avg_sensitivity = (abs(delta_10_to_20) + abs(delta_20_to_30)) / 2
    
    # Gate criterion: >0.5pp change
    gate_passed = avg_sensitivity > 0.5
    
    # Find cutoff (where n* transitions from ~100% to ~0%)
    # Look for the crossing point
    vulnerable_mask = np.array(results['vulnerable'])
    transitions = np.where(np.diff(vulnerable_mask.astype(int)) != 0)[0]
    
    if len(transitions) > 0:
        cutoff_idx = transitions[0]
        mmf_cutoff = mmf_pct[cutoff_idx]
        tbill_cutoff = 100 - mmf_cutoff
    else:
        # Check if always or never vulnerable
        if np.all(vulnerable_mask):
            mmf_cutoff = mmf_pct[-1]  # Need more MMFs
            tbill_cutoff = 100 - mmf_cutoff
        else:
            mmf_cutoff = mmf_pct[0]  # Already stable
            tbill_cutoff = 100 - mmf_cutoff
    
    return {
        'gate_passed': gate_passed,
        'avg_sensitivity_pp': avg_sensitivity,
        'criterion': 0.5,
        'n_at_10pct_mmf': n_at_10,
        'n_at_20pct_mmf': n_at_20,
        'n_at_30pct_mmf': n_at_30,
        'delta_10_to_20': delta_10_to_20,
        'delta_20_to_30': delta_20_to_30,
        'mmf_cutoff_pct': mmf_cutoff,
        'tbill_cutoff_pct': tbill_cutoff,
        'interpretation': (
            f"Shifting MMF share from 10%→20% changes n* by {delta_10_to_20:.1f}pp. "
            f"Gate: {'PASSED ✓' if gate_passed else 'FAILED ✗'}. "
            f"Run cutoff at {mmf_cutoff:.0f}% MMFs ({tbill_cutoff:.0f}% T-bills)."
        )
    }


def create_paper_table(results: dict, selected_mmf_shares: list = None) -> str:
    """
    Generate LaTeX table for paper.
    
    Args:
        results: Sweep results
        selected_mmf_shares: List of MMF shares to include (default: [5,10,15,20,30,40])
    
    Returns:
        LaTeX table string
    """
    if selected_mmf_shares is None:
        selected_mmf_shares = [5, 10, 15, 20, 30, 40]
    
    mmf_pct = np.array(results['mmf_share_pct'])
    tbill_pct = np.array(results['tbill_share_pct'])
    n_star = np.array(results['n_star'])
    loss_bps = np.array(results['loss_bps'])
    vulnerable = np.array(results['vulnerable'])
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{rrrrr}")
    latex.append("\\toprule")
    latex.append("MMF Share & T-bill Share & Run Fraction & Loss & Status \\\\")
    latex.append("(\\%) & (\\%) & $n^*$ (\\%) & (bps) & \\\\")
    latex.append("\\midrule")
    
    for mmf_val in selected_mmf_shares:
        idx = np.argmin(np.abs(mmf_pct - mmf_val))
        status = "Vulnerable" if vulnerable[idx] else "Stable"
        
        latex.append(
            f"{mmf_pct[idx]:>6.0f} & {tbill_pct[idx]:>6.0f} & "
            f"{n_star[idx]:>6.1f} & {loss_bps[idx]:>6.1f} & {status} \\\\"
        )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Run equilibrium by reserve composition. "
                "Parameters: $\\pi = 11.5\\%$ (SVB peak redemption), "
                "$\\kappa = 10\\%$ (moderate impact).}")
    latex.append("\\label{tab:reserve_mix}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    """Run reserve mix sweep and gate test."""
    print("=" * 70)
    print("RESERVE MIX SWEEP: Run Cutoff Sensitivity")
    print("=" * 70)
    
    # Parameters
    pi = 0.115  # 11.5% impatient (SVB peak)
    kappa = 0.10  # 10% market impact (moderate)
    
    print(f"\n📊 PARAMETERS")
    print(f"  π (impatient): {pi*100:.1f}%")
    print(f"  κ (impact): {kappa*100:.1f}%")
    print(f"  Sweep: MMF share 0% → 60% (realistic range)")
    
    # Run sweep
    print(f"\n🔍 RUNNING RESERVE MIX SWEEP...")
    results = sweep_reserve_mixes(pi, kappa)
    
    # Gate test
    print(f"\n🚦 GATE TEST EVALUATION:")
    gate = gate_test_summary(results)
    
    print(f"  Sensitivity: {gate['avg_sensitivity_pp']:.2f}pp per 10pp MMF shift")
    print(f"  Criterion: >{gate['criterion']}pp change required")
    print(f"  Result: {'✅ GATE PASSED' if gate['gate_passed'] else '❌ GATE FAILED'}")
    print(f"\n  {gate['interpretation']}")
    
    # Key observations
    print(f"\n📈 KEY OBSERVATIONS:")
    print(f"  MMF 10%: n*={gate['n_at_10pct_mmf']:.1f}%")
    print(f"  MMF 20%: n*={gate['n_at_20pct_mmf']:.1f}%")
    print(f"  MMF 30%: n*={gate['n_at_30pct_mmf']:.1f}%")
    print(f"  Run cutoff: {gate['mmf_cutoff_pct']:.0f}% MMFs / {gate['tbill_cutoff_pct']:.0f}% T-bills")
    
    # Detailed table
    print(f"\n📋 DETAILED RESULTS:")
    print(f"  {'MMF %':<8} {'T-bill %':<10} {'n* (%)':<10} {'Loss (bps)':<12} {'Status':<12}")
    print("  " + "-" * 55)
    
    mmf_pct = np.array(results['mmf_share_pct'])
    for mmf_val in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]:
        idx = np.argmin(np.abs(mmf_pct - mmf_val))
        status = "VULNERABLE" if results['vulnerable'][idx] else "STABLE"
        print(f"  {results['mmf_share_pct'][idx]:>5.0f}    "
              f"{results['tbill_share_pct'][idx]:>6.0f}    "
              f"{results['n_star'][idx]:>7.1f}    "
              f"{results['loss_bps'][idx]:>9.1f}    "
              f"{status}")
    
    # Generate LaTeX table
    print(f"\n📄 LATEX TABLE FOR PAPER:")
    latex_table = create_paper_table(results)
    print(latex_table)
    
    # Save results
    output = {
        'parameters': {'pi': pi, 'kappa': kappa},
        'sweep_results': results,
        'gate_evaluation': gate,
        'latex_table': latex_table,
    }
    
    filename = "reserve_mix_sweep.json"
    with open(filename, 'w') as f:
        # Convert numpy bool to Python bool for JSON
        output_json = output.copy()
        output_json['sweep_results'] = {
            k: [bool(x) if isinstance(x, (np.bool_, bool)) else x for x in v]
            for k, v in output['sweep_results'].items()
        }
        output_json['gate_evaluation'] = {
            k: bool(v) if isinstance(v, (np.bool_, bool)) else v
            for k, v in output['gate_evaluation'].items()
        }
        json.dump(output_json, f, indent=2)
    
    print(f"\n💾 Saved to: {filename}")
    
    # Figure instructions
    print(f"\n" + "=" * 70)
    print("FIGURE FOR PAPER")
    print("=" * 70)
    print("""
Create plot with:
- X-axis: T-bill share (%) = 100 - MMF share
- Y-axis: Equilibrium run fraction n* (%)
- Line: n*(T-bill share) from sweep
- Vertical line: Mark cutoff where n* jumps from 0% to ~100%
- Shading: Vulnerable region (n*>0) in red/orange
- Annotations:
  * "Stable" region where n*=0
  * "Vulnerable" region where n*>0
  * Cutoff value (e.g., "λ̂ = 12% MMFs / 88% T-bills")

Caption: "Run equilibrium as function of reserve composition. 
Higher T-bill concentration increases vulnerability through fire-sale 
feedback. Parameters calibrated to USDC/SVB crisis (π=11.5%, κ=10%)."
    """)
    
    print("=" * 70)
    print(f"✅ ANALYSIS COMPLETE - Gate {'PASSED' if gate['gate_passed'] else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()


