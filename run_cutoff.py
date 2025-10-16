"""
Run Cutoff Model: Two-Tier Liquidity Ladder
============================================

Diamond-Dybvig coordination with fire-sale pricing.

Model:
- t ∈ {0,1,2}
- Mass-1 investors: π impatient (redeem at t=1), (1-π) patient (choose)
- Reserves: λ = Tier-1 (cash/MMFs at par), (1-λ) = Tier-2 (T-bills with fire-sale)
- Total redemptions: R(n) = π + n(1-π), where n = fraction of patient who run

Fire-sale pricing:
- Tier-1 pays at par up to λ
- Tier-2 price: p(R) = 1 - κ·max(0, R-λ)/(1-λ)
- Must sell (R-λ)/p(R) of T-bills to raise (R-λ) dollars

Payoffs (per remaining patient):
- If R ≤ λ: c₂(R) = 1 (no fire-sale)
- If R > λ: c₂(R) = [1-λ - (R-λ)/p(R)] / (1-R)

Run cutoff:
- Find n* where c₂(R(n*)) = 1 (patient indifference: run payoff = wait payoff)
- λ̂ = smallest λ with n*=0 (stable equilibrium)
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class RunCutoffResult:
    """Results from run cutoff analysis"""
    lambda_grid: np.ndarray      # Tier-1 (cash/MMF) shares tested
    tbill_share_grid: np.ndarray # T-bill shares (1-λ) for plotting
    n_star: np.ndarray           # Equilibrium run fraction for each λ
    vulnerable: np.ndarray       # True if n*>0 (vulnerable to runs)
    lambda_cutoff: float         # Smallest λ with n*=0 (or np.nan if always vulnerable)
    
    # Additional diagnostics
    c2_at_equilibrium: np.ndarray  # Wait payoff at equilibrium
    R_at_equilibrium: np.ndarray   # Total redemptions at equilibrium
    fire_sale_price: np.ndarray    # Price p(R*) at equilibrium


def fire_sale_price(R: float, lam: float, kappa: float) -> float:
    """
    Compute fire-sale price for Tier-2 assets (T-bills).
    
    p(R) = 1 - κ·max(0, R-λ)/(1-λ)
    
    Args:
        R: Total redemptions
        lam: Tier-1 share (cash/MMFs at par)
        kappa: Market impact coefficient
    
    Returns:
        Fire-sale price [0,1]
    """
    if R <= lam:
        return 1.0  # No fire-sale needed
    
    denominator = max(1e-8, 1.0 - lam)  # Avoid division by zero
    impact = kappa * (R - lam) / denominator
    
    return max(1e-6, 1.0 - impact)  # Price floor to avoid negative


def c2_wait(R: float, lam: float, kappa: float) -> float:
    """
    Compute t=2 consumption for patient investors who wait.
    
    If R ≤ λ: c₂(R) = 1 (no fire-sale)
    If R > λ: c₂(R) = [1-λ - (R-λ)/p(R)] / (1-R)
    
    Args:
        R: Total redemptions at t=1
        lam: Tier-1 share
        kappa: Market impact coefficient
    
    Returns:
        Per-capita consumption for waiters at t=2
    """
    if R <= lam:
        return 1.0  # No fire-sale, all assets available at par
    
    p = fire_sale_price(R, lam, kappa)
    
    # Remaining T-bills after fire-sale
    # Must sell (R-λ)/p to raise (R-λ) dollars
    tier2_sold = (R - lam) / p
    tier2_remaining = (1.0 - lam) - tier2_sold
    
    # Total assets at t=2
    A2 = max(0.0, tier2_remaining)  # All Tier-1 exhausted, only T-bills remain
    
    # Remaining holders (patient who didn't run)
    remaining_holders = max(1e-8, 1.0 - R)
    
    return A2 / remaining_holders


def solve_n_star(lam: float, pi: float, kappa: float, 
                 tol: float = 1e-6, max_iter: int = 60) -> Tuple[float, dict]:
    """
    Solve for equilibrium run fraction n* via patient indifference.
    
    Patient best response: run if c₂(R(n)) < 1, wait if c₂(R(n)) > 1.
    Fixed point: c₂(R(n*)) = 1.
    
    Uses bisection on n ∈ [0,1] to find c₂(R(n)) = 1.
    
    Args:
        lam: Tier-1 share (cash/MMFs)
        pi: Impatient fraction (must redeem at t=1)
        kappa: Market impact coefficient
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        (n_star, diagnostics_dict)
    """
    def R_of_n(n):
        """Total redemptions given patient run fraction"""
        return pi + n * (1.0 - pi)
    
    def gap(n):
        """Gap: c₂(R(n)) - 1. Want this = 0."""
        R = R_of_n(n)
        return c2_wait(R, lam, kappa) - 1.0
    
    # Boundary checks
    gap_0 = gap(0.0)  # No patient run
    gap_1 = gap(1.0)  # All patient run
    
    if gap_0 >= -tol:
        # Waiting is better even if no one runs → stable (no run)
        n_star = 0.0
        R_star = R_of_n(n_star)
    elif gap_1 <= tol:
        # Running is better even if everyone runs → full run
        n_star = 1.0
        R_star = R_of_n(n_star)
    else:
        # Interior solution: bisection search
        lo, hi = 0.0, 1.0
        
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            g = gap(mid)
            
            if abs(g) < tol:
                n_star = mid
                break
            
            if g > 0:  # c₂ > 1, waiting too good, need more runners
                lo = mid
            else:  # c₂ < 1, running too good, need fewer runners
                hi = mid
        else:
            # Didn't converge, use midpoint
            n_star = 0.5 * (lo + hi)
        
        R_star = R_of_n(n_star)
    
    # Compute diagnostics
    c2_star = c2_wait(R_star, lam, kappa)
    p_star = fire_sale_price(R_star, lam, kappa)
    
    diagnostics = {
        'R_star': R_star,
        'c2_star': c2_star,
        'p_star': p_star,
        'gap': c2_star - 1.0,
        'boundary_check_0': gap_0,
        'boundary_check_1': gap_1,
    }
    
    return n_star, diagnostics


def run_cutoff_curve(pi: float, kappa: float, 
                     lam_grid: Optional[np.ndarray] = None) -> RunCutoffResult:
    """
    Compute run cutoff curve: n* vs λ (Tier-1 share).
    
    For each λ, solve for equilibrium run fraction n*.
    Identify λ̂ = smallest λ with n*=0 (stable).
    
    Args:
        pi: Impatient fraction (must redeem)
        kappa: Market impact coefficient
        lam_grid: Array of Tier-1 shares to test (default: 5% to 95%)
    
    Returns:
        RunCutoffResult with equilibrium for each λ
    """
    if lam_grid is None:
        # Default: cash/MMF share from 5% to 95%
        lam_grid = np.linspace(0.05, 0.95, 91)
    
    # Solve for each λ
    n_star_list = []
    c2_list = []
    R_list = []
    p_list = []
    
    for lam in lam_grid:
        n_star, diag = solve_n_star(lam, pi, kappa)
        n_star_list.append(n_star)
        c2_list.append(diag['c2_star'])
        R_list.append(diag['R_star'])
        p_list.append(diag['p_star'])
    
    n_star = np.array(n_star_list)
    c2_at_eq = np.array(c2_list)
    R_at_eq = np.array(R_list)
    p_at_eq = np.array(p_list)
    
    # Vulnerability: n*>0
    vulnerable = n_star > 1e-4
    
    # Cutoff: smallest λ with n*=0 (stable)
    stable_indices = np.where(~vulnerable)[0]
    if stable_indices.size > 0:
        lambda_cutoff = float(lam_grid[stable_indices[0]])
    else:
        lambda_cutoff = float('nan')  # Always vulnerable
    
    # T-bill share for plotting
    tbill_share_grid = 1.0 - lam_grid
    
    return RunCutoffResult(
        lambda_grid=lam_grid,
        tbill_share_grid=tbill_share_grid,
        n_star=n_star,
        vulnerable=vulnerable,
        lambda_cutoff=lambda_cutoff,
        c2_at_equilibrium=c2_at_eq,
        R_at_equilibrium=R_at_eq,
        fire_sale_price=p_at_eq,
    )


def run_cutoff_sensitivity(pi: float, kappa: float,
                          lam_baseline: float,
                          delta_lam: float = 0.10) -> dict:
    """
    Sensitivity analysis: how does λ̂ change with reserve mix shifts?
    
    Tests ±delta_lam around baseline to compute marginal impact.
    
    Args:
        pi: Impatient fraction
        kappa: Market impact
        lam_baseline: Baseline Tier-1 share
        delta_lam: Change in λ to test (default: ±10pp)
    
    Returns:
        Dictionary with sensitivity metrics
    """
    # Baseline
    n_base, diag_base = solve_n_star(lam_baseline, pi, kappa)
    
    # Increase Tier-1 (more cash/MMFs, fewer T-bills)
    lam_up = min(0.95, lam_baseline + delta_lam)
    n_up, diag_up = solve_n_star(lam_up, pi, kappa)
    
    # Decrease Tier-1 (fewer cash/MMFs, more T-bills)
    lam_down = max(0.05, lam_baseline - delta_lam)
    n_down, diag_down = solve_n_star(lam_down, pi, kappa)
    
    # Marginal impact
    # How much does n* change per pp change in λ
    if delta_lam > 0:
        marginal_up = (n_up - n_base) / (delta_lam * 100)  # per pp
        marginal_down = (n_base - n_down) / (delta_lam * 100)
        marginal_avg = (marginal_up + marginal_down) / 2
    else:
        marginal_avg = 0.0
    
    # Gate evaluation
    # Proceed if realistic shifts move n* significantly
    delta_n_total = abs(n_up - n_down)
    proceed = delta_n_total > 0.005  # >0.5pp change in run fraction
    
    return {
        'baseline': {
            'lambda': lam_baseline,
            'n_star': n_base,
            'R_star': diag_base['R_star'],
            'c2_star': diag_base['c2_star'],
            'vulnerable': n_base > 1e-4,
        },
        'lambda_up': {
            'lambda': lam_up,
            'n_star': n_up,
            'delta_n': n_up - n_base,
        },
        'lambda_down': {
            'lambda': lam_down,
            'n_star': n_down,
            'delta_n': n_down - n_base,
        },
        'sensitivity': {
            'marginal_dn_dlam_pp': marginal_avg,  # Δn* per pp increase in λ
            'delta_n_total': delta_n_total,
            'delta_lam_tested': delta_lam * 100,  # in pp
        },
        'gate_evaluation': {
            'proceed': proceed,
            'reason': 'Significant sensitivity' if proceed else 'Flat response',
        }
    }


def expected_loss_at_shock(lam: float, pi: float, kappa: float, 
                           shock_R: float) -> dict:
    """
    Compute expected loss under a given redemption shock R.
    
    Loss = (1 - c₂(R)) per remaining holder, in bps.
    
    Args:
        lam: Tier-1 share
        pi: Impatient fraction (for equilibrium context)
        kappa: Market impact
        shock_R: Total redemption fraction (e.g., 0.10 for 10%)
    
    Returns:
        Dictionary with loss metrics
    """
    c2 = c2_wait(shock_R, lam, kappa)
    loss_per_holder = max(0.0, 1.0 - c2)
    loss_bps = loss_per_holder * 10000
    
    p = fire_sale_price(shock_R, lam, kappa)
    
    # Total system loss (from fire-sale)
    if shock_R > lam:
        tier2_sold = (shock_R - lam) / p
        fire_sale_discount = 1.0 - p
        total_loss = tier2_sold * fire_sale_discount
    else:
        total_loss = 0.0
    
    return {
        'shock_R': shock_R,
        'c2_waiters': c2,
        'loss_per_holder': loss_per_holder,
        'loss_bps': loss_bps,
        'fire_sale_price': p,
        'total_system_loss': total_loss,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("RUN CUTOFF MODEL: Two-Tier Liquidity Ladder")
    print("=" * 70)
    
    # Example calibration (USDC-like)
    pi = 0.115  # 11.5% impatient (SVB peak)
    kappa = 0.20  # 20% impact (conservative for T-bills in stress)
    
    print(f"\n📊 PARAMETERS")
    print(f"  Impatient fraction (π): {pi*100:.1f}%")
    print(f"  Market impact (κ): {kappa*100:.1f}%")
    
    # Compute run cutoff curve
    print(f"\n🎯 COMPUTING RUN CUTOFF CURVE...")
    result = run_cutoff_curve(pi, kappa)
    
    print(f"\n✓ Results:")
    print(f"  λ̂ (cutoff): {result.lambda_cutoff*100:.1f}% Tier-1")
    print(f"  → T-bill share at cutoff: {(1-result.lambda_cutoff)*100:.1f}%")
    
    # Find some key points
    idx_20 = np.argmin(np.abs(result.lambda_grid - 0.20))
    idx_40 = np.argmin(np.abs(result.lambda_grid - 0.40))
    idx_60 = np.argmin(np.abs(result.lambda_grid - 0.60))
    
    print(f"\n📈 EQUILIBRIUM RUN FRACTIONS:")
    print(f"  λ=20% (80% T-bills): n*={result.n_star[idx_20]*100:.1f}% {'VULNERABLE' if result.vulnerable[idx_20] else 'STABLE'}")
    print(f"  λ=40% (60% T-bills): n*={result.n_star[idx_40]*100:.1f}% {'VULNERABLE' if result.vulnerable[idx_40] else 'STABLE'}")
    print(f"  λ=60% (40% T-bills): n*={result.n_star[idx_60]*100:.1f}% {'VULNERABLE' if result.vulnerable[idx_60] else 'STABLE'}")
    
    # Sensitivity analysis
    print(f"\n🔍 SENSITIVITY ANALYSIS (baseline λ=40%):")
    sens = run_cutoff_sensitivity(pi, kappa, lam_baseline=0.40, delta_lam=0.10)
    
    print(f"  Baseline: n*={sens['baseline']['n_star']*100:.2f}%")
    print(f"  +10pp cash: n*={sens['lambda_up']['n_star']*100:.2f}% (Δn={sens['lambda_up']['delta_n']*100:.2f}pp)")
    print(f"  -10pp cash: n*={sens['lambda_down']['n_star']*100:.2f}% (Δn={sens['lambda_down']['delta_n']*100:.2f}pp)")
    print(f"  Marginal: Δn/Δλ = {sens['sensitivity']['marginal_dn_dlam_pp']:.3f} per pp")
    
    print(f"\n🚦 GATE EVALUATION:")
    gate = sens['gate_evaluation']
    print(f"  Decision: {'✓ PROCEED' if gate['proceed'] else '✗ REVISIT'}")
    print(f"  Reason: {gate['reason']}")
    
    # Expected loss table
    print(f"\n💥 EXPECTED LOSS (10% redemption shock):")
    print(f"  {'λ (Tier-1)':<15} {'T-bill %':<12} {'Loss (bps)':<15} {'c₂':<10}")
    print("  " + "-" * 52)
    
    for lam_val in [0.20, 0.30, 0.40, 0.50, 0.60]:
        loss = expected_loss_at_shock(lam_val, pi, kappa, shock_R=0.10)
        tbill_pct = (1 - lam_val) * 100
        print(f"  {lam_val*100:>5.0f}%          {tbill_pct:>5.0f}%       {loss['loss_bps']:>8.1f}       {loss['c2_waiters']:>6.3f}")
    
    print("\n" + "=" * 70)
    print("Run cutoff model ready!")

