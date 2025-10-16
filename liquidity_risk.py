"""
Liquidity-at-Risk (LaR) Module
================================

Translates two-tier run model into operational risk metrics:
1. LaR: Maximum redemptions before fire-sales (or depeg ≤ d bps)
2. Run cutoff mapping
3. Time-to-critical-threshold

Builds on run_cutoff.py two-tier model.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from run_cutoff import fire_sale_price as p_of_R, solve_n_star


def liquidity_at_risk(lam: float, pi: float, kappa: float,
                      max_depeg_bps: float = 100) -> Dict:
    """
    Compute Liquidity-at-Risk: maximum redemptions before hitting depeg threshold.
    
    LaR = max R such that depeg ≤ max_depeg_bps
    
    Args:
        lam: Tier-1 share (liquid reserves)
        pi: Impatient fraction
        kappa: Market impact coefficient
        max_depeg_bps: Maximum acceptable depeg (basis points)
    
    Returns:
        Dictionary with LaR metrics
    """
    max_depeg = max_depeg_bps / 10000  # Convert to fraction
    
    # Case 1: No fire-sale (R ≤ λ) → no depeg
    if max_depeg >= 0:
        # Can handle all redemptions up to λ at par
        LaR_no_firesale = lam
    else:
        LaR_no_firesale = 0.0
    
    # Case 2: With fire-sales (R > λ)
    # Depeg occurs when p(R) < 1
    # p(R) = 1 - κ·(R-λ)/(1-λ)
    # Depeg = 1 - p(R) = κ·(R-λ)/(1-λ)
    # 
    # Solve: κ·(R-λ)/(1-λ) = max_depeg
    # R = λ + max_depeg·(1-λ)/κ
    
    if kappa > 0 and (1 - lam) > 0:
        R_at_max_depeg = lam + max_depeg * (1 - lam) / kappa
        R_at_max_depeg = min(R_at_max_depeg, 1.0)  # Cannot exceed 100%
    else:
        R_at_max_depeg = lam
    
    # LaR is the minimum of the two constraints
    LaR = max(LaR_no_firesale, R_at_max_depeg)
    
    # Corresponding depeg at LaR
    depeg_at_LaR = (1.0 - p_of_R(LaR, lam, kappa)) * 10000
    
    # Buffer: LaR - baseline redemptions (π)
    buffer = max(0, LaR - pi)
    buffer_pct = buffer * 100
    
    # Run vulnerability at this LaR level
    n_star, _ = solve_n_star(lam, pi, kappa)
    vulnerable = n_star > 1e-4
    
    return {
        'LaR': LaR,
        'LaR_pct': LaR * 100,
        'max_depeg_bps': max_depeg_bps,
        'depeg_at_LaR_bps': depeg_at_LaR,
        'buffer_pct': buffer_pct,
        'baseline_redemption_pi': pi,
        'vulnerable': vulnerable,
        'n_star': n_star,
    }


def LaR_curve(lam: float, pi: float, kappa: float,
              depeg_thresholds: np.ndarray = None) -> Dict:
    """
    Compute LaR curve across multiple depeg thresholds.
    
    Shows: maximum redemptions before hitting various depeg levels.
    
    Args:
        lam: Tier-1 share
        pi: Impatient fraction
        kappa: Market impact
        depeg_thresholds: Array of depeg thresholds in bps
    
    Returns:
        Dictionary with LaR curve data
    """
    if depeg_thresholds is None:
        # Default: 0 to 1000 bps (0% to 10% depeg)
        depeg_thresholds = np.array([0, 10, 25, 50, 100, 200, 500, 1000])
    
    results = {
        'depeg_threshold_bps': [],
        'LaR_pct': [],
        'buffer_pct': [],
        'vulnerable': [],
    }
    
    for depeg_bps in depeg_thresholds:
        lar = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=depeg_bps)
        
        results['depeg_threshold_bps'].append(depeg_bps)
        results['LaR_pct'].append(lar['LaR_pct'])
        results['buffer_pct'].append(lar['buffer_pct'])
        results['vulnerable'].append(lar['vulnerable'])
    
    return results


def LaR_heatmap(lam_grid: np.ndarray, 
                shock_pcts: np.ndarray,
                kappa_vals: np.ndarray) -> Dict:
    """
    Generate heatmap: (shock %, κ) → depeg bps.
    
    Shows fire-sale impact across stress scenarios and market conditions.
    
    Args:
        lam_grid: Array of Tier-1 shares to test
        shock_pcts: Array of shock sizes (% of supply)
        kappa_vals: Array of market impact coefficients
    
    Returns:
        Dictionary with heatmap data
    """
    # Use single lam for now (can extend to 3D)
    if isinstance(lam_grid, np.ndarray):
        lam = lam_grid[0] if len(lam_grid) > 0 else 0.20
    else:
        lam = lam_grid
    
    # Initialize heatmap
    heatmap = np.zeros((len(shock_pcts), len(kappa_vals)))
    
    for i, shock_pct in enumerate(shock_pcts):
        R = shock_pct / 100  # Convert to fraction
        
        for j, kappa in enumerate(kappa_vals):
            # Compute depeg at this (shock, kappa)
            p = p_of_R(R, lam, kappa)
            depeg = (1.0 - p) * 10000  # In bps
            
            heatmap[i, j] = depeg
    
    return {
        'shock_pcts': shock_pcts,
        'kappa_vals': kappa_vals,
        'depeg_bps_heatmap': heatmap,
        'lambda_tested': lam,
    }


def LaR_vs_reserve_mix(lam_grid: np.ndarray,
                       pi: float,
                       kappa: float,
                       max_depeg_bps: float = 100) -> Dict:
    """
    Plot LaR vs T-bill share (reserve composition).
    
    Key deliverable: shows how reserve mix affects liquidity buffer.
    
    Args:
        lam_grid: Array of Tier-1 shares
        pi: Impatient fraction
        kappa: Market impact
        max_depeg_bps: Depeg threshold
    
    Returns:
        Dictionary with LaR by reserve mix
    """
    results = {
        'lambda_pct': [],
        'tbill_share_pct': [],
        'LaR_pct': [],
        'buffer_pct': [],
        'vulnerable': [],
        'n_star': [],
    }
    
    for lam in lam_grid:
        lar = liquidity_at_risk(lam, pi, kappa, max_depeg_bps)
        
        results['lambda_pct'].append(lam * 100)
        results['tbill_share_pct'].append((1 - lam) * 100)
        results['LaR_pct'].append(lar['LaR_pct'])
        results['buffer_pct'].append(lar['buffer_pct'])
        results['vulnerable'].append(lar['vulnerable'])
        results['n_star'].append(lar['n_star'])
    
    return results


def time_to_critical_threshold(lam: float, 
                               pi: float, 
                               kappa: float,
                               redemption_rate_per_second: float,
                               block_time: float = 12.0) -> Dict:
    """
    Time until critical redemption threshold hit (on-chain friction).
    
    Combines:
    - Run cutoff from two-tier model
    - Block time delays from onchain_frictions.py
    
    Args:
        lam: Tier-1 share
        pi: Impatient baseline
        kappa: Market impact
        redemption_rate_per_second: Redemption flow (fraction/sec)
        block_time: Blockchain block time (seconds)
    
    Returns:
        Dictionary with time-to-threshold metrics
    """
    # Critical thresholds
    threshold_5pct = 0.05
    threshold_10pct = 0.10
    
    # LaR at various depegs
    lar_0bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=0)
    lar_100bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=100)
    
    # Time to hit thresholds (with block quantization)
    def time_to_redeem(target_pct: float) -> float:
        if redemption_rate_per_second <= 0:
            return float('inf')
        
        # Continuous time
        time_continuous = (target_pct - pi) / redemption_rate_per_second
        
        # Quantized to blocks
        blocks_needed = np.ceil(time_continuous / block_time)
        time_quantized = blocks_needed * block_time
        
        return max(0, time_quantized)
    
    time_5pct = time_to_redeem(threshold_5pct)
    time_10pct = time_to_redeem(threshold_10pct)
    time_LaR = time_to_redeem(lar_100bp['LaR'])
    
    return {
        'time_to_5pct_seconds': time_5pct,
        'time_to_10pct_seconds': time_10pct,
        'time_to_LaR_seconds': time_LaR,
        'LaR_pct': lar_100bp['LaR_pct'],
        'circuit_breaker_window': min(time_5pct, 120),  # Practical window
        'block_time': block_time,
        'redemption_rate_per_sec': redemption_rate_per_second,
    }


def comprehensive_LaR_report(lam: float, pi: float, kappa: float) -> Dict:
    """
    Generate comprehensive LaR report for a given reserve configuration.
    
    Args:
        lam: Tier-1 share
        pi: Impatient fraction
        kappa: Market impact
    
    Returns:
        Dictionary with all LaR metrics
    """
    # 1. LaR at key depeg thresholds
    lar_0bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=0)
    lar_50bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=50)
    lar_100bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=100)
    lar_500bp = liquidity_at_risk(lam, pi, kappa, max_depeg_bps=500)
    
    # 2. LaR curve
    lar_curve = LaR_curve(lam, pi, kappa)
    
    # 3. Time-to-threshold (assume 10% daily redemption = 0.001157% per second)
    time_metrics = time_to_critical_threshold(
        lam, pi, kappa, 
        redemption_rate_per_second=0.10 / (24 * 3600)
    )
    
    # 4. Run vulnerability
    n_star, _ = solve_n_star(lam, pi, kappa)
    vulnerable = n_star > 1e-4
    
    return {
        'parameters': {
            'lambda_pct': lam * 100,
            'tbill_share_pct': (1 - lam) * 100,
            'pi_pct': pi * 100,
            'kappa_pct': kappa * 100,
        },
        'LaR_by_threshold': {
            '0bp': lar_0bp,
            '50bp': lar_50bp,
            '100bp': lar_100bp,
            '500bp': lar_500bp,
        },
        'LaR_curve': lar_curve,
        'time_to_threshold': time_metrics,
        'run_vulnerability': {
            'n_star': n_star,
            'vulnerable': vulnerable,
        },
        'risk_summary': {
            'LaR_100bp_pct': lar_100bp['LaR_pct'],
            'buffer_100bp_pct': lar_100bp['buffer_pct'],
            'circuit_breaker_window_sec': time_metrics['circuit_breaker_window'],
            'run_status': 'VULNERABLE' if vulnerable else 'STABLE',
        }
    }


if __name__ == "__main__":
    print("=" * 70)
    print("LIQUIDITY-AT-RISK (LaR) MODULE")
    print("=" * 70)
    
    # Example: USDC-like parameters
    lam = 0.20    # 20% liquid (MMFs)
    pi = 0.115    # 11.5% impatient (SVB peak)
    kappa = 0.10  # 10% market impact
    
    print(f"\n📊 PARAMETERS")
    print(f"  λ (Tier-1): {lam*100:.0f}%")
    print(f"  π (impatient): {pi*100:.1f}%")
    print(f"  κ (impact): {kappa*100:.0f}%")
    
    # Comprehensive report
    print(f"\n🔍 COMPREHENSIVE LaR REPORT")
    report = comprehensive_LaR_report(lam, pi, kappa)
    
    print(f"\n💧 LaR BY DEPEG THRESHOLD:")
    for threshold, lar_data in report['LaR_by_threshold'].items():
        print(f"  {threshold:>5}: LaR={lar_data['LaR_pct']:>5.1f}%, "
              f"buffer={lar_data['buffer_pct']:>5.1f}%, "
              f"depeg={lar_data['depeg_at_LaR_bps']:>6.1f}bps")
    
    print(f"\n⏱️  TIME TO CRITICAL THRESHOLDS:")
    time_m = report['time_to_threshold']
    print(f"  5% redemption:  {time_m['time_to_5pct_seconds']:>6.0f}s")
    print(f"  10% redemption: {time_m['time_to_10pct_seconds']:>6.0f}s")
    print(f"  LaR threshold:  {time_m['time_to_LaR_seconds']:>6.0f}s")
    print(f"  Circuit breaker window: {time_m['circuit_breaker_window']:>6.0f}s")
    
    print(f"\n🎯 RUN VULNERABILITY:")
    vuln = report['run_vulnerability']
    print(f"  n*: {vuln['n_star']*100:.1f}%")
    print(f"  Status: {report['risk_summary']['run_status']}")
    
    # LaR vs reserve mix
    print(f"\n📈 LaR VS RESERVE MIX (100bp depeg threshold):")
    lam_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    lar_by_mix = LaR_vs_reserve_mix(lam_grid, pi, kappa, max_depeg_bps=100)
    
    print(f"  {'λ (MMF%)':<10} {'T-bill%':<10} {'LaR%':<10} {'Buffer%':<10} {'Status':<12}")
    print("  " + "-" * 55)
    for i in range(len(lar_by_mix['lambda_pct'])):
        status = "VULNERABLE" if lar_by_mix['vulnerable'][i] else "STABLE"
        print(f"  {lar_by_mix['lambda_pct'][i]:>5.0f}      "
              f"{lar_by_mix['tbill_share_pct'][i]:>6.0f}    "
              f"{lar_by_mix['LaR_pct'][i]:>7.1f}    "
              f"{lar_by_mix['buffer_pct'][i]:>7.1f}    "
              f"{status}")
    
    # Heatmap
    print(f"\n🔥 DEPEG HEATMAP (shock %, κ):")
    shock_pcts = np.array([5, 10, 15, 20, 30])
    kappa_vals = np.array([0.05, 0.10, 0.15, 0.20])
    heatmap = LaR_heatmap(lam, shock_pcts, kappa_vals)
    
    print(f"  Shock% \\ κ    ", end="")
    for k in kappa_vals:
        print(f"{k*100:>8.0f}%  ", end="")
    print()
    print("  " + "-" * 50)
    
    for i, shock in enumerate(shock_pcts):
        print(f"  {shock:>5.0f}%        ", end="")
        for j in range(len(kappa_vals)):
            depeg = heatmap['depeg_bps_heatmap'][i, j]
            print(f"{depeg:>7.0f}bp ", end="")
        print()
    
    print("\n" + "=" * 70)
    print("LaR module ready! Use for liquidity risk monitoring.")

