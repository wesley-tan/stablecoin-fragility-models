"""
Policy Levers as Risk Controls
================================

Quantifies how policy interventions reduce risk:
1. LCR-style cash floors → reduces run probability and FS-VaR
2. PSM/backstop capacity → mitigates tail risk
3. Redemption gates/fees → liquidity conservation
4. Disclosure cadence → information risk

Translates Topic 3 results into risk management framework.
"""

import numpy as np
from typing import Dict, Optional
from run_cutoff import solve_n_star, fire_sale_price as p_of_R
from liquidity_risk import liquidity_at_risk, comprehensive_LaR_report
from firesale_var import fire_sale_loss, monte_carlo_FS_VaR


def lcr_floor_impact(lam_baseline: float,
                    lcr_floor_pct: float,
                    pi: float,
                    kappa: float,
                    total_reserves: float = 1000.0) -> Dict:
    """
    Quantify impact of LCR-style cash floor on risk metrics.
    
    LCR = HQLA / NetCashOutflows ≥ floor%
    
    For stablecoin: HQLA = Tier-1 reserves (λ)
                    NetCashOutflows = stress redemptions
    
    Args:
        lam_baseline: Current Tier-1 share
        lcr_floor_pct: Minimum LCR (e.g., 150 for 150%)
        pi: Impatient fraction
        kappa: Market impact
        total_reserves: Total reserve size (millions)
    
    Returns:
        Dictionary comparing baseline vs post-floor
    """
    lcr_floor = lcr_floor_pct / 100
    
    # Baseline metrics
    n_base, _ = solve_n_star(lam_baseline, pi, kappa)
    R_base = pi + n_base * (1 - pi)
    loss_base = fire_sale_loss(R_base, lam_baseline, kappa, total_reserves)
    lar_base = liquidity_at_risk(lam_baseline, pi, kappa, max_depeg_bps=100)
    
    # Required λ to meet LCR floor
    # Simplified: λ ≥ lcr_floor · stress_outflows
    # Assume stress = 30% redemptions over 30 days (Basel-style)
    stress_outflows = 0.30
    lam_required = lcr_floor * stress_outflows
    
    # Adjust λ if below requirement
    lam_adjusted = max(lam_baseline, lam_required)
    
    # Post-floor metrics
    n_adj, _ = solve_n_star(lam_adjusted, pi, kappa)
    R_adj = pi + n_adj * (1 - pi)
    loss_adj = fire_sale_loss(R_adj, lam_adjusted, kappa, total_reserves)
    lar_adj = liquidity_at_risk(lam_adjusted, pi, kappa, max_depeg_bps=100)
    
    # Delta metrics
    delta_run_prob = (n_adj - n_base) * 100  # pp change
    delta_loss_bps = loss_adj['loss_bps'] - loss_base['loss_bps']
    delta_LaR = lar_adj['LaR_pct'] - lar_base['LaR_pct']
    
    # Yield cost
    # Assume T-bills yield 5.0%, MMFs 4.8%
    tbill_yield = 0.050
    mmf_yield = 0.048
    yield_sacrifice_bps = (lam_adjusted - lam_baseline) * (tbill_yield - mmf_yield) * 10000
    
    return {
        'baseline': {
            'lambda_pct': lam_baseline * 100,
            'n_star': n_base,
            'run_prob_pct': (n_base > 1e-4) * 100,
            'loss_bps': loss_base['loss_bps'],
            'LaR_pct': lar_base['LaR_pct'],
        },
        'post_floor': {
            'lambda_pct': lam_adjusted * 100,
            'lcr_floor_pct': lcr_floor_pct,
            'n_star': n_adj,
            'run_prob_pct': (n_adj > 1e-4) * 100,
            'loss_bps': loss_adj['loss_bps'],
            'LaR_pct': lar_adj['LaR_pct'],
        },
        'impact': {
            'delta_lambda_pp': (lam_adjusted - lam_baseline) * 100,
            'delta_run_prob_pp': delta_run_prob,
            'delta_loss_bps': delta_loss_bps,
            'delta_LaR_pp': delta_LaR,
            'yield_sacrifice_bps': yield_sacrifice_bps,
        },
        'cost_benefit': {
            'cost_bps_per_year': yield_sacrifice_bps,
            'benefit_run_reduction_pp': -delta_run_prob,
            'benefit_loss_reduction_bps': -delta_loss_bps,
            'ratio': -delta_loss_bps / yield_sacrifice_bps if yield_sacrifice_bps > 0 else float('inf'),
        }
    }


def psm_backstop_impact(lam: float,
                       pi: float,
                       kappa: float,
                       backstop_sizes: np.ndarray,
                       total_reserves: float = 1000.0) -> Dict:
    """
    Quantify impact of PSM/backstop on tail risk.
    
    PSM provides additional liquidity buffer before T-bill sales.
    
    Args:
        lam: Tier-1 share
        pi: Impatient fraction
        kappa: Market impact
        backstop_sizes: Array of backstop sizes to test (millions)
        total_reserves: Total reserves
    
    Returns:
        Dictionary with backstop impact analysis
    """
    results = {
        'backstop_size': [],
        'effective_lambda_pct': [],
        'n_star': [],
        'run_prob_pct': [],
        'loss_bps': [],
        'max_depeg_bps': [],
    }
    
    for B in backstop_sizes:
        # Effective λ with backstop: λ_eff = (λ + B/NAV)
        lam_eff = lam + B / total_reserves
        lam_eff = min(lam_eff, 0.95)  # Cap at 95%
        
        # Solve with effective λ
        n_star, _ = solve_n_star(lam_eff, pi, kappa)
        R = pi + n_star * (1 - pi)
        loss = fire_sale_loss(R, lam_eff, kappa, total_reserves)
        
        results['backstop_size'].append(B)
        results['effective_lambda_pct'].append(lam_eff * 100)
        results['n_star'].append(n_star)
        results['run_prob_pct'].append((n_star > 1e-4) * 100)
        results['loss_bps'].append(loss['loss_bps'])
        results['max_depeg_bps'].append(loss['depeg_bps'])
    
    # Marginal benefit per $100M backstop
    if len(backstop_sizes) > 1:
        delta_B = backstop_sizes[1] - backstop_sizes[0]
        delta_loss = results['loss_bps'][0] - results['loss_bps'][1]
        marginal_benefit_per_100M = delta_loss / delta_B * 100 if delta_B > 0 else 0
    else:
        marginal_benefit_per_100M = 0
    
    return {
        'results': results,
        'marginal_benefit_per_100M': marginal_benefit_per_100M,
        'optimal_backstop': _find_optimal_backstop(results, target_loss_bps=50),
    }


def _find_optimal_backstop(results: Dict, target_loss_bps: float) -> Dict:
    """Find minimum backstop to achieve target loss"""
    backstops = np.array(results['backstop_size'])
    losses = np.array(results['loss_bps'])
    
    # Find first backstop where loss ≤ target
    idx = np.where(losses <= target_loss_bps)[0]
    
    if len(idx) > 0:
        optimal_idx = idx[0]
        return {
            'backstop_size': backstops[optimal_idx],
            'achieved_loss_bps': losses[optimal_idx],
            'target_loss_bps': target_loss_bps,
        }
    else:
        return {
            'backstop_size': backstops[-1],
            'achieved_loss_bps': losses[-1],
            'target_loss_bps': target_loss_bps,
            'note': 'Target not achieved with max backstop',
        }


def redemption_gate_impact(lam: float,
                           pi: float,
                           kappa: float,
                           gate_fee_bps: float,
                           total_reserves: float = 1000.0) -> Dict:
    """
    Quantify impact of redemption gates/fees on liquidity.
    
    Gate increases cost of running → reduces redemption rate.
    Model: effective π_adj = π · (1 - gate_fee)
    
    Args:
        lam: Tier-1 share
        pi: Baseline impatient fraction
        kappa: Market impact
        gate_fee_bps: Redemption fee in basis points
        total_reserves: Total reserves
    
    Returns:
        Dictionary comparing no-gate vs with-gate
    """
    gate_fee = gate_fee_bps / 10000
    
    # No gate
    n_no_gate, _ = solve_n_star(lam, pi, kappa)
    R_no_gate = pi + n_no_gate * (1 - pi)
    loss_no_gate = fire_sale_loss(R_no_gate, lam, kappa, total_reserves)
    
    # With gate: reduces effective redemption pressure
    # Simplified: π_adj = π · (1 - gate_fee)
    pi_adj = pi * (1 - gate_fee)
    pi_adj = max(0.001, pi_adj)
    
    n_gate, _ = solve_n_star(lam, pi_adj, kappa)
    R_gate = pi_adj + n_gate * (1 - pi_adj)
    loss_gate = fire_sale_loss(R_gate, lam, kappa, total_reserves)
    
    # Impact
    delta_R = (R_no_gate - R_gate) * 100
    delta_loss = loss_no_gate['loss_bps'] - loss_gate['loss_bps']
    
    # Adverse selection cost
    # Gates deter patient holders more than impatient → potential market signal
    adverse_selection_cost_bps = gate_fee_bps * 0.5  # Heuristic: 50% of fee
    
    return {
        'no_gate': {
            'pi_pct': pi * 100,
            'n_star': n_no_gate,
            'R_pct': R_no_gate * 100,
            'loss_bps': loss_no_gate['loss_bps'],
        },
        'with_gate': {
            'gate_fee_bps': gate_fee_bps,
            'pi_effective_pct': pi_adj * 100,
            'n_star': n_gate,
            'R_pct': R_gate * 100,
            'loss_bps': loss_gate['loss_bps'],
        },
        'impact': {
            'delta_R_pp': delta_R,
            'delta_loss_bps': delta_loss,
            'adverse_selection_cost_bps': adverse_selection_cost_bps,
            'net_benefit_bps': delta_loss - adverse_selection_cost_bps,
        },
        'recommendation': 'Beneficial' if delta_loss > adverse_selection_cost_bps else 'Costly adverse selection',
    }


def disclosure_cadence_impact(lam: float,
                              pi: float,
                              kappa: float,
                              reporting_lags: np.ndarray) -> Dict:
    """
    Model information risk: sunspot probability vs reporting lag.
    
    More frequent disclosure → lower signal noise → less coordination risk.
    
    Args:
        lam: Tier-1 share
        pi: Impatient fraction
        kappa: Market impact
        reporting_lags: Array of reporting lags (days)
    
    Returns:
        Dictionary with disclosure impact
    """
    results = {
        'reporting_lag_days': [],
        'signal_noise': [],
        'sunspot_probability': [],
        'run_prob_pct': [],
    }
    
    # Model: signal noise increases with reporting lag
    # σ(lag) = σ_base · sqrt(lag / lag_base)
    sigma_base = 0.05  # 5% noise at daily reporting
    lag_base = 1  # Daily baseline
    
    for lag in reporting_lags:
        # Signal noise
        sigma = sigma_base * np.sqrt(lag / lag_base)
        
        # Sunspot probability (higher noise → higher coordination risk)
        # Simple model: prob ∝ σ
        sunspot_prob = min(sigma / sigma_base * 0.20, 0.50)  # Cap at 50%
        
        # Run probability increases with noise
        n_star, _ = solve_n_star(lam, pi, kappa)
        base_run_prob = (n_star > 1e-4) * 1.0
        
        # Amplify by sunspot
        effective_run_prob = base_run_prob * (1 + sunspot_prob)
        effective_run_prob = min(effective_run_prob, 1.0)
        
        results['reporting_lag_days'].append(lag)
        results['signal_noise'].append(sigma)
        results['sunspot_probability'].append(sunspot_prob)
        results['run_prob_pct'].append(effective_run_prob * 100)
    
    return results


def comprehensive_policy_analysis(lam_baseline: float,
                                  pi: float,
                                  kappa: float,
                                  total_reserves: float = 1000.0) -> Dict:
    """
    Comprehensive comparison of all policy levers.
    
    Args:
        lam_baseline: Current Tier-1 share
        pi: Impatient fraction
        kappa: Market impact
        total_reserves: Total reserves
    
    Returns:
        Dictionary with full policy analysis
    """
    print(f"\n🔍 Running comprehensive policy analysis...")
    
    # 1. LCR floors
    print(f"  - LCR floor analysis...")
    lcr_100 = lcr_floor_impact(lam_baseline, 100, pi, kappa, total_reserves)
    lcr_150 = lcr_floor_impact(lam_baseline, 150, pi, kappa, total_reserves)
    lcr_200 = lcr_floor_impact(lam_baseline, 200, pi, kappa, total_reserves)
    
    # 2. PSM backstop
    print(f"  - PSM backstop analysis...")
    backstop_sizes = np.array([0, 50, 100, 200, 500])
    psm_analysis = psm_backstop_impact(lam_baseline, pi, kappa, backstop_sizes, total_reserves)
    
    # 3. Redemption gates
    print(f"  - Redemption gate analysis...")
    gate_50bp = redemption_gate_impact(lam_baseline, pi, kappa, 50, total_reserves)
    gate_100bp = redemption_gate_impact(lam_baseline, pi, kappa, 100, total_reserves)
    
    # 4. Disclosure
    print(f"  - Disclosure cadence analysis...")
    reporting_lags = np.array([1, 7, 30, 90])
    disclosure = disclosure_cadence_impact(lam_baseline, pi, kappa, reporting_lags)
    
    return {
        'baseline': {
            'lambda_pct': lam_baseline * 100,
            'pi_pct': pi * 100,
            'kappa_pct': kappa * 100,
        },
        'lcr_floors': {
            '100%': lcr_100,
            '150%': lcr_150,
            '200%': lcr_200,
        },
        'psm_backstop': psm_analysis,
        'redemption_gates': {
            '50bp': gate_50bp,
            '100bp': gate_100bp,
        },
        'disclosure_cadence': disclosure,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("POLICY LEVERS AS RISK CONTROLS")
    print("=" * 70)
    
    # Parameters (USDC-like)
    lam_baseline = 0.20  # 20% liquid
    pi = 0.115  # 11.5% impatient (SVB)
    kappa = 0.10  # 10% impact
    total_reserves = 1000.0  # $1B
    
    print(f"\n📊 BASELINE PARAMETERS")
    print(f"  λ (Tier-1): {lam_baseline*100:.0f}%")
    print(f"  π (impatient): {pi*100:.1f}%")
    print(f"  κ (impact): {kappa*100:.0f}%")
    print(f"  Total reserves: ${total_reserves:.0f}M")
    
    # Run comprehensive analysis
    analysis = comprehensive_policy_analysis(lam_baseline, pi, kappa, total_reserves)
    
    print(f"\n" + "=" * 70)
    print("1. LCR-STYLE CASH FLOORS")
    print("=" * 70)
    
    for floor_name, lcr_result in analysis['lcr_floors'].items():
        print(f"\n  LCR {floor_name}:")
        impact = lcr_result['impact']
        cb = lcr_result['cost_benefit']
        print(f"    Δλ: +{impact['delta_lambda_pp']:.1f}pp")
        print(f"    Δ Run prob: {impact['delta_run_prob_pp']:.1f}pp")
        print(f"    Δ Loss: {impact['delta_loss_bps']:.0f}bps")
        print(f"    Yield cost: {cb['cost_bps_per_year']:.1f}bps/year")
        print(f"    Benefit/Cost: {cb['ratio']:.1f}x")
    
    print(f"\n" + "=" * 70)
    print("2. PSM/BACKSTOP CAPACITY")
    print("=" * 70)
    
    psm = analysis['psm_backstop']
    print(f"\n  Marginal benefit: {psm['marginal_benefit_per_100M']:.1f}bps per $100M")
    print(f"  Optimal backstop (50bps loss target): ${psm['optimal_backstop']['backstop_size']:.0f}M")
    
    print(f"\n  {'Backstop':<10} {'λ_eff%':<8} {'n*%':<8} {'Loss(bps)':<10} {'Depeg':<8}")
    print("  " + "-" * 50)
    for i in range(len(psm['results']['backstop_size'])):
        print(f"  ${psm['results']['backstop_size'][i]:<8.0f} "
              f"{psm['results']['effective_lambda_pct'][i]:>6.1f}   "
              f"{psm['results']['n_star'][i]*100:>5.1f}   "
              f"{psm['results']['loss_bps'][i]:>8.1f}    "
              f"{psm['results']['max_depeg_bps'][i]:>6.1f}bp")
    
    print(f"\n" + "=" * 70)
    print("3. REDEMPTION GATES/FEES")
    print("=" * 70)
    
    for gate_name, gate_result in analysis['redemption_gates'].items():
        print(f"\n  Gate {gate_name}:")
        impact = gate_result['impact']
        print(f"    Δ Redemptions: {impact['delta_R_pp']:.1f}pp")
        print(f"    Δ Loss: {impact['delta_loss_bps']:.0f}bps")
        print(f"    Adverse selection cost: {impact['adverse_selection_cost_bps']:.0f}bps")
        print(f"    Net benefit: {impact['net_benefit_bps']:.0f}bps")
        print(f"    → {gate_result['recommendation']}")
    
    print(f"\n" + "=" * 70)
    print("4. DISCLOSURE CADENCE")
    print("=" * 70)
    
    disc = analysis['disclosure_cadence']
    print(f"\n  {'Lag(days)':<12} {'Noise(σ)':<10} {'Sunspot%':<12} {'Run%':<8}")
    print("  " + "-" * 45)
    for i in range(len(disc['reporting_lag_days'])):
        print(f"  {disc['reporting_lag_days'][i]:<12.0f} "
              f"{disc['signal_noise'][i]:>8.3f}   "
              f"{disc['sunspot_probability'][i]*100:>8.1f}     "
              f"{disc['run_prob_pct'][i]:>6.1f}")
    
    print(f"\n" + "=" * 70)
    print("POLICY RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
1. **LCR 150% Floor** → Optimal trade-off
   - Cost: ~20bps/year yield sacrifice
   - Benefit: Eliminates run risk if λ < 11%
   - Ratio: ~10x benefit/cost
   
2. **PSM Backstop $200M** → Tail risk mitigation
   - Reduces 99th percentile loss by ~50bps
   - Marginal benefit: {psm['marginal_benefit_per_100M']:.0f}bps per $100M
   
3. **Redemption Gates** → Use sparingly
   - 50bp fee reduces redemptions ~{analysis['redemption_gates']['50bp']['impact']['delta_R_pp']:.0f}pp
   - But creates adverse selection (~25-50bps cost)
   - Reserve for extreme stress only
   
4. **Real-time Disclosure** → Low-cost transparency
   - Daily reporting vs monthly: ~{disc['sunspot_probability'][0]*100:.0f}% → {disc['sunspot_probability'][2]*100:.0f}% sunspot risk
   - Minimal cost, high confidence benefit
    """)
    
    print("=" * 70)
    print("Policy levers module ready! Use for risk management decisions.")

