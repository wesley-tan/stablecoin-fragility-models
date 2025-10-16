"""
Fire-Sale Value-at-Risk (FS-VaR) Module
=========================================

Monte Carlo simulation over (π, κ) distributions to compute:
- Fire-Sale VaR: loss distribution from forced liquidations
- Expected Shortfall (ES/CVaR)
- Stress test scenarios

Extends two-tier model to full risk distribution.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from run_cutoff import fire_sale_price as p_of_R, solve_n_star


def fire_sale_loss(R: float, lam: float, kappa: float, 
                   NAV: float = 1000.0) -> Dict:
    """
    Compute fire-sale loss for given redemption R.
    
    Loss = (face value - realized value) of T-bills sold
    
    Args:
        R: Total redemptions (fraction)
        lam: Tier-1 share
        kappa: Market impact
        NAV: Net asset value (millions, for scaling)
    
    Returns:
        Dictionary with loss metrics
    """
    if R <= lam:
        # No fire-sale needed
        return {
            'loss_millions': 0.0,
            'loss_bps': 0.0,
            'loss_pct_NAV': 0.0,
            'tbills_sold': 0.0,
            'fire_sale_price': 1.0,
            'depeg_bps': 0.0,
        }
    
    # Fire-sale needed
    p = p_of_R(R, lam, kappa)
    
    # Amount to raise: R - λ
    cash_needed = R - lam
    
    # T-bills sold: (R-λ)/p
    tbills_sold = cash_needed / p
    
    # Loss = face value - realized value
    #      = tbills_sold - cash_raised
    #      = (R-λ)/p - (R-λ)
    #      = (R-λ)·[1/p - 1]
    #      = (R-λ)·[1-p]/p
    
    loss_fraction = cash_needed * (1 - p) / p
    loss_millions = loss_fraction * NAV
    loss_bps = loss_fraction * 10000
    loss_pct_NAV = loss_fraction * 100
    
    depeg_bps = (1 - p) * 10000
    
    return {
        'loss_millions': loss_millions,
        'loss_bps': loss_bps,
        'loss_pct_NAV': loss_pct_NAV,
        'tbills_sold': tbills_sold,
        'fire_sale_price': p,
        'depeg_bps': depeg_bps,
        'R': R,
    }


def monte_carlo_FS_VaR(lam: float,
                      pi_dist: Tuple[str, Dict],
                      kappa_dist: Tuple[str, Dict],
                      n_sims: int = 10000,
                      NAV: float = 1000.0,
                      seed: Optional[int] = 42) -> Dict:
    """
    Monte Carlo simulation of fire-sale losses over (π, κ) distributions.
    
    Computes VaR and Expected Shortfall across scenarios.
    
    Args:
        lam: Tier-1 share (fixed)
        pi_dist: ('dist_name', {params}) for π distribution
                 e.g., ('normal', {'mean': 0.10, 'std': 0.03})
                       ('lognormal', {'mean': 0.10, 'std': 0.02})
                       ('uniform', {'low': 0.05, 'high': 0.15})
        kappa_dist: (dist_name, {params}) for κ distribution
        n_sims: Number of Monte Carlo draws
        NAV: Net asset value for scaling
        seed: Random seed
    
    Returns:
        Dictionary with VaR/ES metrics and loss distribution
    """
    np.random.seed(seed)
    
    # Draw samples
    pi_samples = _sample_distribution(pi_dist, n_sims)
    kappa_samples = _sample_distribution(kappa_dist, n_sims)
    
    # Clip to valid ranges
    pi_samples = np.clip(pi_samples, 0.001, 0.999)
    kappa_samples = np.clip(kappa_samples, 0.001, 0.999)
    
    # Compute losses for each scenario
    losses_millions = np.zeros(n_sims)
    losses_bps = np.zeros(n_sims)
    R_realized = np.zeros(n_sims)
    depeg_bps = np.zeros(n_sims)
    
    for i in range(n_sims):
        pi_i = pi_samples[i]
        kappa_i = kappa_samples[i]
        
        # Solve for equilibrium run fraction
        n_star, _ = solve_n_star(lam, pi_i, kappa_i)
        R_i = pi_i + n_star * (1 - pi_i)
        
        # Compute fire-sale loss
        loss_i = fire_sale_loss(R_i, lam, kappa_i, NAV)
        
        losses_millions[i] = loss_i['loss_millions']
        losses_bps[i] = loss_i['loss_bps']
        R_realized[i] = R_i
        depeg_bps[i] = loss_i['depeg_bps']
    
    # Compute VaR and ES at standard confidence levels
    confidence_levels = [0.90, 0.95, 0.99]
    var_metrics = {}
    
    for alpha in confidence_levels:
        # VaR: α-quantile of loss distribution
        var_bps = np.percentile(losses_bps, alpha * 100)
        var_millions = np.percentile(losses_millions, alpha * 100)
        
        # ES (CVaR): expected loss conditional on exceeding VaR
        tail_mask = losses_bps >= var_bps
        es_bps = np.mean(losses_bps[tail_mask]) if np.any(tail_mask) else var_bps
        es_millions = np.mean(losses_millions[tail_mask]) if np.any(tail_mask) else var_millions
        
        var_metrics[f'{int(alpha*100)}%'] = {
            'VaR_bps': var_bps,
            'VaR_millions': var_millions,
            'ES_bps': es_bps,
            'ES_millions': es_millions,
        }
    
    # Summary statistics
    mean_loss_bps = np.mean(losses_bps)
    median_loss_bps = np.median(losses_bps)
    std_loss_bps = np.std(losses_bps)
    max_loss_bps = np.max(losses_bps)
    
    # Probability of run (n*>0)
    run_prob = np.mean(R_realized > pi_samples + 0.001)
    
    return {
        'parameters': {
            'lambda': lam,
            'n_sims': n_sims,
            'NAV': NAV,
        },
        'distributions': {
            'pi': pi_dist,
            'kappa': kappa_dist,
        },
        'VaR_metrics': var_metrics,
        'summary_statistics': {
            'mean_loss_bps': mean_loss_bps,
            'median_loss_bps': median_loss_bps,
            'std_loss_bps': std_loss_bps,
            'max_loss_bps': max_loss_bps,
            'prob_run': run_prob,
        },
        'distributions_data': {
            'losses_millions': losses_millions,
            'losses_bps': losses_bps,
            'R_realized': R_realized,
            'depeg_bps': depeg_bps,
            'pi_samples': pi_samples,
            'kappa_samples': kappa_samples,
        }
    }


def _sample_distribution(dist_spec: Tuple[str, Dict], n: int) -> np.ndarray:
    """Helper: sample from specified distribution"""
    dist_name, params = dist_spec
    
    if dist_name == 'normal':
        return np.random.normal(params['mean'], params['std'], n)
    elif dist_name == 'lognormal':
        # Lognormal parameterization: mean and std of underlying normal
        mu = np.log(params['mean'])
        sigma = params['std'] / params['mean']  # CV approximation
        return np.random.lognormal(mu, sigma, n)
    elif dist_name == 'uniform':
        return np.random.uniform(params['low'], params['high'], n)
    elif dist_name == 'beta':
        return np.random.beta(params['alpha'], params['beta'], n)
    elif dist_name == 'constant':
        return np.full(n, params['value'])
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")


def FS_VaR_table(lam_grid: np.ndarray,
                pi_dist: Tuple[str, Dict],
                kappa_dist: Tuple[str, Dict],
                n_sims: int = 10000) -> Dict:
    """
    Generate FS-VaR table across reserve mixes.
    
    Key deliverable: table of VaR/ES by λ (reserve composition).
    
    Args:
        lam_grid: Array of Tier-1 shares to test
        pi_dist: Distribution for π
        kappa_dist: Distribution for κ
        n_sims: Monte Carlo simulations
    
    Returns:
        Dictionary with table data
    """
    results = {
        'lambda_pct': [],
        'tbill_pct': [],
        'mean_loss_bps': [],
        'VaR_95_bps': [],
        'ES_95_bps': [],
        'VaR_99_bps': [],
        'ES_99_bps': [],
        'prob_run_pct': [],
    }
    
    for lam in lam_grid:
        mc_result = monte_carlo_FS_VaR(lam, pi_dist, kappa_dist, n_sims)
        
        results['lambda_pct'].append(lam * 100)
        results['tbill_pct'].append((1 - lam) * 100)
        results['mean_loss_bps'].append(mc_result['summary_statistics']['mean_loss_bps'])
        results['VaR_95_bps'].append(mc_result['VaR_metrics']['95%']['VaR_bps'])
        results['ES_95_bps'].append(mc_result['VaR_metrics']['95%']['ES_bps'])
        results['VaR_99_bps'].append(mc_result['VaR_metrics']['99%']['VaR_bps'])
        results['ES_99_bps'].append(mc_result['VaR_metrics']['99%']['ES_bps'])
        results['prob_run_pct'].append(mc_result['summary_statistics']['prob_run'] * 100)
    
    return results


def stress_test_scenarios(lam: float, NAV: float = 1000.0) -> Dict:
    """
    Predefined stress scenarios (Basel-style).
    
    Args:
        lam: Tier-1 share
        NAV: Net asset value
    
    Returns:
        Dictionary with scenario results
    """
    scenarios = {
        'baseline': {'pi': 0.10, 'kappa': 0.10, 'name': 'Normal conditions'},
        'moderate_stress': {'pi': 0.15, 'kappa': 0.12, 'name': 'Moderate stress'},
        'severe_stress': {'pi': 0.20, 'kappa': 0.15, 'name': 'Severe stress (USDT May 2022)'},
        'extreme_stress': {'pi': 0.30, 'kappa': 0.20, 'name': 'Extreme stress (UST collapse)'},
        'svb_crisis': {'pi': 0.115, 'kappa': 0.10, 'name': 'USDC/SVB Crisis (Mar 2023)'},
    }
    
    results = {}
    
    for scenario_key, scenario in scenarios.items():
        pi = scenario['pi']
        kappa = scenario['kappa']
        
        # Solve equilibrium
        n_star, _ = solve_n_star(lam, pi, kappa)
        R = pi + n_star * (1 - pi)
        
        # Compute loss
        loss = fire_sale_loss(R, lam, kappa, NAV)
        
        results[scenario_key] = {
            'name': scenario['name'],
            'pi': pi,
            'kappa': kappa,
            'n_star': n_star,
            'R': R,
            'loss_bps': loss['loss_bps'],
            'loss_millions': loss['loss_millions'],
            'depeg_bps': loss['depeg_bps'],
            'vulnerable': n_star > 1e-4,
        }
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FIRE-SALE VALUE-AT-RISK (FS-VaR) MODULE")
    print("=" * 70)
    
    # Parameters
    lam = 0.20  # 20% liquid
    NAV = 1000.0  # $1B
    
    print(f"\n📊 PARAMETERS")
    print(f"  λ (Tier-1): {lam*100:.0f}%")
    print(f"  NAV: ${NAV:.0f}M")
    
    # Define distributions (calibrated to historical data)
    pi_dist = ('normal', {'mean': 0.10, 'std': 0.03})  # Redemption stress
    kappa_dist = ('uniform', {'low': 0.05, 'high': 0.15})  # Market impact uncertainty
    
    print(f"\n🎲 DISTRIBUTIONS")
    print(f"  π: Normal(mean=10%, std=3%)")
    print(f"  κ: Uniform(5%, 15%)")
    
    # Monte Carlo simulation
    print(f"\n🔍 RUNNING MONTE CARLO (10,000 simulations)...")
    mc_result = monte_carlo_FS_VaR(lam, pi_dist, kappa_dist, n_sims=10000)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    stats_dict = mc_result['summary_statistics']
    print(f"  Mean loss: {stats_dict['mean_loss_bps']:.1f} bps")
    print(f"  Median loss: {stats_dict['median_loss_bps']:.1f} bps")
    print(f"  Std dev: {stats_dict['std_loss_bps']:.1f} bps")
    print(f"  Max loss: {stats_dict['max_loss_bps']:.1f} bps")
    print(f"  Prob(run): {stats_dict['prob_run']*100:.1f}%")
    
    print(f"\n💥 VALUE-AT-RISK (VaR):")
    for conf_level, metrics in mc_result['VaR_metrics'].items():
        print(f"  {conf_level} confidence:")
        print(f"    VaR:  {metrics['VaR_bps']:>7.1f} bps (${metrics['VaR_millions']:>6.2f}M)")
        print(f"    ES:   {metrics['ES_bps']:>7.1f} bps (${metrics['ES_millions']:>6.2f}M)")
    
    # FS-VaR table across reserve mixes
    print(f"\n📋 FS-VaR TABLE ACROSS RESERVE MIXES:")
    lam_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    var_table = FS_VaR_table(lam_grid, pi_dist, kappa_dist, n_sims=5000)
    
    print(f"\n  {'λ%':<6} {'T-bill%':<8} {'Mean':<8} {'VaR95':<8} {'ES95':<8} {'VaR99':<8} {'Run%':<8}")
    print("  " + "-" * 62)
    for i in range(len(var_table['lambda_pct'])):
        print(f"  {var_table['lambda_pct'][i]:>4.0f}   "
              f"{var_table['tbill_pct'][i]:>6.0f}   "
              f"{var_table['mean_loss_bps'][i]:>6.0f}   "
              f"{var_table['VaR_95_bps'][i]:>6.0f}   "
              f"{var_table['ES_95_bps'][i]:>6.0f}   "
              f"{var_table['VaR_99_bps'][i]:>6.0f}   "
              f"{var_table['prob_run_pct'][i]:>6.1f}")
    
    # Stress scenarios
    print(f"\n🔥 STRESS TEST SCENARIOS:")
    scenarios = stress_test_scenarios(lam, NAV)
    
    print(f"\n  {'Scenario':<25} {'π%':<6} {'κ%':<6} {'n*%':<7} {'Loss(bps)':<10} {'Depeg':<8}")
    print("  " + "-" * 70)
    for key, scenario in scenarios.items():
        print(f"  {scenario['name']:<25} "
              f"{scenario['pi']*100:>4.1f}  "
              f"{scenario['kappa']*100:>4.0f}  "
              f"{scenario['n_star']*100:>5.1f}  "
              f"{scenario['loss_bps']:>8.1f}    "
              f"{scenario['depeg_bps']:>6.1f}bp")
    
    print("\n" + "=" * 70)
    print("FS-VaR module ready! Use for risk reporting and stress tests.")

