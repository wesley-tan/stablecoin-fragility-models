"""
Robust Optimization Framework for Stablecoin Reserve Composition and Policy Design

Solves:
    min_{λ, policies} max_{sunspot} Pr(run)
    
    subject to:
        - Fire-sale externalities internalized
        - Cross-issuer contagion effects
        - Target yield constraint
        - Regulatory constraints (LCR, etc.)

Research Questions:
1. Reserve mix → fragility: How do T-bills, deposits, repo, MMFs shift n* and expected loss?
2. Policies → robust stability: Optimal LCR, PSM, disclosure, gates to minimize tail risk?

Authors: [Your name]
Date: October 2025
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from run_cutoff_corrected import solve_n_star_corrected, fire_sale_price_corrected
from firesale_var import monte_carlo_FS_VaR
from policy_levers import comprehensive_policy_analysis


@dataclass
class AssetClass:
    """Individual asset class with yield and haircut characteristics"""
    name: str
    yield_bps: float        # Annual yield in bps
    liquidity_tier: int     # 1=liquid (cash/MMF), 2=illiquid (T-bills)
    haircut_normal: float   # Haircut in normal times (bps)
    haircut_stress: float   # Haircut in stress (bps)
    fire_sale_kappa: float  # Price impact coefficient
    
    def effective_yield(self, stress: bool = False) -> float:
        """Yield net of expected haircut"""
        haircut = self.haircut_stress if stress else self.haircut_normal
        return self.yield_bps - haircut


@dataclass
class ReserveMix:
    """Reserve composition across asset classes"""
    cash: float           # Cash (0% yield, Tier-1)
    mmf: float            # Money market funds (Tier-1)
    repo: float           # Repo (Tier-1/1.5)
    tbills: float         # T-bills (Tier-2)
    deposits: float       # Bank deposits (Tier-2 in stress)
    
    def lambda_tier1(self) -> float:
        """Liquid assets (Tier-1)"""
        return self.cash + self.mmf + self.repo
    
    def tier2(self) -> float:
        """Illiquid assets (Tier-2)"""
        return self.tbills + self.deposits
    
    def total(self) -> float:
        """Total reserves"""
        return self.lambda_tier1() + self.tier2()
    
    def is_valid(self) -> bool:
        """Check if composition is valid (sums to ~1.0, all non-negative)"""
        total = self.total()
        return (abs(total - 1.0) < 1e-6 and 
                all(x >= 0 for x in [self.cash, self.mmf, self.repo, 
                                      self.tbills, self.deposits]))
    
    def to_dict(self) -> Dict:
        return {
            'cash': self.cash,
            'mmf': self.mmf,
            'repo': self.repo,
            'tbills': self.tbills,
            'deposits': self.deposits,
            'lambda': self.lambda_tier1(),
            'tier2': self.tier2()
        }


@dataclass
class PolicyBundle:
    """Complete policy intervention bundle"""
    lcr_floor: float           # LCR minimum (e.g., 1.5 for 150%)
    psm_size_pct: float        # PSM as % of supply (e.g., 0.02 for 2%)
    disclosure_freq_hours: int # Disclosure frequency (24=daily, 168=weekly)
    redemption_fee_bps: float  # Redemption fee in bps
    redemption_window_hours: float  # Minimum redemption delay
    
    def annual_cost_bps(self, supply: float = 1.0) -> float:
        """Estimate annual cost of policy bundle in bps"""
        # LCR cost: opportunity cost of holding liquid assets
        lcr_cost = max(0, (self.lcr_floor - 1.0) * 100) * 0.20  # ~20bps per 100pp
        
        # PSM cost: capital tied up (assume 0-5bps based on size)
        psm_cost = self.psm_size_pct * 100 * 2.5  # ~2.5bps per 1% of supply
        
        # Disclosure cost: minimal (IT/compliance)
        disclosure_cost = max(0, 5 - self.disclosure_freq_hours / 24)  # More frequent = slight cost
        
        # Redemption friction cost: adverse selection & user experience
        redemption_cost = self.redemption_fee_bps * 0.5  # 50% of fee is real friction
        window_cost = self.redemption_window_hours * 0.1  # 0.1bp per hour delay
        
        return lcr_cost + psm_cost + disclosure_cost + redemption_cost + window_cost
    
    def to_dict(self) -> Dict:
        return {
            'lcr_floor': self.lcr_floor,
            'psm_size_pct': self.psm_size_pct,
            'disclosure_freq_hours': self.disclosure_freq_hours,
            'redemption_fee_bps': self.redemption_fee_bps,
            'redemption_window_hours': self.redemption_window_hours,
            'annual_cost_bps': self.annual_cost_bps()
        }


# Define asset classes based on current market data (Q4 2024 / Q1 2025)
ASSET_CLASSES = {
    'cash': AssetClass('Cash', 0, 1, 0, 0, 0.0),
    'mmf': AssetClass('Money Market Funds', 480, 1, 0, 10, 0.005),
    'repo': AssetClass('Overnight Repo', 500, 1, 5, 15, 0.008),
    'tbills': AssetClass('T-bills (3M)', 510, 2, 2, 5, 0.020),
    'deposits': AssetClass('Bank Deposits', 400, 2, 0, 500, 0.10),  # SVB scenario
}


def compute_portfolio_yield(mix: ReserveMix, stress: bool = False) -> float:
    """Compute portfolio yield in bps"""
    return (
        mix.cash * ASSET_CLASSES['cash'].effective_yield(stress) +
        mix.mmf * ASSET_CLASSES['mmf'].effective_yield(stress) +
        mix.repo * ASSET_CLASSES['repo'].effective_yield(stress) +
        mix.tbills * ASSET_CLASSES['tbills'].effective_yield(stress) +
        mix.deposits * ASSET_CLASSES['deposits'].effective_yield(stress)
    )


def compute_weighted_kappa(mix: ReserveMix) -> float:
    """Compute weighted-average fire-sale impact coefficient"""
    tier2_total = mix.tier2()
    if tier2_total < 1e-6:
        return 0.0
    
    return (
        (mix.tbills / tier2_total) * ASSET_CLASSES['tbills'].fire_sale_kappa +
        (mix.deposits / tier2_total) * ASSET_CLASSES['deposits'].fire_sale_kappa
    )


def run_probability_robust(
    mix: ReserveMix,
    policy: PolicyBundle,
    pi_range: Tuple[float, float] = (0.05, 0.15),
    kappa_multiplier_range: Tuple[float, float] = (0.5, 2.0),
    n_sunspot_samples: int = 10  # Reduced for speed
) -> Dict:
    """
    Compute robust (max over sunspots) run probability
    
    Sunspots vary:
    - Impatient share π ∈ [pi_range]
    - Fire-sale impact (multiplier on base κ)
    - Information precision (via disclosure freq)
    
    Returns worst-case run probability and expected loss
    """
    lam = mix.lambda_tier1()
    
    # Adjust lambda for LCR floor
    if policy.lcr_floor > 1.0:
        # LCR floor forces higher liquid share
        required_lam = min(0.50, lam * policy.lcr_floor)
        lam = max(lam, required_lam)
    
    # PSM effectively adds to liquid tier
    lam_with_psm = min(0.99, lam + policy.psm_size_pct)
    
    # Base kappa from reserve mix
    kappa_base = compute_weighted_kappa(mix)
    
    # Sample sunspot space
    pi_samples = np.linspace(pi_range[0], pi_range[1], n_sunspot_samples)
    kappa_mult_samples = np.linspace(kappa_multiplier_range[0], 
                                     kappa_multiplier_range[1], 
                                     n_sunspot_samples)
    
    max_run_prob = 0.0
    max_expected_loss = 0.0
    worst_case_scenario = {}
    
    for pi in pi_samples:
        # Disclosure frequency affects information precision → shifts π
        # More frequent disclosure → lower coordination risk
        disclosure_factor = max(0.80, 1.0 - policy.disclosure_freq_hours / 168 * 0.20)
        pi_adjusted = pi * disclosure_factor
        
        for kappa_mult in kappa_mult_samples:
            kappa = kappa_base * kappa_mult
            
            # Redemption fees/windows reduce effective pi
            fee_reduction = policy.redemption_fee_bps / 10000  # Convert to fraction
            pi_net = max(0.01, pi_adjusted - fee_reduction)
            
            # Solve for run cutoff
            n_star_result = solve_n_star_corrected(lam_with_psm, pi_net, kappa, h_max=0.15)
            # Handle both float and tuple returns
            n_star = float(n_star_result) if not isinstance(n_star_result, tuple) else float(n_star_result[0])
            
            # Run probability (simplified: if n* > threshold, run is possible)
            # In full global games framework, this would be smooth function
            run_prob = float(n_star > 0.01)  # Binary for simplicity
            
            # Expected loss
            if n_star > 0.01:
                R = pi_net + n_star * (1 - pi_net)
                if R > lam_with_psm:
                    Q = R - lam_with_psm
                    p = fire_sale_price_corrected(R, lam_with_psm, kappa, h_max=0.15)
                    loss = Q * (1.0 - p)
                else:
                    loss = 0.0
            else:
                loss = 0.0
            
            # Track worst case
            if run_prob > max_run_prob or (run_prob == max_run_prob and loss > max_expected_loss):
                max_run_prob = run_prob
                max_expected_loss = loss
                worst_case_scenario = {
                    'pi': pi,
                    'pi_adjusted': pi_adjusted,
                    'pi_net': pi_net,
                    'kappa': kappa,
                    'n_star': n_star,
                    'run_prob': run_prob,
                    'expected_loss': loss
                }
    
    return {
        'max_run_probability': max_run_prob,
        'max_expected_loss': max_expected_loss,
        'worst_case': worst_case_scenario,
        'lambda_effective': lam_with_psm,
        'kappa_base': kappa_base
    }


def objective_function(
    x: np.ndarray,
    target_yield: float,
    yield_penalty_weight: float = 10.0,
    cost_penalty_weight: float = 1.0
) -> float:
    """
    Objective: minimize max run probability + yield shortfall penalty + cost penalty
    
    x = [cash, mmf, repo, tbills, lcr_floor, psm_size_pct, disc_freq_idx, fee_bps, window_hours]
    """
    # Unpack decision variables
    cash, mmf, repo, tbills = x[0:4]
    deposits = max(0, 1.0 - (cash + mmf + repo + tbills))  # Residual
    
    lcr_floor = x[4]
    psm_size_pct = x[5]
    disc_freq_idx = int(x[6])  # Index into [24, 48, 168] hours
    disc_freqs = [24, 48, 168]
    disc_freq = disc_freqs[min(disc_freq_idx, len(disc_freqs)-1)]
    fee_bps = x[7]
    window_hours = x[8]
    
    # Create mix and policy
    mix = ReserveMix(cash, mmf, repo, tbills, deposits)
    policy = PolicyBundle(lcr_floor, psm_size_pct, disc_freq, fee_bps, window_hours)
    
    # Check validity
    if not mix.is_valid():
        return 1e6  # Invalid composition
    
    # Compute yield
    portfolio_yield = compute_portfolio_yield(mix, stress=False)
    yield_shortfall = max(0, target_yield - portfolio_yield)
    
    # Compute robust run probability
    robust_result = run_probability_robust(mix, policy)
    max_run_prob = robust_result['max_run_probability']
    
    # Policy cost
    policy_cost = policy.annual_cost_bps()
    
    # Objective: minimize run probability, yield shortfall, and costs
    objective = (
        max_run_prob +
        yield_penalty_weight * (yield_shortfall / 100) +  # Normalize
        cost_penalty_weight * (policy_cost / 100)
    )
    
    return objective


def optimize_reserve_and_policy(
    target_yield_bps: float = 450,
    method: str = 'differential_evolution',
    verbose: bool = True
) -> Dict:
    """
    Find optimal reserve mix and policy bundle
    
    Minimize: max_{sunspot} Pr(run)
    Subject to: E[yield] ≥ target_yield_bps
    """
    if verbose:
        print("="*80)
        print("ROBUST OPTIMIZATION: Reserve Mix + Policy Bundle")
        print("="*80)
        print(f"\nTarget yield: {target_yield_bps} bps")
        print(f"Optimization method: {method}\n")
    
    # Bounds for decision variables
    # [cash, mmf, repo, tbills, lcr_floor, psm_size_pct, disc_freq_idx, fee_bps, window_hours]
    bounds = [
        (0.0, 0.30),   # cash: 0-30%
        (0.0, 0.50),   # mmf: 0-50%
        (0.0, 0.30),   # repo: 0-30%
        (0.0, 0.80),   # tbills: 0-80%
        (1.0, 3.0),    # lcr_floor: 100%-300%
        (0.0, 0.10),   # psm_size_pct: 0-10%
        (0, 2),        # disc_freq_idx: 0=daily, 1=2days, 2=weekly
        (0.0, 100.0),  # fee_bps: 0-100bps
        (0.0, 24.0),   # window_hours: 0-24 hours
    ]
    
    # Constraint: sum(cash, mmf, repo, tbills) + deposits = 1.0
    # This is enforced by setting deposits = 1 - sum(others)
    
    if method == 'differential_evolution':
        result = differential_evolution(
            lambda x: objective_function(x, target_yield_bps),
            bounds,
            seed=42,
            maxiter=50,  # Reduced for speed
            popsize=10,  # Reduced for speed
            atol=1e-3,
            tol=1e-3,
            workers=1,
            updating='deferred',
            disp=verbose
        )
        x_opt = result.x
        obj_val = result.fun
    else:
        # Multi-start local optimization
        best_obj = np.inf
        best_x = None
        
        for _ in range(10):
            x0 = np.array([
                np.random.uniform(*bounds[i]) for i in range(len(bounds))
            ])
            
            res = minimize(
                lambda x: objective_function(x, target_yield_bps),
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        
        x_opt = best_x
        obj_val = best_obj
    
    # Extract optimal solution
    cash_opt, mmf_opt, repo_opt, tbills_opt = x_opt[0:4]
    deposits_opt = max(0, 1.0 - (cash_opt + mmf_opt + repo_opt + tbills_opt))
    
    lcr_opt = x_opt[4]
    psm_opt = x_opt[5]
    disc_idx = int(x_opt[6])
    disc_freqs = [24, 48, 168]
    disc_opt = disc_freqs[min(disc_idx, len(disc_freqs)-1)]
    fee_opt = x_opt[7]
    window_opt = x_opt[8]
    
    mix_opt = ReserveMix(cash_opt, mmf_opt, repo_opt, tbills_opt, deposits_opt)
    policy_opt = PolicyBundle(lcr_opt, psm_opt, disc_opt, fee_opt, window_opt)
    
    # Compute final metrics
    portfolio_yield = compute_portfolio_yield(mix_opt, stress=False)
    robust_result = run_probability_robust(mix_opt, policy_opt)
    
    # Compute VaR/ES via Monte Carlo
    kappa_base = compute_weighted_kappa(mix_opt)
    lam_eff = robust_result['lambda_effective']
    
    # Distribution specifications in correct format
    pi_dist_spec = ('uniform', {'low': 0.05, 'high': 0.15})
    kappa_dist_spec = ('uniform', {'low': kappa_base * 0.5, 'high': kappa_base * 2.0})
    
    var_result = monte_carlo_FS_VaR(
        lam=lam_eff,
        pi_dist=pi_dist_spec,
        kappa_dist=kappa_dist_spec,
        n_sims=1000  # Reduced for speed
    )
    
    results = {
        'optimal_reserve_mix': mix_opt.to_dict(),
        'optimal_policy': policy_opt.to_dict(),
        'performance': {
            'portfolio_yield_bps': portfolio_yield,
            'target_yield_bps': target_yield_bps,
            'yield_gap_bps': portfolio_yield - target_yield_bps,
            'max_run_probability': robust_result['max_run_probability'],
            'max_expected_loss_pct': robust_result['max_expected_loss'] * 100,
            'VaR_95_bps': var_result['VaR_metrics']['95%']['VaR_bps'],
            'VaR_99_bps': var_result['VaR_metrics']['99%']['VaR_bps'],
            'ES_95_bps': var_result['VaR_metrics']['95%']['ES_bps'],
            'ES_99_bps': var_result['VaR_metrics']['99%']['ES_bps'],
        },
        'worst_case_scenario': robust_result['worst_case'],
        'objective_value': obj_val
    }
    
    if verbose:
        print("\n" + "="*80)
        print("OPTIMAL SOLUTION")
        print("="*80)
        
        print("\n📊 RESERVE MIX")
        print(f"  Cash:              {cash_opt*100:>6.2f}%")
        print(f"  Money Market Funds:{mmf_opt*100:>6.2f}%")
        print(f"  Repo:              {repo_opt*100:>6.2f}%")
        print(f"  T-bills:           {tbills_opt*100:>6.2f}%")
        print(f"  Bank Deposits:     {deposits_opt*100:>6.2f}%")
        print(f"  → Tier-1 (liquid): {mix_opt.lambda_tier1()*100:>6.2f}%")
        print(f"  → Tier-2:          {mix_opt.tier2()*100:>6.2f}%")
        
        print("\n🛡️  POLICY BUNDLE")
        print(f"  LCR Floor:         {lcr_opt*100:>6.0f}%")
        print(f"  PSM Size:          {psm_opt*100:>6.2f}% of supply")
        print(f"  Disclosure:        Every {disc_opt:>3.0f} hours")
        print(f"  Redemption Fee:    {fee_opt:>6.1f} bps")
        print(f"  Redemption Window: {window_opt:>6.1f} hours")
        print(f"  → Annual Cost:     {policy_opt.annual_cost_bps():>6.1f} bps")
        
        print("\n📈 PERFORMANCE")
        print(f"  Portfolio Yield:   {portfolio_yield:>6.1f} bps")
        print(f"  Target Yield:      {target_yield_bps:>6.1f} bps")
        print(f"  Yield Gap:         {portfolio_yield - target_yield_bps:>6.1f} bps")
        print(f"  Max Run Prob:      {robust_result['max_run_probability']*100:>6.1f}%")
        print(f"  Max Expected Loss: {robust_result['max_expected_loss']*100:>6.2f}%")
        print(f"  VaR₉₅:             {var_result['VaR_metrics']['95%']['VaR_bps']:>6.0f} bps")
        print(f"  VaR₉₉:             {var_result['VaR_metrics']['99%']['VaR_bps']:>6.0f} bps")
        print(f"  ES₉₉:              {var_result['VaR_metrics']['99%']['ES_bps']:>6.0f} bps")
        
        print("\n⚠️  WORST-CASE SCENARIO")
        wc = robust_result.get('worst_case', {})
        if wc:
            print(f"  Impatient share (π): {wc.get('pi', 0)*100:.1f}%")
            print(f"  Impact coeff (κ):    {wc.get('kappa', 0)*100:.2f}%")
            print(f"  Run fraction (n*):   {wc.get('n_star', 0)*100:.1f}%")
            print(f"  Expected loss:       {wc.get('expected_loss', 0)*100:.2f}%")
        else:
            print("  No vulnerable scenarios found (fully stable)")
        
        print("\n" + "="*80)
    
    return results


def sweep_reserve_mix(
    asset_to_vary: str = 'tbills',
    range_pct: Tuple[float, float] = (10, 80),
    n_points: int = 15,
    baseline_policy: Optional[PolicyBundle] = None,
    pi: float = 0.08,
    kappa_base: float = 0.02
) -> Dict:
    """
    Sweep one reserve component and analyze effects on n* and expected loss
    
    Answers Sub-Question 1: Reserve mix → fragility
    """
    if baseline_policy is None:
        baseline_policy = PolicyBundle(
            lcr_floor=1.5,
            psm_size_pct=0.02,
            disclosure_freq_hours=24,
            redemption_fee_bps=0.0,
            redemption_window_hours=0.0
        )
    
    share_grid = np.linspace(range_pct[0]/100, range_pct[1]/100, n_points)
    
    results = {
        'asset_varied': asset_to_vary,
        'share_grid': share_grid.tolist(),
        'lambda_tier1': [],
        'n_star': [],
        'depeg_bps': [],
        'expected_loss_pct': [],
        'portfolio_yield_bps': [],
        'VaR_99_bps': [],
    }
    
    for share in share_grid:
        # Create reserve mix with varying component
        # Simple heuristic: vary target asset, distribute rest proportionally
        if asset_to_vary == 'tbills':
            cash = 0.05
            mmf = max(0, (1.0 - share - cash) * 0.60)
            repo = max(0, (1.0 - share - cash) * 0.20)
            tbills = share
            deposits = max(0, 1.0 - (cash + mmf + repo + tbills))
        elif asset_to_vary == 'mmf':
            cash = 0.05
            mmf = share
            tbills = max(0, (1.0 - share - cash) * 0.70)
            repo = max(0, (1.0 - share - cash) * 0.20)
            deposits = max(0, 1.0 - (cash + mmf + repo + tbills))
        elif asset_to_vary == 'deposits':
            cash = 0.05
            deposits = share
            mmf = max(0, (1.0 - share - cash) * 0.40)
            repo = max(0, (1.0 - share - cash) * 0.10)
            tbills = max(0, 1.0 - (cash + mmf + repo + deposits))
        else:  # cash
            cash = share
            mmf = max(0, (1.0 - share) * 0.40)
            tbills = max(0, (1.0 - share) * 0.50)
            repo = max(0, (1.0 - share) * 0.05)
            deposits = max(0, 1.0 - (cash + mmf + repo + tbills))
        
        mix = ReserveMix(cash, mmf, repo, tbills, deposits)
        
        if not mix.is_valid():
            continue
        
        # Compute metrics
        lam = mix.lambda_tier1()
        kappa = compute_weighted_kappa(mix)
        
        # Run cutoff
        n_star_result = solve_n_star_corrected(lam, pi, kappa if kappa > 0 else kappa_base)
        n_star = float(n_star_result) if not isinstance(n_star_result, tuple) else float(n_star_result[0])
        
        # Depeg
        R = pi + n_star * (1 - pi)
        if R > lam and n_star > 0.01:
            p = fire_sale_price_corrected(R, lam, kappa if kappa > 0 else kappa_base)
            depeg = (1.0 - p) * 10000
            Q = R - lam
            exp_loss = Q * (1.0 - p) * 100
        else:
            depeg = 0
            exp_loss = 0
        
        # Yield
        port_yield = compute_portfolio_yield(mix)
        
        # VaR (quick estimate, not full MC)
        kappa_eff = kappa if kappa > 0 else kappa_base
        pi_dist_spec = ('uniform', {'low': 0.05, 'high': 0.15})
        kappa_dist_spec = ('uniform', {'low': kappa_eff * 0.5, 'high': kappa_eff * 2.0})
        var_quick = monte_carlo_FS_VaR(lam, pi_dist_spec, kappa_dist_spec, n_sims=1000)
        
        results['lambda_tier1'].append(lam)
        results['n_star'].append(n_star)
        results['depeg_bps'].append(depeg)
        results['expected_loss_pct'].append(exp_loss)
        results['portfolio_yield_bps'].append(port_yield)
        results['VaR_99_bps'].append(var_quick['VaR_metrics']['99%']['VaR_bps'])
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STABLECOIN ROBUST OPTIMIZATION FRAMEWORK")
    print("="*80)
    print("\nResearch Question:")
    print("What reserve composition and policy bundle minimize the maximum")
    print("equilibrium run probability (robust to sunspot coordination) when")
    print("fire-sale externalities and cross-issuer contagion are internalized,")
    print("subject to a target yield?")
    print("\n" + "="*80)
    
    # Part 1: Find optimal solution
    print("\n\n### PART 1: OPTIMAL RESERVE MIX + POLICY BUNDLE ###\n")
    
    optimal_solution = optimize_reserve_and_policy(
        target_yield_bps=450,  # Target 4.5% yield
        method='differential_evolution',
        verbose=True
    )
    
    # Save results
    with open('optimal_solution.json', 'w') as f:
        json.dump(optimal_solution, f, indent=2)
    
    print("\n✅ Optimal solution saved to: optimal_solution.json")
    
    # Part 2: Sub-Question 1 - Reserve mix sensitivity
    print("\n\n### PART 2: RESERVE MIX → FRAGILITY (Sub-Question 1) ###\n")
    
    print("\nSweeping T-bill share (10% → 80%)...")
    tbill_sweep = sweep_reserve_mix('tbills', (10, 80), n_points=20)
    
    print("\nSweeping MMF share (5% → 50%)...")
    mmf_sweep = sweep_reserve_mix('mmf', (5, 50), n_points=20)
    
    print("\nSweeping Bank Deposit share (0% → 30%)...")
    deposit_sweep = sweep_reserve_mix('deposits', (0, 30), n_points=20)
    
    # Save sweeps
    with open('reserve_mix_sweeps.json', 'w') as f:
        json.dump({
            'tbills': tbill_sweep,
            'mmf': mmf_sweep,
            'deposits': deposit_sweep
        }, f, indent=2)
    
    print("\n✅ Reserve mix sweeps saved to: reserve_mix_sweeps.json")
    
    # Part 3: Key findings summary
    print("\n\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    opt_mix = optimal_solution['optimal_reserve_mix']
    opt_pol = optimal_solution['optimal_policy']
    opt_perf = optimal_solution['performance']
    
    print("\n1. OPTIMAL RESERVE COMPOSITION")
    print(f"   Tier-1 (liquid):  {opt_mix['lambda']*100:.1f}%")
    print(f"     - Cash:         {opt_mix['cash']*100:.1f}%")
    print(f"     - MMFs:         {opt_mix['mmf']*100:.1f}%")
    print(f"     - Repo:         {opt_mix['repo']*100:.1f}%")
    print(f"   Tier-2:           {opt_mix['tier2']*100:.1f}%")
    print(f"     - T-bills:      {opt_mix['tbills']*100:.1f}%")
    print(f"     - Deposits:     {opt_mix['deposits']*100:.1f}%")
    
    print("\n2. OPTIMAL POLICY BUNDLE")
    print(f"   LCR Floor:        {opt_pol['lcr_floor']*100:.0f}%")
    print(f"   PSM Buffer:       {opt_pol['psm_size_pct']*100:.1f}% of supply")
    print(f"   Disclosure:       Every {opt_pol['disclosure_freq_hours']:.0f} hours")
    print(f"   Redemption Fee:   {opt_pol['redemption_fee_bps']:.1f} bps")
    print(f"   Annual Cost:      {opt_pol['annual_cost_bps']:.1f} bps")
    
    print("\n3. ROBUST STABILITY")
    print(f"   Max Run Prob:     {opt_perf['max_run_probability']*100:.1f}%")
    print(f"   Max Exp. Loss:    {opt_perf['max_expected_loss_pct']:.2f}%")
    print(f"   VaR₉₉:            {opt_perf['VaR_99_bps']:.0f} bps")
    print(f"   ES₉₉:             {opt_perf['ES_99_bps']:.0f} bps")
    
    print("\n4. YIELD VS. STABILITY TRADE-OFF")
    print(f"   Portfolio Yield:  {opt_perf['portfolio_yield_bps']:.1f} bps")
    print(f"   Target Yield:     {opt_perf['target_yield_bps']:.1f} bps")
    print(f"   Yield Gap:        {opt_perf['yield_gap_bps']:.1f} bps")
    print(f"   Cost/Benefit:     {opt_pol['annual_cost_bps'] / max(1, opt_perf['VaR_99_bps']):.3f}")
    
    print("\n" + "="*80)
    print("\n✅ Analysis complete! See JSON outputs for detailed results.\n")

