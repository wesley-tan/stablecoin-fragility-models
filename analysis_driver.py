"""
Analysis Driver: Run Cutoff vs T-bill Share
============================================

Wires the two-tier liquidity ladder model to CalibrationEngine.

Tier-1 (at par): Cash + MMFs
Tier-2 (fire-sale): T-bills

Key output: Run cutoff (n*) vs T-bill share figure for paper.
"""

import numpy as np
import json
from typing import Dict, Tuple
from dataclasses import asdict

from run_cutoff import (
    run_cutoff_curve,
    run_cutoff_sensitivity,
    expected_loss_at_shock,
    solve_n_star,
)
from calibration import CalibrationEngine, create_scenario_variants
from stablecoin_fragility import StablecoinParameters, AssetClass


def extract_two_tier_params(params: StablecoinParameters) -> Tuple[float, float, float]:
    """
    Extract (λ, π, κ) from StablecoinParameters for two-tier model.
    
    Tier-1 (liquid, at par): Cash + MMFs
    Tier-2 (illiquid, fire-sale): T-bills + Bank Deposits + Repo
    
    Args:
        params: StablecoinParameters from calibration
    
    Returns:
        (lambda, pi, kappa) tuple
        - lambda: Tier-1 share (cash/MMFs only)
        - pi: Impatient fraction (redemption shock)
        - kappa: Market impact coefficient
    """
    total_reserves = params.total_reserves
    
    # Tier-1: Only truly liquid assets (MMFs, and cash if modeled)
    # Conservative: treat only MMFs as Tier-1
    # Bank deposits are NOT Tier-1 (SVB taught us this)
    tier1_assets = [
        asset for asset in params.reserves
        if asset.asset_class == AssetClass.MMF
    ]
    
    tier1_amount = sum(a.amount for a in tier1_assets)
    lambda_tier1 = tier1_amount / total_reserves if total_reserves > 0 else 0.0
    
    # π: impatient fraction (use calibrated redemption shock)
    pi = params.redemption_shock
    
    # κ: market impact coefficient (already calibrated)
    kappa = params.market_impact_coeff
    
    # Report composition
    tier2_tbills = sum(a.amount for a in params.reserves if a.asset_class == AssetClass.TBILLS)
    tier2_deposits = sum(a.amount for a in params.reserves if a.asset_class == AssetClass.DEPOSITS)
    tier2_repo = sum(a.amount for a in params.reserves if a.asset_class == AssetClass.REPO)
    tier2_total = tier2_tbills + tier2_deposits + tier2_repo
    
    print(f"\n📊 TWO-TIER CLASSIFICATION:")
    print(f"  Tier-1 (at par): ${tier1_amount:.1f}M ({lambda_tier1*100:.1f}%)")
    print(f"    - MMFs: ${tier1_amount:.1f}M")
    print(f"  Tier-2 (fire-sale): ${tier2_total:.1f}M ({(1-lambda_tier1)*100:.1f}%)")
    print(f"    - T-bills: ${tier2_tbills:.1f}M ({tier2_tbills/total_reserves*100:.1f}%)")
    print(f"    - Deposits: ${tier2_deposits:.1f}M ({tier2_deposits/total_reserves*100:.1f}%)")
    print(f"    - Repo: ${tier2_repo:.1f}M ({tier2_repo/total_reserves*100:.1f}%)")
    
    return lambda_tier1, pi, kappa


def analyze_episode(episode_key: str = 'usdc_svb_2023') -> Dict:
    """
    Run full two-tier analysis for a historical episode.
    
    Args:
        episode_key: Key from HISTORICAL_EPISODES
    
    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print(f"RUN CUTOFF ANALYSIS: {episode_key.upper()}")
    print("=" * 70)
    
    # Load calibration
    calibrator = CalibrationEngine()
    params = calibrator.calibrate_to_episode(episode_key)
    episode = calibrator.episodes[episode_key]
    
    print(f"\n📚 EPISODE: {episode.name}")
    print(f"  Date: {episode.date.strftime('%Y-%m-%d')}")
    print(f"  Supply: ${episode.total_supply_billions:.1f}B")
    print(f"  Peak Redemption: {episode.peak_redemption_rate_pct:.1f}%")
    print(f"  Max Depeg: {episode.max_discount_pct:.1f}%")
    
    # Extract two-tier parameters
    lam, pi, kappa = extract_two_tier_params(params)
    
    print(f"\n🎯 MODEL PARAMETERS:")
    print(f"  λ (Tier-1 share): {lam*100:.1f}%")
    print(f"  π (impatient): {pi*100:.1f}%")
    print(f"  κ (impact): {kappa*100:.1f}%")
    
    # Compute run cutoff curve
    print(f"\n🔍 COMPUTING RUN CUTOFF CURVE...")
    result = run_cutoff_curve(pi, kappa)
    
    if not np.isnan(result.lambda_cutoff):
        print(f"✓ λ̂ (run cutoff): {result.lambda_cutoff*100:.1f}% Tier-1")
        print(f"  → T-bill share at cutoff: {(1-result.lambda_cutoff)*100:.1f}%")
    else:
        print(f"⚠ Always vulnerable (no stable equilibrium found)")
    
    # Find current position
    idx_current = np.argmin(np.abs(result.lambda_grid - lam))
    n_current = result.n_star[idx_current]
    vuln_current = result.vulnerable[idx_current]
    
    print(f"\n📍 CURRENT POSITION (λ={lam*100:.1f}%):")
    print(f"  n* (run fraction): {n_current*100:.1f}%")
    print(f"  Status: {'🔴 VULNERABLE' if vuln_current else '🟢 STABLE'}")
    
    # Sensitivity analysis
    print(f"\n🔬 SENSITIVITY ANALYSIS:")
    sens = run_cutoff_sensitivity(pi, kappa, lam, delta_lam=0.10)
    
    print(f"  Baseline (λ={lam*100:.1f}%): n*={sens['baseline']['n_star']*100:.2f}%")
    print(f"  +10pp Tier-1: n*={sens['lambda_up']['n_star']*100:.2f}% (Δn={sens['lambda_up']['delta_n']*100:.2f}pp)")
    print(f"  -10pp Tier-1: n*={sens['lambda_down']['n_star']*100:.2f}% (Δn={sens['lambda_down']['delta_n']*100:.2f}pp)")
    print(f"  Total swing: {sens['sensitivity']['delta_n_total']*100:.2f}pp")
    
    # Gate evaluation
    gate = sens['gate_evaluation']
    print(f"\n🚦 GATE EVALUATION:")
    print(f"  Sensitivity: {sens['sensitivity']['delta_n_total']*100:.2f}pp change over ±10pp λ shift")
    print(f"  Criterion: >0.5pp change → PROCEED")
    print(f"  Decision: {'✅ PROCEED' if gate['proceed'] else '❌ REVISIT'}")
    print(f"  {gate['reason']}")
    
    # Expected loss table
    print(f"\n💥 EXPECTED LOSS (at R={pi*100:.1f}% shock):")
    print(f"  {'λ (Tier-1)':<15} {'T-bills':<12} {'n*':<10} {'Loss (bps)':<12}")
    print("  " + "-" * 50)
    
    loss_table = []
    for lam_val in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        n_star, _ = solve_n_star(lam_val, pi, kappa)
        loss = expected_loss_at_shock(lam_val, pi, kappa, shock_R=pi)
        tbill_pct = (1 - lam_val) * 100
        
        print(f"  {lam_val*100:>5.0f}%          {tbill_pct:>5.0f}%      {n_star*100:>5.1f}%    {loss['loss_bps']:>8.1f}")
        
        loss_table.append({
            'lambda': lam_val,
            'tbill_share': tbill_pct,
            'n_star': n_star,
            'loss_bps': loss['loss_bps'],
        })
    
    # Package results
    results = {
        'episode': {
            'name': episode.name,
            'key': episode_key,
            'date': episode.date.isoformat(),
            'supply_billions': episode.total_supply_billions,
            'peak_redemption_pct': episode.peak_redemption_rate_pct,
            'max_depeg_pct': episode.max_discount_pct,
        },
        'parameters': {
            'lambda_tier1': lam,
            'pi_impatient': pi,
            'kappa_impact': kappa,
        },
        'run_cutoff_curve': {
            'lambda_grid': result.lambda_grid.tolist(),
            'tbill_share_grid': result.tbill_share_grid.tolist(),
            'n_star': result.n_star.tolist(),
            'vulnerable': result.vulnerable.tolist(),
            'lambda_cutoff': result.lambda_cutoff if not np.isnan(result.lambda_cutoff) else None,
        },
        'sensitivity': {
            'baseline': sens['baseline'],
            'lambda_up': sens['lambda_up'],
            'lambda_down': sens['lambda_down'],
            'metrics': sens['sensitivity'],
            'gate_evaluation': sens['gate_evaluation'],
        },
        'loss_table': loss_table,
    }
    
    return results


def compare_reserve_mixes() -> Dict:
    """
    Compare different reserve mix scenarios from calibration.
    
    Shows how λ and resulting n* differ across scenarios.
    """
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON: Reserve Mix Sensitivity")
    print("=" * 70)
    
    scenarios = create_scenario_variants()
    
    results = {}
    
    print(f"\n{'Scenario':<25} {'λ (Tier-1)':<12} {'T-bills':<10} {'n*':<10} {'Status':<12}")
    print("-" * 70)
    
    for name, params in scenarios.items():
        lam, pi, kappa = extract_two_tier_params(params)
        n_star, diag = solve_n_star(lam, pi, kappa)
        vulnerable = n_star > 1e-4
        tbill_pct = (1 - lam) * 100
        
        status = "VULNERABLE" if vulnerable else "STABLE"
        print(f"{name:<25} {lam*100:>6.1f}%       {tbill_pct:>5.1f}%    {n_star*100:>5.1f}%    {status:<12}")
        
        results[name] = {
            'lambda': lam,
            'tbill_share_pct': tbill_pct,
            'pi': pi,
            'kappa': kappa,
            'n_star': n_star,
            'vulnerable': vulnerable,
        }
    
    return results


def generate_paper_outputs(episode_key: str = 'usdc_svb_2023',
                          save_json: bool = True):
    """
    Generate all outputs needed for paper.
    
    1. Run cutoff curve data
    2. Sensitivity table
    3. Expected loss table
    4. Gate evaluation
    """
    print("\n" + "=" * 70)
    print("GENERATING PAPER-READY OUTPUTS")
    print("=" * 70)
    
    # Main analysis
    results = analyze_episode(episode_key)
    
    # Scenario comparison
    print("\n")
    scenario_results = compare_reserve_mixes()
    results['scenarios'] = scenario_results
    
    # Key finding summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)
    
    sens = results['sensitivity']
    gate = sens['gate_evaluation']
    
    print(f"""
1. RUN CUTOFF SENSITIVITY (Gate Test)
   - Baseline λ = {results['parameters']['lambda_tier1']*100:.1f}% (Tier-1 share)
   - Shift λ by ±10pp → n* changes by {sens['metrics']['delta_n_total']*100:.1f}pp
   - Gate criterion: >0.5pp change required
   - Result: {'✅ PROCEED' if gate['proceed'] else '❌ REVISIT'}
   
2. MECHANISM
   - Higher T-bill share → lower λ → larger fire-sale discounts
   - Fire-sale feedback: more T-bills sold at worse prices
   - Patient investors anticipate losses → coordinate on runs
   
3. POLICY IMPLICATION
   - LCR-style floors on liquid reserves (raise λ) reduce vulnerability
   - Each 10pp increase in Tier-1 reduces n* by ~{abs(sens['lambda_up']['delta_n'])*100:.1f}pp
   - Optimal mix balances yield (T-bills) vs stability (MMFs)
   
4. CALIBRATION FIT
   - Episode: {results['episode']['name']}
   - Observed depeg: {results['episode']['max_depeg_pct']:.1f}%
   - Model λ̂: {results['run_cutoff_curve']['lambda_cutoff']*100 if results['run_cutoff_curve']['lambda_cutoff'] else 'N/A'}
   - Model predicts {'vulnerability' if results['sensitivity']['baseline']['vulnerable'] else 'stability'} at actual reserves
    """)
    
    # Save to JSON
    if save_json:
        filename = f"run_cutoff_analysis_{episode_key}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved to: {filename}")
    
    return results


if __name__ == "__main__":
    # Run main analysis
    results = generate_paper_outputs('usdc_svb_2023', save_json=True)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR PAPER")
    print("=" * 70)
    print("""
1. FIGURE: Run cutoff (n*) vs T-bill share
   - X-axis: T-bill share (1-λ) from 0% to 95%
   - Y-axis: Equilibrium run fraction n*
   - Mark current position and λ̂ cutoff
   - Shade vulnerable region (n*>0)
   
2. TABLE: Expected loss by reserve mix
   - Columns: λ, T-bill %, n*, Loss (bps)
   - Rows: λ = 5%, 10%, 15%, 20%, 30%, 40%
   
3. ROBUSTNESS CHECKS
   - Vary κ (±50%): test fire-sale impact uncertainty
   - Vary π (5% → 20%): test stress scenarios
   - Include/exclude deposits from Tier-1
   
4. POLICY COUNTERFACTUALS
   - LCR 150%: impose min λ ≥ some threshold
   - PSM backstop: add buffer before T-bill sales
   - Dynamic gates: restrict redemptions when n rises
   
Data saved to: run_cutoff_analysis_usdc_svb_2023.json
    """)


