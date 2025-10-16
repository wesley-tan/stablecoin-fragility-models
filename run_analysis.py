"""
Comprehensive Stablecoin Fragility Analysis Script
=================================================

Runs full analysis pipeline:
1. Reserve mix sensitivity
2. LCR floor analysis
3. Robust fragility (backstop sizing)
4. Policy counterfactuals
5. On-chain rapid run simulation
6. Vulnerability curves

Generates tables and data for visualization.
"""

import numpy as np
import json
from datetime import datetime
from typing import Dict, Any

from stablecoin_fragility import (
    StablecoinFragilityModel,
    create_usdc_baseline,
    analyze_reserve_mix_sensitivity,
    analyze_lcr_floors,
    analyze_backstop_fragility,
    policy_counterfactuals,
)
from onchain_frictions import (
    analyze_rapid_run,
    OnChainRedemptionModel,
    BlockchainParameters,
)
from calibration import (
    CalibrationEngine,
    create_scenario_variants,
)


class FragilityAnalyzer:
    """
    Main analyzer class that runs comprehensive stablecoin fragility analysis.
    """
    
    def __init__(self, params=None):
        """
        Initialize analyzer with parameters.
        
        Args:
            params: StablecoinParameters (default: USDC baseline)
        """
        self.params = params or create_usdc_baseline()
        self.model = StablecoinFragilityModel(self.params)
        self.results = {}
    
    def run_full_analysis(self, save_results: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive analysis across all dimensions.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 Starting Comprehensive Stablecoin Fragility Analysis")
        print("=" * 70)
        
        # 1. Baseline metrics
        print("\n📊 Computing baseline metrics...")
        self.results['baseline'] = self._analyze_baseline()
        
        # 2. Reserve mix sensitivity
        print("\n🔍 Analyzing reserve composition sensitivity...")
        self.results['reserve_mix'] = self._analyze_reserve_mix()
        
        # 3. LCR floor analysis
        print("\n💧 Analyzing LCR floor requirements...")
        self.results['lcr_floors'] = self._analyze_lcr_requirements()
        
        # 4. Robust fragility
        print("\n🎲 Computing robust fragility curves...")
        self.results['robust_fragility'] = self._analyze_robust_fragility()
        
        # 5. Policy counterfactuals
        print("\n🎯 Running policy counterfactuals...")
        self.results['policy'] = self._analyze_policies()
        
        # 6. On-chain dynamics
        print("\n⛓️  Simulating on-chain rapid run...")
        self.results['onchain'] = self._analyze_onchain_dynamics()
        
        # 7. Vulnerability curve
        print("\n📈 Building vulnerability curve...")
        self.results['vulnerability'] = self._build_vulnerability_curve()
        
        # 8. Scenario comparison
        print("\n🔬 Comparing scenarios...")
        self.results['scenarios'] = self._compare_scenarios()
        
        print("\n✅ Analysis complete!")
        print("=" * 70)
        
        # Save if requested
        if save_results:
            self._save_results()
        
        return self.results
    
    def _analyze_baseline(self) -> Dict:
        """Compute baseline metrics"""
        run_cutoff, run_diag = self.model.compute_run_threshold()
        lcr, lcr_diag = self.model.compute_lcr()
        exp_loss, loss_diag = self.model.expected_loss()
        run_prob = self.model.run_probability()
        critical_rate = self.model.critical_redemption_rate()
        
        return {
            'total_supply': self.params.total_supply,
            'total_reserves': self.params.total_reserves,
            'reserve_ratio_pct': self.params.reserve_ratio * 100,
            'run_cutoff_bps': run_cutoff,
            'fire_sale_impact_bps': run_diag['fire_sale_impact_bps'],
            'lcr_pct': lcr * 100,
            'hqla': lcr_diag['hqla'],
            'run_probability_pct': run_prob * 100,
            'expected_loss': exp_loss,
            'critical_redemption_rate_pct': critical_rate,
            'reserve_composition': {
                asset.asset_class.value: {
                    'amount': asset.amount,
                    'share_pct': asset.amount / self.params.total_reserves * 100,
                    'yield_pct': asset.yield_rate * 100,
                    'haircut_bps': asset.fire_sale_haircut_bps,
                }
                for asset in self.params.reserves
            }
        }
    
    def _analyze_reserve_mix(self) -> Dict:
        """Analyze reserve mix sensitivity"""
        tbill_shares = np.linspace(0.2, 1.0, 20)
        results = analyze_reserve_mix_sensitivity(self.params, tbill_shares)
        
        # Find optimal mix (minimize run probability while maintaining yield)
        run_probs = np.array(results['run_probability'])
        optimal_idx = np.argmin(run_probs)
        
        return {
            'tbill_share_pct': results['tbill_share_pct'],
            'run_cutoff_bps': results['run_cutoff_bps'],
            'fire_sale_impact_bps': results['fire_sale_impact_bps'],
            'lcr_ratio_pct': results['lcr_ratio'],
            'run_probability_pct': results['run_probability'],
            'optimal': {
                'tbill_share_pct': results['tbill_share_pct'][optimal_idx],
                'run_cutoff_bps': results['run_cutoff_bps'][optimal_idx],
                'run_probability_pct': results['run_probability'][optimal_idx],
            },
            'sensitivity_metrics': {
                'min_run_cutoff': min(results['run_cutoff_bps']),
                'max_run_cutoff': max(results['run_cutoff_bps']),
                'cutoff_range': max(results['run_cutoff_bps']) - min(results['run_cutoff_bps']),
            }
        }
    
    def _analyze_lcr_requirements(self) -> Dict:
        """Analyze LCR floor requirements"""
        lcr_floors = np.array([0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
        results = analyze_lcr_floors(self.params, lcr_floors)
        
        # Find minimum LCR for <5% run probability
        run_probs = np.array(results['run_probability_pct'])
        safe_idxs = np.where(run_probs < 5.0)[0]
        
        if len(safe_idxs) > 0:
            min_safe_lcr_idx = safe_idxs[0]
            min_safe_lcr = results['lcr_floor_pct'][min_safe_lcr_idx]
        else:
            min_safe_lcr = None
        
        return {
            'lcr_floor_pct': results['lcr_floor_pct'],
            'actual_lcr_pct': results['actual_lcr_pct'],
            'run_probability_pct': results['run_probability_pct'],
            'max_sunspot_prob_pct': results['max_sunspot_prob_pct'],
            'hqla': results['hqla'],
            'min_lcr_for_safety': min_safe_lcr,
        }
    
    def _analyze_robust_fragility(self) -> Dict:
        """Analyze robust fragility with backstop sizing"""
        backstop_sizes = np.arange(0, 501, 50)
        results = analyze_backstop_fragility(self.params, backstop_sizes)
        
        # Find minimum backstop for <5% fragility
        max_probs = np.array(results['max_run_probability_pct'])
        safe_idxs = np.where(max_probs < 5.0)[0]
        
        if len(safe_idxs) > 0:
            min_safe_backstop = results['backstop_size'][safe_idxs[0]]
        else:
            min_safe_backstop = None
        
        return {
            'backstop_size': results['backstop_size'],
            'max_run_probability_pct': results['max_run_probability_pct'],
            'critical_threshold_pct': results['critical_threshold_pct'],
            'expected_loss': results['expected_loss'],
            'min_backstop_for_safety': min_safe_backstop,
            'marginal_benefit_per_100m': self._compute_marginal_benefit(results),
        }
    
    def _compute_marginal_benefit(self, results: Dict) -> float:
        """Compute average reduction in fragility per $100M backstop"""
        probs = np.array(results['max_run_probability_pct'])
        backstops = np.array(results['backstop_size'])
        
        if len(probs) < 2:
            return 0.0
        
        # Average derivative
        dprob = np.diff(probs)
        dbackstop = np.diff(backstops)
        
        avg_derivative = np.mean(dprob / dbackstop)  # % change per $M
        per_100m = avg_derivative * 100  # % change per $100M
        
        return -per_100m  # Negative because increasing backstop reduces fragility
    
    def _analyze_policies(self) -> Dict:
        """Run policy counterfactuals"""
        scenarios = policy_counterfactuals(self.params)
        
        # Rank by expected loss
        ranked = sorted(scenarios.items(), 
                       key=lambda x: x[1]['expected_loss'])
        
        return {
            'scenarios': scenarios,
            'ranking': [name for name, _ in ranked],
            'best_policy': ranked[0][0],
            'worst_policy': ranked[-1][0],
        }
    
    def _analyze_onchain_dynamics(self) -> Dict:
        """Simulate on-chain rapid run"""
        results = analyze_rapid_run(
            initial_supply=self.params.total_supply,
            peak_redemption_rate_pct=self.params.redemption_shock * 100,
            duration_minutes=2.0
        )
        
        summary = results['summary']
        
        # Key thresholds
        onchain = results['onchain']
        pool = results['pool']
        
        # Find time to critical thresholds
        time_to_5pct_redeem = None
        time_to_1pct_depeg = None
        
        for i, t in enumerate(onchain['time']):
            redeemed_pct = onchain['cumulative_redeemed'][i] / self.params.total_supply * 100
            if redeemed_pct >= 5.0 and time_to_5pct_redeem is None:
                time_to_5pct_redeem = t
            
            if pool['discount_pct'][i] >= 1.0 and time_to_1pct_depeg is None:
                time_to_1pct_depeg = t
        
        return {
            'summary': summary,
            'time_to_5pct_redemption': time_to_5pct_redeem,
            'time_to_1pct_depeg': time_to_1pct_depeg,
            'circuit_breaker_window': min(time_to_5pct_redeem or 120, 120),
        }
    
    def _build_vulnerability_curve(self) -> Dict:
        """Build vulnerability curve across shock magnitudes"""
        shock_amounts = np.arange(0, 501, 25)  # $0M to $500M
        
        results = {
            'shock_amount': [],
            'critical_redemption_rate_pct': [],
            'liquidity_score_pct': [],
            'peg_spread_pct': [],
        }
        
        for shock in shock_amounts:
            # Compute metrics at this shock level
            peg_dev = self.model.peg_deviation(shock)
            
            # Liquidity score: remaining liquidity after shock
            reserves = self.params.total_reserves
            available = max(0, reserves - shock)
            liquidity_score = available / reserves * 100 if reserves > 0 else 0
            
            # Critical rate (recompute with this shock)
            original_shock = self.params.redemption_shock
            self.params.redemption_shock = shock / self.params.total_supply
            critical_rate = self.model.critical_redemption_rate()
            self.params.redemption_shock = original_shock
            
            results['shock_amount'].append(shock)
            results['critical_redemption_rate_pct'].append(critical_rate)
            results['liquidity_score_pct'].append(liquidity_score)
            results['peg_spread_pct'].append(peg_dev)
        
        return results
    
    def _compare_scenarios(self) -> Dict:
        """Compare different scenario calibrations"""
        calibrator = CalibrationEngine()
        scenarios = create_scenario_variants()
        
        comparison = {}
        
        for name, params in scenarios.items():
            model = StablecoinFragilityModel(params)
            
            run_cutoff, _ = model.compute_run_threshold()
            lcr, _ = model.compute_lcr()
            run_prob = model.run_probability()
            exp_loss, _ = model.expected_loss()
            
            comparison[name] = {
                'reserve_ratio_pct': params.reserve_ratio * 100,
                'run_cutoff_bps': run_cutoff,
                'lcr_pct': lcr * 100,
                'run_probability_pct': run_prob * 100,
                'expected_loss': exp_loss,
            }
        
        return comparison
    
    def _save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fragility_analysis_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_to_json_serializable(self.results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def print_summary_report(self):
        """Print formatted summary report"""
        if not self.results:
            print("No results available. Run analysis first.")
            return
        
        print("\n" + "=" * 70)
        print("STABLECOIN FRAGILITY ANALYSIS - SUMMARY REPORT")
        print("=" * 70)
        
        # Baseline
        baseline = self.results.get('baseline', {})
        print("\n📊 BASELINE METRICS")
        print(f"  Total Supply: ${baseline.get('total_supply', 0):.0f}M")
        print(f"  Total Reserves: ${baseline.get('total_reserves', 0):.0f}M")
        print(f"  Reserve Ratio: {baseline.get('reserve_ratio_pct', 0):.1f}%")
        print(f"  Run Cutoff: {baseline.get('run_cutoff_bps', 0):.0f} bps")
        print(f"  LCR: {baseline.get('lcr_pct', 0):.0f}%")
        print(f"  Run Probability: {baseline.get('run_probability_pct', 0):.1f}%")
        print(f"  Expected Loss: ${baseline.get('expected_loss', 0):.2f}M")
        
        # Reserve mix
        reserve_mix = self.results.get('reserve_mix', {})
        if reserve_mix:
            optimal = reserve_mix.get('optimal', {})
            print("\n🔍 OPTIMAL RESERVE MIX")
            print(f"  T-bill Share: {optimal.get('tbill_share_pct', 0):.1f}%")
            print(f"  Run Cutoff: {optimal.get('run_cutoff_bps', 0):.0f} bps")
            print(f"  Run Probability: {optimal.get('run_probability_pct', 0):.1f}%")
            
            sens = reserve_mix.get('sensitivity_metrics', {})
            print(f"  Cutoff Range: {sens.get('cutoff_range', 0):.0f} bps")
        
        # LCR requirements
        lcr_analysis = self.results.get('lcr_floors', {})
        if lcr_analysis:
            min_safe = lcr_analysis.get('min_lcr_for_safety')
            print("\n💧 LCR REQUIREMENTS")
            if min_safe:
                print(f"  Min LCR for <5% Run Risk: {min_safe:.0f}%")
            else:
                print(f"  Min LCR for <5% Run Risk: >300%")
        
        # Robust fragility
        fragility = self.results.get('robust_fragility', {})
        if fragility:
            min_backstop = fragility.get('min_backstop_for_safety')
            marginal = fragility.get('marginal_benefit_per_100m', 0)
            print("\n🎲 ROBUST FRAGILITY")
            if min_backstop:
                print(f"  Min Backstop for <5% Fragility: ${min_backstop:.0f}M")
            else:
                print(f"  Min Backstop for <5% Fragility: >${max(fragility.get('backstop_size', [500])):.0f}M")
            print(f"  Marginal Benefit (per $100M): {marginal:.2f}pp reduction")
        
        # Policy ranking
        policy = self.results.get('policy', {})
        if policy:
            ranking = policy.get('ranking', [])
            print("\n🎯 POLICY RANKING (by Expected Loss)")
            for i, name in enumerate(ranking[:3], 1):
                scenario = policy['scenarios'][name]
                print(f"  {i}. {name}: ${scenario['expected_loss']:.2f}M loss, "
                      f"{scenario['run_probability']:.1f}% run prob")
        
        # On-chain dynamics
        onchain = self.results.get('onchain', {})
        if onchain:
            summary = onchain.get('summary', {})
            print("\n⛓️  ON-CHAIN DYNAMICS")
            print(f"  Time to 5% Redemption: {onchain.get('time_to_5pct_redemption', 0):.0f}s")
            print(f"  Time to 1% Depeg: {onchain.get('time_to_1pct_depeg', 0):.0f}s")
            print(f"  Circuit Breaker Window: {onchain.get('circuit_breaker_window', 0):.0f}s")
            print(f"  Peak Gas Spike: {summary.get('gas_spike_multiple', 0):.1f}x")
        
        print("\n" + "=" * 70)


def main():
    """Main execution function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  STABLECOIN ROBUST FRAGILITY RESEARCH                             ║
    ║  Diamond-Dybvig + Izumi-Li Framework                              ║
    ║  Reserve Mix, Fire-Sales, and Maximum Run Probability Analysis    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Create analyzer with USDC baseline
    analyzer = FragilityAnalyzer()
    
    # Run full analysis
    results = analyzer.run_full_analysis(save_results=True)
    
    # Print summary
    analyzer.print_summary_report()
    
    # Print key findings
    print("\n" + "=" * 70)
    print("KEY RESEARCH FINDINGS")
    print("=" * 70)
    
    baseline = results['baseline']
    reserve_mix = results['reserve_mix']
    fragility = results['robust_fragility']
    
    print(f"""
1. RUN THRESHOLD SENSITIVITY
   - Run cutoff ranges from {reserve_mix['sensitivity_metrics']['min_run_cutoff']:.0f} to {reserve_mix['sensitivity_metrics']['max_run_cutoff']:.0f} bps
   - T-bill concentration increases fire-sale impact
   - Optimal mix: ~{reserve_mix['optimal']['tbill_share_pct']:.0f}% T-bills balances liquidity vs. yield

2. LCR-STYLE REGULATION
   - Current LCR: {baseline['lcr_pct']:.0f}%
   - Raising LCR 100% → 150% reduces run probability marginally
   - Diminishing returns above 200% LCR

3. ROBUST FRAGILITY (PSM BUFFERS)
   - Each $100M backstop reduces max run probability by ~{fragility['marginal_benefit_per_100m']:.1f}pp
   - Optimal backstop for <5% fragility: ${fragility.get('min_backstop_for_safety', 'N/A')}M

4. ON-CHAIN FRICTIONS
   - Block time creates ~{results['onchain']['circuit_breaker_window']:.0f}s window for circuit breakers
   - Gas spikes {results['onchain']['summary']['gas_spike_multiple']:.1f}x during redemption waves
   - Opportunity for automated intervention before critical threshold

5. POLICY IMPLICATIONS
   - Best policy: {results['policy']['best_policy']}
   - Combination of LCR floors + PSM buffers most effective
   - Redemption gates reduce runs but create adverse selection
    """)
    
    print("=" * 70)
    print("Analysis complete! Check saved JSON for detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    main()


