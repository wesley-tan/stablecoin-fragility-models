"""
Basic Example: Stablecoin Fragility Analysis
============================================

Quick demonstration of core functionality.
"""

from stablecoin_fragility import (
    create_usdc_baseline,
    StablecoinFragilityModel,
    StablecoinParameters,
    ReserveAsset,
    AssetClass,
)

def example_1_basic_analysis():
    """Example 1: Basic fragility analysis"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Fragility Analysis")
    print("=" * 70)
    
    # Create baseline USDC model
    params = create_usdc_baseline()
    model = StablecoinFragilityModel(params)
    
    print(f"\n📊 Model Setup:")
    print(f"  Supply: ${params.total_supply:.0f}M")
    print(f"  Reserves: ${params.total_reserves:.0f}M")
    print(f"  Reserve Ratio: {params.reserve_ratio*100:.1f}%")
    print(f"  Redemption Shock: {params.redemption_shock*100:.1f}%")
    
    # Compute run threshold
    print(f"\n🎯 Run Threshold Analysis:")
    run_cutoff, diag = model.compute_run_threshold()
    print(f"  Run Cutoff: {run_cutoff:.0f} bps")
    print(f"  Fire-Sale Impact: {diag['fire_sale_impact_bps']:.1f} bps")
    print(f"  Available Liquidity: ${diag['available_liquidity']:.1f}M")
    print(f"  Liquidity Gap: ${diag['liquidity_gap']:.1f}M")
    
    # LCR
    print(f"\n💧 Liquidity Coverage:")
    lcr, lcr_diag = model.compute_lcr()
    print(f"  LCR: {lcr*100:.0f}%")
    print(f"  HQLA: ${lcr_diag['hqla']:.1f}M")
    print(f"  Meets 100% Requirement: {lcr_diag['meets_requirement']}")
    
    # Run probability
    print(f"\n🎲 Run Risk:")
    prob = model.run_probability()
    print(f"  Run Probability: {prob*100:.2f}%")
    
    # Expected loss
    print(f"\n💥 Expected Loss:")
    exp_loss, loss_diag = model.expected_loss()
    print(f"  Expected Loss: ${exp_loss:.2f}M")
    print(f"  Loss if Run: ${loss_diag['total_loss_if_run']:.2f}M")
    print(f"  Loss % of Supply: {loss_diag['loss_as_pct_supply']:.2f}%")


def example_2_compare_reserves():
    """Example 2: Compare different reserve compositions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Reserve Composition Comparison")
    print("=" * 70)
    
    # Conservative: 90% T-bills
    conservative_reserves = [
        ReserveAsset(AssetClass.TBILLS, 900.0, 0.0465, 2, 1.0),
        ReserveAsset(AssetClass.MMF, 100.0, 0.0442, 10, 0.95),
    ]
    
    # Balanced: 60% T-bills (baseline)
    balanced_reserves = [
        ReserveAsset(AssetClass.TBILLS, 600.0, 0.0465, 2, 1.0),
        ReserveAsset(AssetClass.DEPOSITS, 200.0, 0.035, 500, 0.0),
        ReserveAsset(AssetClass.REPO, 100.0, 0.048, 10, 0.85),
        ReserveAsset(AssetClass.MMF, 100.0, 0.0442, 10, 0.95),
    ]
    
    # Risky: 30% T-bills
    risky_reserves = [
        ReserveAsset(AssetClass.TBILLS, 300.0, 0.0465, 2, 1.0),
        ReserveAsset(AssetClass.DEPOSITS, 500.0, 0.035, 500, 0.0),
        ReserveAsset(AssetClass.REPO, 100.0, 0.048, 10, 0.85),
        ReserveAsset(AssetClass.MMF, 100.0, 0.0442, 10, 0.95),
    ]
    
    scenarios = {
        'Conservative (90% T-bills)': conservative_reserves,
        'Balanced (60% T-bills)': balanced_reserves,
        'Risky (30% T-bills)': risky_reserves,
    }
    
    print(f"\n{'Scenario':<30} {'Run Cutoff':<15} {'LCR':<10} {'Run Prob':<12} {'Exp Loss':<12}")
    print("-" * 80)
    
    for name, reserves in scenarios.items():
        params = StablecoinParameters(
            total_supply=1000.0,
            reserves=reserves,
            redemption_shock=0.11,
        )
        model = StablecoinFragilityModel(params)
        
        run_cutoff, _ = model.compute_run_threshold()
        lcr, _ = model.compute_lcr()
        prob = model.run_probability()
        exp_loss, _ = model.expected_loss()
        
        print(f"{name:<30} {run_cutoff:>10.0f} bps   {lcr*100:>5.0f}%   {prob*100:>8.2f}%   ${exp_loss:>8.2f}M")


def example_3_backstop_analysis():
    """Example 3: Analyze PSM backstop sizing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: PSM Backstop Analysis")
    print("=" * 70)
    
    base_params = create_usdc_baseline()
    
    print(f"\n{'Backstop Size':<15} {'Max Run Prob':<15} {'Critical Rate':<15}")
    print("-" * 50)
    
    for backstop in [0, 50, 100, 200, 500]:
        params = StablecoinParameters(
            total_supply=base_params.total_supply,
            reserves=base_params.reserves.copy(),
            redemption_shock=base_params.redemption_shock,
            psm_buffer=backstop,
        )
        
        model = StablecoinFragilityModel(params)
        max_prob = model.max_run_probability()
        critical_rate = model.critical_redemption_rate()
        
        print(f"${backstop:>5.0f}M         {max_prob*100:>8.2f}%        {critical_rate:>8.2f}%")
    
    print(f"\n💡 Finding: Each $100M backstop reduces fragility")
    print(f"   Optimal backstop depends on systemic risk tolerance")


def example_4_policy_comparison():
    """Example 4: Compare policy interventions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Policy Intervention Comparison")
    print("=" * 70)
    
    from stablecoin_fragility import policy_counterfactuals
    
    base_params = create_usdc_baseline()
    scenarios = policy_counterfactuals(base_params)
    
    print(f"\n{'Policy':<20} {'Exp Loss':<12} {'LCR':<8} {'Peg Spread':<12} {'Run Prob':<10}")
    print("-" * 70)
    
    for name, metrics in scenarios.items():
        print(f"{name:<20} "
              f"${metrics['expected_loss']:>7.2f}M   "
              f"{metrics['lcr_pct']:>5.0f}%   "
              f"{metrics['peg_spread_pct']:>7.2f}%     "
              f"{metrics['run_probability']:>6.2f}%")
    
    print(f"\n💡 Key Insight: Multi-pronged approach (T-bills + LCR + PSM) most effective")


def main():
    """Run all examples"""
    example_1_basic_analysis()
    example_2_compare_reserves()
    example_3_backstop_analysis()
    example_4_policy_comparison()
    
    print("\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run 'python run_analysis.py' for comprehensive analysis")
    print("  - Check calibration.py for historical episodes")
    print("  - See onchain_frictions.py for rapid run dynamics")


if __name__ == "__main__":
    main()


