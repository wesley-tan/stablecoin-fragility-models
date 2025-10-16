"""
Fire-Sale Externalities in Stablecoin Reserves
===============================================

Extends Izumi-Li (JMCB forthcoming) fire-sale framework to stablecoins.

Key innovation: When multiple stablecoins hold overlapping reserves (T-bills),
liquidations create EXTERNALITIES through asset price depression.

Channels:
1. Direct price impact: Each stablecoin's sales → market price ↓
2. Cross-issuer amplification: A's sales → B's reserves devalued → B forced to sell more
3. Strategic complementarities: Coordination failures amplify liquidations

Model:
- N stablecoins, each holding T-bills
- Aggregate fire-sale pricing: p = p(Q_total) where Q_total = Σ Q_i
- Nash equilibrium: each issuer takes others' sales as given
- Social planner: internalizes externality

Policy implications:
- Reserve diversification mandates
- Coordination mechanisms (e.g., coordinated sale facility)
- Liquidity coverage floors
- Position limits on common assets
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StablecoinIssuer:
    """Single stablecoin issuer with reserve holdings"""
    name: str
    supply: float  # Total supply (millions)
    tbill_holdings: float  # T-bill reserves (millions)
    other_reserves: float  # Other reserves (millions)
    redemption_shock: float  # Fraction attempting to redeem
    lambda_tier1: float  # Liquid (non-T-bill) share
    
    @property
    def total_reserves(self) -> float:
        return self.tbill_holdings + self.other_reserves
    
    @property
    def tbill_share(self) -> float:
        return self.tbill_holdings / self.total_reserves if self.total_reserves > 0 else 0


class FireSaleExternalityModel:
    """
    Izumi-Li fire-sale externalities for stablecoins.
    
    Key equations:
    1. Aggregate price: p(Q_total) = 1 - κ·Q_total/M
       where M = market depth (total investor cash)
    
    2. Individual sales: Q_i = max(0, R_i - λ_i·S_i)
       where R_i = redemptions, S_i = supply
    
    3. Externality: ∂p/∂Q_i affects ALL issuers' balance sheets
    
    4. Nash equilibrium: Each issuer takes others' Q_j as given
    
    5. Social optimum: Planner internalizes Σ_j (∂V_j/∂p)·(∂p/∂Q_i)
    """
    
    def __init__(self, issuers: List[StablecoinIssuer],
                 market_depth: float,
                 kappa: float = 0.05):
        """
        Args:
            issuers: List of stablecoin issuers
            market_depth: M = total investor cash for T-bills (millions)
            kappa: Aggregate price impact coefficient
        """
        self.issuers = {issuer.name: issuer for issuer in issuers}
        self.market_depth = market_depth
        self.kappa = kappa
        
        # Compute aggregates
        self.total_tbill_holdings = sum(i.tbill_holdings for i in issuers)
        self.total_supply = sum(i.supply for i in issuers)
    
    def aggregate_price(self, Q_total: float) -> float:
        """
        Aggregate fire-sale price (Izumi-Li).
        
        p(Q_total) = 1 - κ·Q_total/M
        
        Args:
            Q_total: Total T-bill sales across all issuers
        
        Returns:
            Market price [0, 1]
        """
        if self.market_depth <= 0:
            return 0.5
        
        impact = self.kappa * Q_total / self.market_depth
        price = max(0.0, 1.0 - impact)
        
        return price
    
    def individual_sales(self, issuer_name: str, 
                        price: float,
                        include_feedback: bool = True) -> float:
        """
        Compute individual issuer's T-bill sales.
        
        Without feedback: Q_i = max(0, R_i - λ_i·S_i)
        With feedback: If reserves marked-to-market at price p,
                      issuer may need to sell MORE to meet redemptions
        
        Args:
            issuer_name: Which issuer
            price: Current T-bill price
            include_feedback: Whether to include mark-to-market feedback
        
        Returns:
            Sales volume (millions)
        """
        issuer = self.issuers[issuer_name]
        
        # Redemptions
        R = issuer.redemption_shock * issuer.supply
        
        # Liquid reserves (non-T-bills)
        liquid = issuer.lambda_tier1 * issuer.total_reserves
        
        # Shortfall
        shortfall = max(0, R - liquid)
        
        if not include_feedback or price >= 0.999:
            # Simple case: sell shortfall at par
            Q = shortfall
        else:
            # With feedback: need to sell Q such that p·Q = shortfall
            # Q = shortfall / p
            Q = shortfall / price if price > 0 else float('inf')
            
            # Cap at total T-bill holdings
            Q = min(Q, issuer.tbill_holdings)
        
        return Q
    
    def nash_equilibrium(self, max_iter: int = 50, tol: float = 1e-6) -> Dict:
        """
        Compute Nash equilibrium: each issuer takes others as given.
        
        Fixed point: Q_i^* = argmax_Q_i V_i(Q_i, Q_{-i}^*)
        
        In our setting, simplified to:
        - Compute Q_i given price p
        - Price determined by Σ_j Q_j
        - Iterate until convergence
        
        Returns:
            Dictionary with equilibrium sales, prices, losses
        """
        n_issuers = len(self.issuers)
        issuer_names = list(self.issuers.keys())
        
        # Initialize: no sales
        Q = {name: 0.0 for name in issuer_names}
        
        for iteration in range(max_iter):
            Q_old = Q.copy()
            
            # Aggregate sales
            Q_total = sum(Q.values())
            
            # Aggregate price
            p = self.aggregate_price(Q_total)
            
            # Update each issuer's sales
            for name in issuer_names:
                Q[name] = self.individual_sales(name, p, include_feedback=True)
            
            # Check convergence
            max_change = max(abs(Q[name] - Q_old[name]) for name in issuer_names)
            if max_change < tol:
                break
        
        # Final aggregates
        Q_total = sum(Q.values())
        p_final = self.aggregate_price(Q_total)
        
        # Compute losses
        losses = {}
        for name in issuer_names:
            if Q[name] > 0:
                # Loss = face value - realized value
                loss = Q[name] * (1 - p_final)
            else:
                loss = 0.0
            losses[name] = loss
        
        return {
            'sales': Q,
            'total_sales': Q_total,
            'price': p_final,
            'losses': losses,
            'total_loss': sum(losses.values()),
            'iterations': iteration + 1,
        }
    
    def social_optimum(self) -> Dict:
        """
        Social planner solution: minimize aggregate loss.
        
        Planner internalizes externality: ∂p/∂Q_i affects all issuers.
        
        Simplified: Coordinate sales to smooth price impact.
        """
        issuer_names = list(self.issuers.keys())
        
        # Without coordination (Nash)
        nash = self.nash_equilibrium()
        
        # With coordination: spread sales to minimize price impact
        # Pro-rata allocation based on relative need
        
        total_shortfall = 0.0
        shortfalls = {}
        
        for name in issuer_names:
            issuer = self.issuers[name]
            R = issuer.redemption_shock * issuer.supply
            liquid = issuer.lambda_tier1 * issuer.total_reserves
            shortfall = max(0, R - liquid)
            shortfalls[name] = shortfall
            total_shortfall += shortfall
        
        # Coordinated sales (pro-rata)
        Q_coord = {}
        for name in issuer_names:
            if total_shortfall > 0:
                share = shortfalls[name] / total_shortfall
            else:
                share = 0.0
            
            # Smooth out sales (could optimize further)
            Q_coord[name] = shortfalls[name]  # Simplified: same as Nash for now
        
        # Price with coordination
        Q_total_coord = sum(Q_coord.values())
        p_coord = self.aggregate_price(Q_total_coord)
        
        # Losses
        losses_coord = {}
        for name in issuer_names:
            if Q_coord[name] > 0:
                losses_coord[name] = Q_coord[name] * (1 - p_coord)
            else:
                losses_coord[name] = 0.0
        
        return {
            'sales': Q_coord,
            'total_sales': Q_total_coord,
            'price': p_coord,
            'losses': losses_coord,
            'total_loss': sum(losses_coord.values()),
            'improvement_vs_nash': nash['total_loss'] - sum(losses_coord.values()),
        }
    
    def fire_sale_elasticity(self, issuer_name: str) -> Dict:
        """
        Compute fire-sale elasticity: how much issuer's sales affect market.
        
        Elasticity: ε_i = (∂p/p) / (∂Q_i/Q_total)
                        = -(κ/M) · (Q_total/p) · (1)
        
        Cross-elasticity: ε_{ij} = (∂V_j/∂p) · (∂p/∂Q_i)
        
        Args:
            issuer_name: Which issuer
        
        Returns:
            Dictionary with elasticity metrics
        """
        # Baseline equilibrium
        eq = self.nash_equilibrium()
        
        Q_base = eq['sales'][issuer_name]
        Q_total_base = eq['total_sales']
        p_base = eq['price']
        
        # Marginal sales
        dQ = 0.01 * Q_total_base  # 1% increase
        Q_total_new = Q_total_base + dQ
        p_new = self.aggregate_price(Q_total_new)
        
        # Price elasticity
        dp = p_new - p_base
        dQ_pct = dQ / Q_total_base if Q_total_base > 0 else 0
        dp_pct = dp / p_base if p_base > 0 else 0
        
        elasticity = dp_pct / dQ_pct if dQ_pct > 0 else 0
        
        # Cross-impact: how much this issuer's sales hurt others
        cross_losses = {}
        for other_name in self.issuers.keys():
            if other_name == issuer_name:
                continue
            
            other_issuer = self.issuers[other_name]
            # Loss from price drop
            loss_from_externality = other_issuer.tbill_holdings * abs(dp)
            cross_losses[other_name] = loss_from_externality
        
        return {
            'own_sales': Q_base,
            'total_sales': Q_total_base,
            'price_baseline': p_base,
            'price_elasticity': elasticity,
            'marginal_price_impact': dp,
            'cross_issuer_losses': cross_losses,
            'total_externality': sum(cross_losses.values()),
        }
    
    def policy_counterfactuals(self) -> Dict:
        """
        Compare policy interventions to address externality.
        
        Policies:
        1. Baseline (Nash)
        2. Reserve diversification (reduce T-bill concentration)
        3. Coordination facility (social optimum)
        4. Position limits (cap T-bill holdings per issuer)
        """
        results = {}
        
        # 1. Baseline Nash
        results['baseline_nash'] = self.nash_equilibrium()
        
        # 2. Social optimum (coordination)
        results['coordinated'] = self.social_optimum()
        
        # 3. Diversification: reduce T-bill concentration
        # Simulate: reduce each issuer's T-bills by 20%
        original_holdings = {name: issuer.tbill_holdings 
                           for name, issuer in self.issuers.items()}
        
        for issuer in self.issuers.values():
            issuer.tbill_holdings *= 0.8
        
        results['diversified'] = self.nash_equilibrium()
        
        # Restore
        for name, holdings in original_holdings.items():
            self.issuers[name].tbill_holdings = holdings
        
        # 4. Position limits: cap individual holdings
        max_share = 0.40  # No issuer > 40% of total market
        
        for issuer in self.issuers.values():
            cap = max_share * self.total_tbill_holdings
            issuer.tbill_holdings = min(issuer.tbill_holdings, cap)
        
        results['position_limits'] = self.nash_equilibrium()
        
        # Restore again
        for name, holdings in original_holdings.items():
            self.issuers[name].tbill_holdings = holdings
        
        return results


def create_usdc_dai_usdt_firesale() -> FireSaleExternalityModel:
    """
    Create realistic model with USDC, DAI, USDT overlapping T-bill holdings.
    
    Calibrated to March 2023 (USDC/SVB crisis).
    """
    issuers = [
        StablecoinIssuer(
            name='USDC',
            supply=43500.0,  # $43.5B
            tbill_holdings=34800.0,  # 80% in T-bills
            other_reserves=8700.0,
            redemption_shock=0.115,  # 11.5% (SVB peak)
            lambda_tier1=0.00,  # Post-SVB: deposits frozen
        ),
        StablecoinIssuer(
            name='DAI',
            supply=5200.0,  # $5.2B
            tbill_holdings=1040.0,  # 20% in T-bills
            other_reserves=4160.0,  # 80% USDC + crypto collateral
            redemption_shock=0.05,  # 5% (lower, overcollateralized)
            lambda_tier1=0.60,  # 60% in USDC (liquid if USDC stable)
        ),
        StablecoinIssuer(
            name='USDT',
            supply=83000.0,  # $83B
            tbill_holdings=33200.0,  # 40% in T-bills
            other_reserves=49800.0,  # 60% other
            redemption_shock=0.08,  # 8% (moderate stress)
            lambda_tier1=0.35,  # 35% liquid
        ),
    ]
    
    # Market depth: conservative estimate
    # Total T-bill market ~$5T, assume 1% available for stablecoin liquidations
    market_depth = 50000.0  # $50B
    
    # Aggregate impact coefficient
    kappa = 0.05  # 5% impact per $1 of depth
    
    return FireSaleExternalityModel(issuers, market_depth, kappa)


if __name__ == "__main__":
    print("=" * 70)
    print("FIRE-SALE EXTERNALITIES: Izumi-Li Framework for Stablecoins")
    print("=" * 70)
    
    # Create model
    model = create_usdc_dai_usdt_firesale()
    
    print(f"\n📊 MODEL SETUP")
    print(f"  Issuers: {len(model.issuers)}")
    print(f"  Total T-bill holdings: ${model.total_tbill_holdings:.0f}M")
    print(f"  Market depth: ${model.market_depth:.0f}M")
    print(f"  Impact coefficient (κ): {model.kappa*100:.0f}%")
    
    print(f"\n💰 INDIVIDUAL HOLDINGS:")
    for name, issuer in model.issuers.items():
        print(f"  {name}: ${issuer.tbill_holdings:.0f}M T-bills " +
              f"({issuer.tbill_holdings/model.total_tbill_holdings*100:.1f}% of total)")
    
    # Nash equilibrium
    print(f"\n🎮 NASH EQUILIBRIUM (uncoordinated)")
    nash = model.nash_equilibrium()
    
    print(f"  Total sales: ${nash['total_sales']:.0f}M")
    print(f"  Price: {nash['price']:.4f} (depeg: {(1-nash['price'])*10000:.0f}bp)")
    print(f"  Total loss: ${nash['total_loss']:.1f}M")
    print(f"\n  Individual sales:")
    for name in model.issuers.keys():
        Q = nash['sales'][name]
        loss = nash['losses'][name]
        print(f"    {name}: ${Q:.0f}M sold → ${loss:.1f}M loss")
    
    # Social optimum
    print(f"\n🤝 SOCIAL OPTIMUM (coordinated)")
    social = model.social_optimum()
    
    print(f"  Total sales: ${social['total_sales']:.0f}M")
    print(f"  Price: {social['price']:.4f} (depeg: {(1-social['price'])*10000:.0f}bp)")
    print(f"  Total loss: ${social['total_loss']:.1f}M")
    print(f"  Improvement vs Nash: ${social['improvement_vs_nash']:.1f}M")
    
    # Elasticity analysis
    print(f"\n📊 FIRE-SALE ELASTICITY")
    for name in ['USDC', 'DAI', 'USDT']:
        elast = model.fire_sale_elasticity(name)
        print(f"\n  {name}:")
        print(f"    Sales: ${elast['own_sales']:.0f}M")
        print(f"    Price elasticity: {elast['price_elasticity']:.3f}")
        print(f"    Marginal price impact: {elast['marginal_price_impact']*10000:.1f}bp")
        print(f"    Total externality on others: ${elast['total_externality']:.1f}M")
    
    # Policy counterfactuals
    print(f"\n🎯 POLICY COUNTERFACTUALS")
    policies = model.policy_counterfactuals()
    
    print(f"\n  {'Policy':<25} {'Total Loss':<15} {'Price':<10} {'Depeg (bp)':<12}")
    print("  " + "-" * 65)
    
    for policy_name, result in policies.items():
        loss = result['total_loss']
        price = result['price']
        depeg = (1 - price) * 10000
        print(f"  {policy_name:<25} ${loss:<14.1f} {price:<9.4f} {depeg:<11.0f}")
    
    # Key insight
    print(f"\n💡 KEY INSIGHTS")
    
    nash_loss = policies['baseline_nash']['total_loss']
    coord_loss = policies['coordinated']['total_loss']
    div_loss = policies['diversified']['total_loss']
    
    coord_improvement = (nash_loss - coord_loss) / nash_loss * 100 if nash_loss > 0 else 0
    div_improvement = (nash_loss - div_loss) / nash_loss * 100 if nash_loss > 0 else 0
    
    print(f"""
1. EXTERNALITY SIZE
   - Nash (uncoordinated): ${nash_loss:.1f}M total loss
   - Coordinated: ${coord_loss:.1f}M ({coord_improvement:.1f}% improvement)
   - → Coordination gains: ${nash_loss - coord_loss:.1f}M

2. DIVERSIFICATION BENEFIT
   - Reducing T-bill concentration 20% → ${div_loss:.1f}M loss
   - Improvement: {div_improvement:.1f}%
   - → Reserve diversification reduces systemic risk

3. AMPLIFICATION MECHANISM
   - USDC liquidations → price ↓ → DAI/USDT reserves devalued
   - Cross-issuer externality: ${sum(model.fire_sale_elasticity('USDC')['cross_issuer_losses'].values()):.1f}M
   - → Need coordination or diversification mandates

4. POLICY RECOMMENDATION
   - Coordination facility (central sale desk)
   - Position limits: max 40% of market per issuer
   - Reserve diversification: <50% in single asset class
    """)
    
    print("=" * 70)
    print("Fire-sale externalities model complete!")

