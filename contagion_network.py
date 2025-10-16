"""
Cross-Issuer Contagion: Network Model
=======================================

Models spillovers between stablecoins via:
1. Shared DEX pools (Curve, Uniswap) - liquidity linkage
2. Common reserve assets (T-bills) - fire-sale cascade
3. Arbitrage flows - peg pressure transmission
4. PSM connections (e.g., DAI-USDC via MakerDAO)

Extends two-tier run model to multi-issuer network.

Key channels:
- Direct: DEX pool imbalance (USDC depeg → DAI sells in USDC/DAI pool)
- Indirect: Fire-sale cascade (USDC sells T-bills → price impact → affects others)
- Information: Coordination spillover (USDC run → fear of DAI run)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from run_cutoff import solve_n_star, fire_sale_price as p_of_R


@dataclass
class Stablecoin:
    """Individual stablecoin with reserves and parameters"""
    name: str
    supply: float  # Total supply (millions)
    lambda_tier1: float  # Liquid reserves share
    pi_baseline: float  # Baseline impatient fraction
    kappa: float  # Market impact coefficient
    reserves_tbills: float  # T-bill holdings (millions)
    reserves_other: float  # Other reserves (millions)
    
    def total_reserves(self) -> float:
        return self.reserves_tbills + self.reserves_other


@dataclass
class DEXPool:
    """DEX pool linking two stablecoins"""
    coin_a: str
    coin_b: str
    reserve_a: float  # Coin A reserves in pool (millions)
    reserve_b: float  # Coin B reserves in pool (millions)
    amplification: float = 100.0  # Curve A parameter (higher = flatter)
    fee_bps: float = 4.0  # Pool fee (4 = 0.04%)
    
    def get_price(self) -> float:
        """Current price: A per B"""
        # Simplified: for balanced Curve pool, price ≈ 1.0
        # For unbalanced: price = reserve_b / reserve_a
        return self.reserve_b / self.reserve_a if self.reserve_a > 0 else 1.0
    
    def price_impact(self, trade_size_a: float) -> float:
        """Price impact of selling trade_size_a of coin A for B"""
        # Simplified constant product with fee
        fee = self.fee_bps / 10000
        
        # New reserves after trade
        new_reserve_a = self.reserve_a + trade_size_a * (1 - fee)
        
        # Constant product: k = reserve_a * reserve_b
        k = self.reserve_a * self.reserve_b
        new_reserve_b = k / new_reserve_a
        
        # Amount of B received
        b_out = self.reserve_b - new_reserve_b
        
        # Effective price
        price = b_out / trade_size_a if trade_size_a > 0 else 1.0
        
        # Deviation from par
        impact_pct = (1.0 - price) * 100
        
        return impact_pct


class ContagionNetwork:
    """
    Multi-issuer contagion model.
    
    Network structure:
    - Nodes: Stablecoins
    - Edges: DEX pools + common reserve exposure
    
    Transmission channels:
    1. DEX arbitrage: Coin A depeg → arbitrage selling in A/B pool → B pressure
    2. Fire-sale cascade: A sells T-bills → price impact → B's T-bills devalued
    3. Information spillover: A run → increases π for B
    """
    
    def __init__(self, stablecoins: List[Stablecoin], dex_pools: List[DEXPool]):
        self.stablecoins = {coin.name: coin for coin in stablecoins}
        self.dex_pools = dex_pools
        
        # Build adjacency matrix
        self.n_coins = len(stablecoins)
        self.coin_names = list(self.stablecoins.keys())
        self.adjacency = self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build network adjacency matrix from DEX pools"""
        n = self.n_coins
        adj = np.zeros((n, n))
        
        for pool in self.dex_pools:
            i = self.coin_names.index(pool.coin_a)
            j = self.coin_names.index(pool.coin_b)
            
            # Weight by pool size relative to supply
            coin_a = self.stablecoins[pool.coin_a]
            weight = pool.reserve_a / coin_a.supply
            
            adj[i, j] = weight
            adj[j, i] = weight  # Symmetric
        
        return adj
    
    def simulate_shock(self, shock_coin: str, shock_size: float,
                      n_rounds: int = 5) -> Dict:
        """
        Simulate contagion from initial shock to one coin.
        
        Propagates through:
        1. DEX arbitrage (immediate)
        2. Fire-sale cascade (T-bill market)
        3. Information spillover (updated beliefs)
        
        Args:
            shock_coin: Which coin experiences initial shock
            shock_size: Size of shock as fraction of supply
            n_rounds: Number of contagion rounds
        
        Returns:
            Dictionary with time series of depegs, runs, losses
        """
        # Initialize state
        state = {
            'round': [],
            'depegs': {name: [] for name in self.coin_names},
            'redemptions': {name: [] for name in self.coin_names},
            'run_status': {name: [] for name in self.coin_names},
            'tbill_price': [],
        }
        
        # Initial shock
        current_redemptions = {name: 0.0 for name in self.coin_names}
        current_redemptions[shock_coin] = shock_size
        
        # T-bill price starts at par
        tbill_price = 1.0
        
        for round_num in range(n_rounds):
            state['round'].append(round_num)
            
            # 1. Each coin responds to its redemption pressure
            round_tbill_sales = 0.0
            
            for name in self.coin_names:
                coin = self.stablecoins[name]
                R = current_redemptions[name]
                
                # Solve for equilibrium run
                # Adjust pi based on information spillover
                pi_adj = self._adjust_pi_for_spillover(coin, current_redemptions)
                
                n_star, _ = solve_n_star(coin.lambda_tier1, pi_adj, coin.kappa)
                R_equil = pi_adj + n_star * (1 - pi_adj)
                
                # Update redemptions
                current_redemptions[name] = R_equil
                
                # Compute depeg
                depeg_bps = (1 - p_of_R(R_equil, coin.lambda_tier1, coin.kappa)) * 10000
                
                # T-bill sales if fire-sale needed
                if R_equil > coin.lambda_tier1:
                    sales = (R_equil - coin.lambda_tier1) * coin.supply
                    round_tbill_sales += sales
                
                # Record
                state['depegs'][name].append(depeg_bps)
                state['redemptions'][name].append(R_equil * 100)
                state['run_status'][name].append('RUN' if n_star > 0.01 else 'STABLE')
            
            # 2. Fire-sale cascade: T-bill price impact
            # Aggregate T-bill sales affect price
            total_tbill_market = sum(coin.reserves_tbills for coin in self.stablecoins.values())
            
            if total_tbill_market > 0:
                # Price impact from sales
                # Simplified: P = 1 - β·(Sales/Market)
                beta = 0.10  # 10% impact per 100% of market
                impact = beta * round_tbill_sales / total_tbill_market
                tbill_price *= (1 - impact)
                tbill_price = max(0.5, tbill_price)  # Floor at 50% of par
            
            state['tbill_price'].append(tbill_price)
            
            # 3. DEX arbitrage spillovers
            # Redemption pressure creates DEX imbalances → arbitrage flows
            for pool in self.dex_pools:
                # If coin A has redemptions, arbitrageurs sell A for B in pool
                R_a = current_redemptions[pool.coin_a]
                
                if R_a > 0.01:  # Threshold for arbitrage
                    # Arbitrage flow proportional to redemptions
                    arb_flow = R_a * 0.1 * self.stablecoins[pool.coin_a].supply
                    
                    # This creates sell pressure on coin B
                    impact = pool.price_impact(arb_flow)
                    
                    # Increase redemption pressure on B
                    additional_redemption = impact / 1000  # Scale down
                    current_redemptions[pool.coin_b] += additional_redemption
            
            # 4. Check convergence
            if round_num > 0:
                # If redemptions stable, stop early
                prev_total = sum(state['redemptions'][name][-2] for name in self.coin_names)
                curr_total = sum(state['redemptions'][name][-1] for name in self.coin_names)
                
                if abs(curr_total - prev_total) < 0.1:  # <0.1pp change
                    break
        
        # Summary statistics
        state['summary'] = self._compute_summary(state)
        
        return state
    
    def _adjust_pi_for_spillover(self, coin: Stablecoin, 
                                 redemptions: Dict[str, float]) -> float:
        """
        Adjust impatient fraction for information spillover.
        
        If other coins are running, increases fear/coordination.
        """
        pi_base = coin.pi_baseline
        
        # Count how many other coins are running
        other_runs = sum(1 for name, R in redemptions.items() 
                        if name != coin.name and R > 0.15)  # >15% threshold
        
        # Spillover effect: +2pp per running coin
        spillover = other_runs * 0.02
        
        pi_adj = min(pi_base + spillover, 0.50)  # Cap at 50%
        
        return pi_adj
    
    def _compute_summary(self, state: Dict) -> Dict:
        """Compute summary statistics from simulation"""
        summary = {}
        
        for name in self.coin_names:
            final_depeg = state['depegs'][name][-1]
            max_depeg = max(state['depegs'][name])
            final_redemption = state['redemptions'][name][-1]
            
            summary[name] = {
                'final_depeg_bps': final_depeg,
                'max_depeg_bps': max_depeg,
                'final_redemption_pct': final_redemption,
                'experienced_run': final_redemption > 15.0,
            }
        
        # System-wide
        summary['system'] = {
            'total_redemptions_pct': sum(s['final_redemption_pct'] for s in summary.values() if isinstance(s, dict)),
            'coins_running': sum(1 for s in summary.values() if isinstance(s, dict) and s['experienced_run']),
            'final_tbill_price': state['tbill_price'][-1],
        }
        
        return summary
    
    def contagion_matrix(self, shock_size: float = 0.15) -> np.ndarray:
        """
        Compute contagion matrix: C[i,j] = depeg in j when shock to i.
        
        Args:
            shock_size: Size of initial shock (15% = severe)
        
        Returns:
            n × n matrix of contagion effects
        """
        n = self.n_coins
        contagion = np.zeros((n, n))
        
        for i, shock_coin in enumerate(self.coin_names):
            print(f"  Simulating shock to {shock_coin}...")
            
            # Simulate shock
            result = self.simulate_shock(shock_coin, shock_size, n_rounds=5)
            
            # Record final depegs
            for j, target_coin in enumerate(self.coin_names):
                contagion[i, j] = result['summary'][target_coin]['max_depeg_bps']
        
        return contagion
    
    def impulse_response(self, shock_coin: str, shock_size: float,
                        horizon: int = 10) -> Dict:
        """
        Compute impulse response function: response of all coins over time.
        
        Similar to VAR impulse response.
        
        Args:
            shock_coin: Initial shocked coin
            shock_size: Size of shock
            horizon: Number of periods
        
        Returns:
            Dictionary with IRFs
        """
        # Run extended simulation
        result = self.simulate_shock(shock_coin, shock_size, n_rounds=horizon)
        
        # Format as IRF
        irf = {
            'shock_coin': shock_coin,
            'shock_size': shock_size,
            'horizon': horizon,
            'responses': {}
        }
        
        for name in self.coin_names:
            irf['responses'][name] = {
                'depeg_bps': result['depegs'][name],
                'redemption_pct': result['redemptions'][name],
            }
        
        return irf


def create_usdc_dai_usdt_network() -> ContagionNetwork:
    """
    Create realistic network: USDC, DAI, USDT with DEX pools.
    
    Calibrated to March 2023 state.
    """
    # Stablecoins
    usdc = Stablecoin(
        name='USDC',
        supply=43500.0,  # $43.5B (Mar 2023)
        lambda_tier1=0.00,  # Post-SVB: deposits frozen
        pi_baseline=0.115,  # 11.5% peak
        kappa=0.10,
        reserves_tbills=34800.0,  # 80% in T-bills
        reserves_other=8700.0,  # 20% other
    )
    
    dai = Stablecoin(
        name='DAI',
        supply=5200.0,  # $5.2B
        lambda_tier1=0.60,  # 60% USDC in PSM (liquid if USDC stable)
        pi_baseline=0.05,  # Lower baseline (overcollateralized)
        kappa=0.12,
        reserves_tbills=1000.0,  # Some direct T-bill exposure
        reserves_other=4200.0,  # Mostly crypto collateral + USDC
    )
    
    usdt = Stablecoin(
        name='USDT',
        supply=83000.0,  # $83B
        lambda_tier1=0.35,  # 35% liquid (MMFs + short-term)
        pi_baseline=0.08,  # Moderate
        kappa=0.08,  # Deeper market
        reserves_tbills=33200.0,  # 40% T-bills
        reserves_other=49800.0,  # 60% other
    )
    
    # DEX Pools (simplified)
    pools = [
        DEXPool(
            coin_a='USDC',
            coin_b='DAI',
            reserve_a=1500.0,  # $1.5B USDC
            reserve_b=1500.0,  # $1.5B DAI
            amplification=200.0,  # High A (Curve 3pool)
            fee_bps=4.0,
        ),
        DEXPool(
            coin_a='USDC',
            coin_b='USDT',
            reserve_a=500.0,  # $500M USDC
            reserve_b=500.0,  # $500M USDT
            amplification=100.0,
            fee_bps=4.0,
        ),
        DEXPool(
            coin_a='DAI',
            coin_b='USDT',
            reserve_a=300.0,  # $300M DAI
            reserve_b=300.0,  # $300M USDT
            amplification=100.0,
            fee_bps=4.0,
        ),
    ]
    
    return ContagionNetwork([usdc, dai, usdt], pools)


if __name__ == "__main__":
    print("=" * 70)
    print("CROSS-ISSUER CONTAGION: NETWORK MODEL")
    print("=" * 70)
    
    # Create network
    print("\n🌐 Building USDC-DAI-USDT network...")
    network = create_usdc_dai_usdt_network()
    
    print(f"  Nodes: {network.n_coins} stablecoins")
    print(f"  Edges: {len(network.dex_pools)} DEX pools")
    
    print("\n📊 NETWORK STRUCTURE:")
    print(f"  Adjacency matrix:")
    for i, name_i in enumerate(network.coin_names):
        print(f"    {name_i:>5}: ", end="")
        for j, name_j in enumerate(network.coin_names):
            print(f"{network.adjacency[i,j]:.3f}  ", end="")
        print()
    
    # Simulate USDC shock (SVB crisis)
    print("\n💥 SIMULATING USDC SHOCK (SVB crisis, 11.5% redemption)...")
    result = network.simulate_shock('USDC', shock_size=0.115, n_rounds=5)
    
    print(f"\n📈 CONTAGION DYNAMICS:")
    print(f"  {'Round':<8} {'USDC depeg':<15} {'DAI depeg':<15} {'USDT depeg':<15} {'T-bill P':<10}")
    print("  " + "-" * 70)
    
    for i, round_num in enumerate(result['round']):
        print(f"  {round_num:<8} "
              f"{result['depegs']['USDC'][i]:>10.1f}bp    "
              f"{result['depegs']['DAI'][i]:>10.1f}bp    "
              f"{result['depegs']['USDT'][i]:>10.1f}bp    "
              f"{result['tbill_price'][i]:>6.3f}")
    
    print(f"\n🎯 FINAL STATE:")
    for name, stats in result['summary'].items():
        if name == 'system':
            continue
        print(f"  {name}:")
        print(f"    Max depeg: {stats['max_depeg_bps']:.0f}bp")
        print(f"    Final redemptions: {stats['final_redemption_pct']:.1f}%")
        print(f"    Status: {'🔴 RUN' if stats['experienced_run'] else '🟢 STABLE'}")
    
    print(f"\n  System:")
    sys_stats = result['summary']['system']
    print(f"    Total redemptions: {sys_stats['total_redemptions_pct']:.1f}%")
    print(f"    Coins running: {sys_stats['coins_running']}/{network.n_coins}")
    print(f"    T-bill price: {sys_stats['final_tbill_price']:.3f}")
    
    # Contagion matrix
    print(f"\n🔥 CONTAGION MATRIX (15% shock):")
    print(f"  Computing full contagion matrix...")
    contagion = network.contagion_matrix(shock_size=0.15)
    
    print(f"\n  Shock to ↓ \\ Depeg in →  ", end="")
    for name in network.coin_names:
        print(f"{name:>8}", end="")
    print()
    print("  " + "-" * 40)
    
    for i, name_i in enumerate(network.coin_names):
        print(f"  {name_i:<10}               ", end="")
        for j in range(network.n_coins):
            print(f"{contagion[i,j]:>7.0f}bp", end="")
        print()
    
    # Impulse response
    print(f"\n📉 IMPULSE RESPONSE FUNCTION (USDC → DAI):")
    irf = network.impulse_response('USDC', shock_size=0.115, horizon=8)
    
    print(f"  {'Period':<8} {'DAI depeg':<12} {'DAI redemption':<15}")
    print("  " + "-" * 40)
    for t in range(len(irf['responses']['DAI']['depeg_bps'])):
        depeg = irf['responses']['DAI']['depeg_bps'][t]
        redemption = irf['responses']['DAI']['redemption_pct'][t]
        print(f"  {t:<8} {depeg:>8.1f}bp    {redemption:>10.1f}%")
    
    print("\n" + "=" * 70)
    print("Contagion network model ready!")
    print("Key finding: USDC shock causes ~{:.0f}bp DAI depeg (observed: ~500bp)".format(
        result['summary']['DAI']['max_depeg_bps']))

