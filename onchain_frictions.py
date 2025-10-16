"""
On-Chain Friction Models for Stablecoin Runs
============================================

Models rapid redemption dynamics with:
- Block time delays
- Gas price spikes during congestion
- Transaction ordering and MEV
- Liquidity pool mechanics (Curve, Uniswap)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math


@dataclass
class BlockchainParameters:
    """Parameters for on-chain friction model"""
    block_time_seconds: float = 12.0  # Ethereum block time
    base_gas_price_gwei: float = 30.0  # Base gas price
    gas_elasticity: float = 2.0  # How fast gas rises with demand
    max_gas_price_gwei: float = 500.0  # Gas price ceiling
    redemption_gas_units: int = 150_000  # Gas units per redemption
    blocks_per_confirmation: int = 1  # Confirmations needed
    

@dataclass
class LiquidityPoolParameters:
    """Parameters for AMM pool (Curve, Uniswap)"""
    pool_reserve_x: float  # Stablecoin reserve (e.g., USDC)
    pool_reserve_y: float  # Other asset reserve (e.g., DAI, USDT)
    amplification_coeff: float = 100.0  # Curve A parameter (higher = flatter)
    fee_bps: float = 4.0  # Pool fee in basis points (e.g., 4 = 0.04%)
    
    @property
    def fee_rate(self) -> float:
        """Convert fee from bps to decimal"""
        return self.fee_bps / 10000


class OnChainRedemptionModel:
    """
    Model rapid on-chain redemption dynamics with block time and gas frictions.
    
    Tracks:
    - Cumulative redemptions over time
    - Gas price evolution
    - Transaction latency
    - Peg deviation from secondary markets
    """
    
    def __init__(self, blockchain_params: BlockchainParameters,
                 initial_supply: float,
                 redemption_rate_per_second: float):
        """
        Args:
            blockchain_params: Blockchain friction parameters
            initial_supply: Initial stablecoin supply (millions)
            redemption_rate_per_second: Attempted redemptions per second (millions)
        """
        self.params = blockchain_params
        self.initial_supply = initial_supply
        self.redemption_rate = redemption_rate_per_second
        
    def gas_price_at_time(self, t: float, base_demand: float = 1.0) -> float:
        """
        Compute gas price at time t given redemption pressure.
        
        Gas price spikes with congestion:
        gas(t) = base_gas * (1 + demand_multiplier)^elasticity
        
        Args:
            t: Time in seconds
            base_demand: Baseline demand multiplier
        
        Returns:
            Gas price in gwei
        """
        # Redemption intensity as demand proxy
        cumulative_redeemed = min(t * self.redemption_rate, self.initial_supply)
        intensity = cumulative_redeemed / self.initial_supply
        
        # Demand multiplier
        demand = base_demand * (1 + intensity * 10)  # 10x multiplier at full redemption
        
        # Price with elasticity
        gas_price = self.params.base_gas_price_gwei * (demand ** self.params.gas_elasticity)
        
        # Cap at maximum
        gas_price = min(gas_price, self.params.max_gas_price_gwei)
        
        return gas_price
    
    def transaction_cost_usd(self, t: float, eth_price: float = 2000.0) -> float:
        """
        Compute transaction cost in USD at time t.
        
        Args:
            t: Time in seconds
            eth_price: ETH price in USD
        
        Returns:
            Transaction cost in USD
        """
        gas_price_gwei = self.gas_price_at_time(t)
        gas_price_eth = gas_price_gwei * 1e-9  # Convert gwei to ETH
        
        cost_eth = gas_price_eth * self.params.redemption_gas_units
        cost_usd = cost_eth * eth_price
        
        return cost_usd
    
    def cumulative_redeemed(self, t: float) -> float:
        """
        Compute cumulative redemptions by time t (with block time delays).
        
        Args:
            t: Time in seconds
        
        Returns:
            Cumulative redeemed amount (millions USD)
        """
        # Account for block time quantization
        blocks_elapsed = math.floor(t / self.params.block_time_seconds)
        effective_time = blocks_elapsed * self.params.block_time_seconds
        
        # Redemptions processed per block
        per_block = self.redemption_rate * self.params.block_time_seconds
        
        cumulative = blocks_elapsed * per_block
        
        # Cap at total supply
        cumulative = min(cumulative, self.initial_supply)
        
        return cumulative
    
    def latency_to_redemption(self, t: float) -> float:
        """
        Compute latency (seconds) from redemption request to confirmation.
        
        Includes:
        - Mempool wait time (congestion-dependent)
        - Block time
        - Confirmation blocks
        
        Args:
            t: Current time in seconds
        
        Returns:
            Latency in seconds
        """
        # Base latency: block time * confirmations
        base_latency = self.params.block_time_seconds * self.params.blocks_per_confirmation
        
        # Additional mempool congestion delay
        gas_price = self.gas_price_at_time(t)
        congestion_factor = gas_price / self.params.base_gas_price_gwei
        
        # Higher gas → faster inclusion (inversely related)
        # But at extreme congestion, even high gas has delays
        if congestion_factor > 5:
            congestion_delay = (congestion_factor - 5) * 2  # Extra seconds per multiple above 5x
        else:
            congestion_delay = 0
        
        total_latency = base_latency + congestion_delay
        
        return total_latency
    
    def simulate_redemption_wave(self, duration_seconds: float,
                                 time_step: float = 1.0) -> Dict:
        """
        Simulate redemption wave over time period.
        
        Args:
            duration_seconds: Total simulation time
            time_step: Time step for simulation
        
        Returns:
            Dictionary with time series data
        """
        times = np.arange(0, duration_seconds + time_step, time_step)
        
        results = {
            'time': times,
            'cumulative_redeemed': np.zeros(len(times)),
            'gas_price_gwei': np.zeros(len(times)),
            'transaction_cost_usd': np.zeros(len(times)),
            'latency_seconds': np.zeros(len(times)),
            'redemption_rate': np.zeros(len(times)),
        }
        
        for i, t in enumerate(times):
            results['cumulative_redeemed'][i] = self.cumulative_redeemed(t)
            results['gas_price_gwei'][i] = self.gas_price_at_time(t)
            results['transaction_cost_usd'][i] = self.transaction_cost_usd(t)
            results['latency_seconds'][i] = self.latency_to_redemption(t)
            
            # Instantaneous redemption rate (derivative)
            if i > 0:
                dt = times[i] - times[i-1]
                dredeemed = results['cumulative_redeemed'][i] - results['cumulative_redeemed'][i-1]
                results['redemption_rate'][i] = dredeemed / dt if dt > 0 else 0
            else:
                results['redemption_rate'][i] = self.redemption_rate
        
        return results


class CurvePoolModel:
    """
    Model Curve StableSwap pool for stablecoin secondary market pricing.
    
    Implements Curve invariant:
    A*n^n*sum(x_i) + D = A*D*n^n + D^(n+1)/(n^n*prod(x_i))
    
    For 2-asset pool (n=2): simplified to standard StableSwap formula
    """
    
    def __init__(self, params: LiquidityPoolParameters):
        self.params = params
        
    def compute_d(self) -> float:
        """
        Compute D (total liquidity invariant) for current reserves.
        
        Returns:
            D value
        """
        x = self.params.pool_reserve_x
        y = self.params.pool_reserve_y
        A = self.params.amplification_coeff
        
        # Simplified for 2-asset pool
        # D ≈ x + y for balanced pools with high A
        S = x + y
        
        # Newton's method iteration (simplified)
        D = S
        for _ in range(10):  # Converge
            D_P = D * D / (2 * x * y)
            D_prev = D
            D = (A * S + 2 * D_P) * D / ((A - 1) * D + 3 * D_P)
            
            if abs(D - D_prev) < 1e-6:
                break
        
        return D
    
    def get_output_amount(self, input_amount: float, 
                         sell_x: bool = True) -> Tuple[float, float]:
        """
        Compute output amount for given input (including fees).
        
        Args:
            input_amount: Amount of input token
            sell_x: If True, sell X for Y; if False, sell Y for X
        
        Returns:
            (output_amount, effective_price)
        """
        fee = self.params.fee_rate
        
        # Apply fee to input
        input_after_fee = input_amount * (1 - fee)
        
        if sell_x:
            # Selling X (stablecoin) for Y
            x_new = self.params.pool_reserve_x + input_after_fee
            
            # Compute new Y from invariant
            D = self.compute_d()
            A = self.params.amplification_coeff
            
            # Solve for y: A*x*y + y = A*D - (D^2)/(2*x)
            # Quadratic formula
            b = A * x_new + D / 2
            c = D * D / (4 * x_new)
            
            y_new = (-b + math.sqrt(b*b + 4*c)) / 2
            
            output_amount = self.params.pool_reserve_y - y_new
            
        else:
            # Selling Y for X (stablecoin)
            y_new = self.params.pool_reserve_y + input_after_fee
            
            D = self.compute_d()
            A = self.params.amplification_coeff
            
            b = A * y_new + D / 2
            c = D * D / (4 * y_new)
            
            x_new = (-b + math.sqrt(b*b + 4*c)) / 2
            
            output_amount = self.params.pool_reserve_x - x_new
        
        # Effective price
        effective_price = output_amount / input_amount if input_amount > 0 else 1.0
        
        return output_amount, effective_price
    
    def price_impact(self, trade_size: float, sell_x: bool = True) -> float:
        """
        Compute price impact (deviation from peg) for trade.
        
        Args:
            trade_size: Size of trade
            sell_x: Direction of trade
        
        Returns:
            Price impact as percentage deviation from 1.0
        """
        output, price = self.get_output_amount(trade_size, sell_x)
        
        # Impact: deviation from 1:1 peg
        impact_pct = (1.0 - price) * 100
        
        return impact_pct
    
    def simulate_redemption_pressure(self, redemption_amounts: np.ndarray) -> Dict:
        """
        Simulate cumulative pool imbalance from redemption wave.
        
        As users redeem on-chain, arbitrageurs sell redeemed stablecoin to pool.
        
        Args:
            redemption_amounts: Array of cumulative redemption amounts
        
        Returns:
            Dictionary with pool state over time
        """
        results = {
            'cumulative_redeemed': redemption_amounts,
            'pool_x': np.zeros(len(redemption_amounts)),
            'pool_y': np.zeros(len(redemption_amounts)),
            'peg_price': np.zeros(len(redemption_amounts)),
            'discount_pct': np.zeros(len(redemption_amounts)),
        }
        
        # Initial state
        pool_x = self.params.pool_reserve_x
        pool_y = self.params.pool_reserve_y
        
        for i, redeemed in enumerate(redemption_amounts):
            # Arbitrage flow: redeemed stablecoins sold to pool
            # Simplified: assume fraction of redemptions hit pool
            arb_flow = redeemed * 0.1  # 10% of redemptions flow through pool
            
            if i > 0:
                incremental = arb_flow - (redemption_amounts[i-1] * 0.1 if i > 0 else 0)
                
                if incremental > 0 and pool_y > incremental:
                    # Sell X for Y
                    pool_params = LiquidityPoolParameters(
                        pool_reserve_x=pool_x,
                        pool_reserve_y=pool_y,
                        amplification_coeff=self.params.amplification_coeff,
                        fee_bps=self.params.fee_bps,
                    )
                    
                    pool_model = CurvePoolModel(pool_params)
                    output, price = pool_model.get_output_amount(incremental, sell_x=True)
                    
                    # Update reserves
                    pool_x += incremental
                    pool_y -= output
            
            # Record state
            results['pool_x'][i] = pool_x
            results['pool_y'][i] = pool_y
            
            # Current peg price (marginal)
            if pool_x > 0 and pool_y > 0:
                pool_params = LiquidityPoolParameters(
                    pool_reserve_x=pool_x,
                    pool_reserve_y=pool_y,
                    amplification_coeff=self.params.amplification_coeff,
                    fee_bps=self.params.fee_bps,
                )
                pool_model = CurvePoolModel(pool_params)
                
                # Small trade to get marginal price
                _, price = pool_model.get_output_amount(1.0, sell_x=True)
                results['peg_price'][i] = price
                results['discount_pct'][i] = (1.0 - price) * 100
            else:
                results['peg_price'][i] = 0.0
                results['discount_pct'][i] = 100.0
        
        return results


def analyze_rapid_run(initial_supply: float = 1000.0,
                     peak_redemption_rate_pct: float = 50.0,
                     duration_minutes: float = 2.0) -> Dict:
    """
    Analyze rapid run scenario with on-chain frictions.
    
    Args:
        initial_supply: Initial stablecoin supply (millions)
        peak_redemption_rate_pct: Peak redemption as % of supply
        duration_minutes: Duration to analyze
    
    Returns:
        Dictionary with comprehensive analysis
    """
    # Parameters
    blockchain_params = BlockchainParameters(
        block_time_seconds=12.0,
        base_gas_price_gwei=30.0,
        gas_elasticity=2.0,
        max_gas_price_gwei=500.0,
        redemption_gas_units=150_000,
        blocks_per_confirmation=1,
    )
    
    # Redemption rate
    peak_amount = initial_supply * peak_redemption_rate_pct / 100
    duration_seconds = duration_minutes * 60
    redemption_rate_per_second = peak_amount / duration_seconds
    
    # On-chain model
    onchain_model = OnChainRedemptionModel(
        blockchain_params,
        initial_supply,
        redemption_rate_per_second
    )
    
    # Simulate
    onchain_results = onchain_model.simulate_redemption_wave(
        duration_seconds=duration_seconds,
        time_step=12.0  # One block
    )
    
    # Pool model for secondary market
    pool_params = LiquidityPoolParameters(
        pool_reserve_x=500.0,  # $500M stablecoin
        pool_reserve_y=500.0,  # $500M other stables
        amplification_coeff=100.0,
        fee_bps=4.0,
    )
    
    pool_model = CurvePoolModel(pool_params)
    pool_results = pool_model.simulate_redemption_pressure(
        onchain_results['cumulative_redeemed']
    )
    
    # Combine results
    combined = {
        'onchain': onchain_results,
        'pool': pool_results,
        'summary': {
            'total_redeemed': onchain_results['cumulative_redeemed'][-1],
            'redemption_pct': (onchain_results['cumulative_redeemed'][-1] / initial_supply * 100),
            'peak_gas_gwei': np.max(onchain_results['gas_price_gwei']),
            'peak_discount_pct': np.max(pool_results['discount_pct']),
            'total_latency': onchain_results['latency_seconds'][-1],
            'gas_spike_multiple': np.max(onchain_results['gas_price_gwei']) / blockchain_params.base_gas_price_gwei,
        }
    }
    
    return combined


if __name__ == "__main__":
    print("=" * 70)
    print("ON-CHAIN FRICTION MODEL")
    print("Rapid Redemption Analysis")
    print("=" * 70)
    
    # Analyze rapid run scenario
    results = analyze_rapid_run(
        initial_supply=1000.0,
        peak_redemption_rate_pct=11.0,  # USDC SVB peak
        duration_minutes=2.0
    )
    
    summary = results['summary']
    
    print("\n📊 RAPID RUN SCENARIO")
    print(f"Total Redeemed: ${summary['total_redeemed']:.1f}M ({summary['redemption_pct']:.1f}%)")
    print(f"Peak Gas Price: {summary['peak_gas_gwei']:.0f} gwei ({summary['gas_spike_multiple']:.1f}x base)")
    print(f"Peak Peg Discount: {summary['peak_discount_pct']:.2f}%")
    print(f"Total Latency: {summary['total_latency']:.0f} seconds")
    
    print("\n⛽ GAS DYNAMICS")
    onchain = results['onchain']
    for i in [0, len(onchain['time'])//4, len(onchain['time'])//2, -1]:
        t = onchain['time'][i]
        gas = onchain['gas_price_gwei'][i]
        cost = onchain['transaction_cost_usd'][i]
        print(f"  t={t:3.0f}s: {gas:5.0f} gwei (${cost:.2f} per redemption)")
    
    print("\n💱 POOL IMPACT")
    pool = results['pool']
    for i in [0, len(pool['cumulative_redeemed'])//2, -1]:
        redeemed = pool['cumulative_redeemed'][i]
        discount = pool['discount_pct'][i]
        print(f"  Redeemed ${redeemed:5.1f}M → {discount:.3f}% discount")
    
    print("\n" + "=" * 70)
    print("On-chain friction models implemented successfully!")

