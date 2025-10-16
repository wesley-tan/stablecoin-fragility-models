"""
Stablecoin Robust Fragility Research Models
============================================

Implementation of Diamond-Dybvig coordination game with cash-in-market fire-sale 
pricing (Izumi-Li framework) for stablecoin run analysis.

Core equations:
- Fire-sale price: P = P₀ * (1 - κ*Q/C)
- Run threshold: P*(R+C) < V
- LCR: HQLA / NetCashOutflows₃₀d ≥ 100%

Calibrated to USDC/Circle Reserve Fund and SVB crisis (March 2023).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings


class AssetClass(Enum):
    """Asset classes for stablecoin reserves"""
    TBILLS = "T-bills"
    DEPOSITS = "Bank Deposits"
    REPO = "Repo"
    MMF = "Money Market Funds"


@dataclass
class ReserveAsset:
    """Individual reserve asset with liquidity characteristics"""
    asset_class: AssetClass
    amount: float  # in millions USD
    yield_rate: float  # annual yield (e.g., 0.0465 for 4.65%)
    fire_sale_haircut_bps: float  # haircut in basis points (e.g., 2 for 0.02%)
    liquidity_weight: float  # HQLA weight (1.0 = full liquid, 0 = illiquid)
    
    @property
    def fire_sale_haircut(self) -> float:
        """Convert haircut from bps to decimal"""
        return self.fire_sale_haircut_bps / 10000
    
    @property
    def hqla_value(self) -> float:
        """High Quality Liquid Asset value for LCR calculation"""
        return self.amount * self.liquidity_weight


@dataclass
class StablecoinParameters:
    """Core parameters for the stablecoin model"""
    # Liabilities
    total_supply: float  # Total stablecoin supply (millions USD)
    
    # Reserve composition
    reserves: List[ReserveAsset]
    
    # Fire-sale parameters
    market_impact_coeff: float = 0.05  # κ in fire-sale pricing (5% impact per $1B)
    investor_cash: float = None  # C in Izumi-Li model (auto-computed if None)
    
    # Run dynamics
    redemption_shock: float = 0.0  # Fraction attempting to redeem
    block_time_seconds: float = 12.0  # On-chain block time
    base_gas_price_gwei: float = 30.0  # Base gas price
    
    # Coordination game parameters
    noise_std: float = 0.05  # Standard deviation of private signals (lognormal)
    
    # Policy parameters
    psm_buffer: float = 0.0  # Peg Stability Module buffer (millions USD)
    redemption_gate_cost: float = 0.0  # Additional cost to redeem (fraction)
    lcr_floor: float = 1.0  # Minimum LCR ratio (1.0 = 100%)
    
    def __post_init__(self):
        """Compute derived parameters"""
        if self.investor_cash is None:
            # Default: investor cash = total T-bill reserves (Izumi-Li assumption)
            self.investor_cash = sum(
                r.amount for r in self.reserves 
                if r.asset_class == AssetClass.TBILLS
            )
    
    @property
    def total_reserves(self) -> float:
        """Total reserve value"""
        return sum(r.amount for r in self.reserves)
    
    @property
    def reserve_ratio(self) -> float:
        """Reserve ratio (reserves / supply)"""
        return self.total_reserves / self.total_supply if self.total_supply > 0 else 0
    
    @property
    def total_hqla(self) -> float:
        """Total High Quality Liquid Assets for LCR"""
        return sum(r.hqla_value for r in self.reserves)
    
    def get_asset_allocation(self) -> Dict[AssetClass, float]:
        """Get allocation by asset class"""
        allocation = {}
        for reserve in self.reserves:
            if reserve.asset_class in allocation:
                allocation[reserve.asset_class] += reserve.amount
            else:
                allocation[reserve.asset_class] = reserve.amount
        return allocation


class StablecoinFragilityModel:
    """
    Main model implementing Diamond-Dybvig coordination with Izumi-Li fire-sales.
    
    Analyzes:
    - Run thresholds based on reserve composition
    - Fire-sale pricing and market impact
    - LCR-style liquidity coverage
    - Robust fragility (max sunspot probability)
    """
    
    def __init__(self, params: StablecoinParameters):
        self.params = params
    
    def fire_sale_price(self, forced_sales: float, 
                       cash_available: Optional[float] = None) -> float:
        """
        Compute fire-sale price with market impact.
        
        P = P₀ * (1 - κ*Q/C)
        
        Args:
            forced_sales: Q = volume of forced sales (millions USD)
            cash_available: C = investor cash in market (millions USD)
        
        Returns:
            Price as fraction of par (1.0 = no discount)
        """
        if cash_available is None:
            cash_available = self.params.investor_cash
        
        if cash_available == 0:
            return 0.0  # No market
        
        # Market impact
        kappa = self.params.market_impact_coeff
        impact = kappa * forced_sales / cash_available
        
        # Price cannot go negative
        price = max(0.0, 1.0 - impact)
        
        return price
    
    def asset_specific_haircut(self, asset: ReserveAsset, 
                               volume: float) -> float:
        """
        Compute realized price after asset-specific haircut and market impact.
        
        Args:
            asset: Reserve asset being liquidated
            volume: Volume being sold
        
        Returns:
            Effective price as fraction of par
        """
        # Start with asset-specific haircut
        base_price = 1.0 - asset.fire_sale_haircut
        
        # Apply market impact if selling
        if volume > 0:
            fire_sale_discount = self.params.market_impact_coeff * volume / self.params.investor_cash
            base_price *= (1.0 - fire_sale_discount)
        
        return max(0.0, base_price)
    
    def compute_run_threshold(self) -> Tuple[float, Dict]:
        """
        Compute run threshold using patient indifference condition.
        
        Patient indifference: E[payoff wait] = E[payoff run]
        
        With fire-sales: P*(R+C) < V → Run occurs
        
        Returns:
            (run_cutoff_bps, diagnostics_dict)
        """
        R = self.params.total_reserves
        C = self.params.investor_cash
        V = self.params.total_supply
        PSM = self.params.psm_buffer
        
        # Assume redemption shock determines forced sales
        forced_sales = self.params.redemption_shock * V
        
        # Compute fire-sale price
        P = self.fire_sale_price(forced_sales, C)
        
        # Left side of inequality: P*(R+C+PSM)
        available_liquidity = P * (R + C + PSM)
        
        # Run occurs if available_liquidity < V
        # Run cutoff is the indifference point
        # For threshold calculation, find P such that P*(R+C+PSM) = V
        
        if R + C + PSM > 0:
            critical_price = V / (R + C + PSM)
        else:
            critical_price = float('inf')
        
        # Convert to basis points (deviation from par)
        # If P = 0.9, discount is 10% = 1000 bps
        # If P = 1.0, discount is 0% = 0 bps
        if critical_price <= 1.0:
            discount_bps = (1.0 - critical_price) * 10000
        else:
            # Liquidity insufficient even at par
            discount_bps = 0  # Instant run
        
        # Run cutoff in bps (higher = more vulnerable)
        # This represents the threshold signal that triggers runs
        run_cutoff_bps = discount_bps
        
        # Fire-sale impact at current shock level
        fire_sale_impact_bps = (1.0 - P) * 10000
        
        diagnostics = {
            'available_liquidity': available_liquidity,
            'total_liabilities': V,
            'critical_price': critical_price,
            'current_fire_sale_price': P,
            'fire_sale_impact_bps': fire_sale_impact_bps,
            'forced_sales': forced_sales,
            'liquidity_gap': V - available_liquidity,
            'reserve_ratio': R / V if V > 0 else 0,
        }
        
        return run_cutoff_bps, diagnostics
    
    def compute_lcr(self, stress_scenario_days: int = 30) -> Tuple[float, Dict]:
        """
        Compute Liquidity Coverage Ratio (Basel III style).
        
        LCR = HQLA / Net Cash Outflows (30 days) ≥ 100%
        
        Args:
            stress_scenario_days: Stress period (default 30 days)
        
        Returns:
            (lcr_ratio, diagnostics_dict)
        """
        # Total High Quality Liquid Assets
        hqla = self.params.total_hqla
        
        # Assume stress: X% of supply redeems over 30 days
        # Use redemption shock as daily rate
        daily_outflow_rate = self.params.redemption_shock
        stress_outflow = daily_outflow_rate * self.params.total_supply * stress_scenario_days
        
        # Net cash outflows (simplified: no inflows assumed in stress)
        net_cash_outflows = max(stress_outflow, self.params.total_supply * 0.1)  # Min 10% assumption
        
        # LCR ratio
        if net_cash_outflows > 0:
            lcr = hqla / net_cash_outflows
        else:
            lcr = float('inf')
        
        # Gap to required floor
        required_hqla = self.params.lcr_floor * net_cash_outflows
        gap = hqla - required_hqla
        
        diagnostics = {
            'hqla': hqla,
            'net_cash_outflows_30d': net_cash_outflows,
            'required_hqla': required_hqla,
            'gap': gap,
            'meets_requirement': lcr >= self.params.lcr_floor,
        }
        
        return lcr, diagnostics
    
    def run_probability(self, sunspot_strength: float = 1.0) -> float:
        """
        Compute run probability using global games framework.
        
        Probability depends on:
        - Distance from run threshold
        - Noise in private signals
        - Sunspot coordination
        
        Args:
            sunspot_strength: Multiplier on coordination (1.0 = baseline)
        
        Returns:
            Probability of run occurring [0, 1]
        """
        run_cutoff_bps, _ = self.compute_run_threshold()
        
        # Current fundamentals (reserve ratio as proxy)
        fundamental_bps = (1.0 - self.params.reserve_ratio) * 10000
        
        # Distance to threshold
        distance = run_cutoff_bps - fundamental_bps
        
        # Probability via normal CDF (global games)
        # More noise → higher probability at given distance
        noise = self.params.noise_std * 10000  # Convert to bps
        
        if noise > 0:
            z_score = -distance / noise * sunspot_strength
            # Standard normal CDF approximation
            prob = 0.5 * (1 + np.tanh(z_score / np.sqrt(2)))
        else:
            # No noise: deterministic threshold
            prob = 1.0 if distance <= 0 else 0.0
        
        return np.clip(prob, 0.0, 1.0)
    
    def max_run_probability(self, backstop_size: float = 0.0) -> float:
        """
        Robust fragility: maximum run probability under worst-case sunspot.
        
        Izumi-Li framework with PSM/backstop.
        
        Args:
            backstop_size: PSM buffer size (millions USD)
        
        Returns:
            Maximum run probability [0, 1]
        """
        # Temporarily add backstop
        original_psm = self.params.psm_buffer
        self.params.psm_buffer = backstop_size
        
        # Worst-case coordination (high sunspot)
        max_prob = self.run_probability(sunspot_strength=2.0)
        
        # Restore original
        self.params.psm_buffer = original_psm
        
        return max_prob
    
    def expected_loss(self) -> Tuple[float, Dict]:
        """
        Compute expected loss to patient holders in run scenario.
        
        Returns:
            (expected_loss_millions, breakdown_dict)
        """
        run_prob = self.run_probability()
        
        # If run occurs, compute losses from fire-sales
        forced_sales = self.params.redemption_shock * self.params.total_supply
        
        # Liquidation waterfall
        total_loss = 0.0
        liquidation_schedule = []
        
        remaining_to_sell = forced_sales
        for asset in sorted(self.params.reserves, 
                          key=lambda a: a.fire_sale_haircut_bps):
            if remaining_to_sell <= 0:
                break
            
            # Sell up to available amount
            sell_amount = min(asset.amount, remaining_to_sell)
            
            # Compute loss from haircut
            haircut_loss = sell_amount * asset.fire_sale_haircut
            
            # Market impact loss
            if self.params.investor_cash > 0:
                market_impact = self.params.market_impact_coeff * sell_amount / self.params.investor_cash
                market_loss = sell_amount * market_impact
            else:
                market_impact = 0.0
                market_loss = 0.0
            
            asset_total_loss = haircut_loss + market_loss
            total_loss += asset_total_loss
            
            liquidation_schedule.append({
                'asset': asset.asset_class.value,
                'amount_sold': sell_amount,
                'haircut_loss': haircut_loss,
                'market_impact_loss': market_loss,
                'total_loss': asset_total_loss,
            })
            
            remaining_to_sell -= sell_amount
        
        # Expected loss = probability × loss
        expected_loss = run_prob * total_loss
        
        diagnostics = {
            'run_probability': run_prob,
            'total_loss_if_run': total_loss,
            'expected_loss': expected_loss,
            'liquidation_schedule': liquidation_schedule,
            'loss_as_pct_supply': (total_loss / self.params.total_supply * 100) if self.params.total_supply > 0 else 0,
        }
        
        return expected_loss, diagnostics
    
    def peg_deviation(self, shock_amount: float) -> float:
        """
        Compute peg deviation (discount from $1.00) under redemption shock.
        
        Args:
            shock_amount: Redemption amount (millions USD)
        
        Returns:
            Peg discount as percentage (e.g., 0.5 for 0.5% depeg)
        """
        # Fire-sale price determines peg
        P = self.fire_sale_price(shock_amount)
        
        # Deviation from par
        deviation_pct = (1.0 - P) * 100
        
        return deviation_pct
    
    def critical_redemption_rate(self) -> float:
        """
        Find critical redemption rate that triggers run.
        
        Returns:
            Critical rate as percentage of supply
        """
        # Binary search for threshold
        low, high = 0.0, 1.0
        tolerance = 0.001
        
        original_shock = self.params.redemption_shock
        
        while high - low > tolerance:
            mid = (low + high) / 2
            self.params.redemption_shock = mid
            
            run_cutoff_bps, diag = self.compute_run_threshold()
            
            # Check if run occurs
            if diag['liquidity_gap'] > 0:
                # Run happens - try lower shock
                high = mid
            else:
                # No run - try higher shock
                low = mid
        
        critical_rate = (low + high) / 2
        
        # Restore original
        self.params.redemption_shock = original_shock
        
        return critical_rate * 100  # Return as percentage


def create_usdc_baseline() -> StablecoinParameters:
    """
    Create baseline calibration based on USDC/Circle Reserve Fund (March 2023).
    
    Circle Reserve Fund (Feb 2023): $43.5B total
    - T-bills: ~80%
    - Bank deposits: ~20% (including $3.3B at SVB)
    
    Simplified $1B model for analysis.
    """
    reserves = [
        ReserveAsset(
            asset_class=AssetClass.TBILLS,
            amount=600.0,  # $600M
            yield_rate=0.0465,  # 4.65% (3-month T-bill, Mar 2023)
            fire_sale_haircut_bps=2,  # 2 bps (highly liquid)
            liquidity_weight=1.0,  # 100% HQLA
        ),
        ReserveAsset(
            asset_class=AssetClass.DEPOSITS,
            amount=200.0,  # $200M
            yield_rate=0.035,  # 3.5% (deposit rate)
            fire_sale_haircut_bps=500,  # 500 bps (SVB scenario)
            liquidity_weight=0.0,  # Not HQLA in stress
        ),
        ReserveAsset(
            asset_class=AssetClass.REPO,
            amount=100.0,  # $100M
            yield_rate=0.048,  # 4.8% (repo rate)
            fire_sale_haircut_bps=10,  # 10 bps
            liquidity_weight=0.85,  # 85% HQLA
        ),
        ReserveAsset(
            asset_class=AssetClass.MMF,
            amount=100.0,  # $100M
            yield_rate=0.0442,  # 4.42% (Prime MMF)
            fire_sale_haircut_bps=10,  # 10 bps
            liquidity_weight=0.95,  # 95% HQLA
        ),
    ]
    
    return StablecoinParameters(
        total_supply=1000.0,  # $1B
        reserves=reserves,
        market_impact_coeff=0.05,  # 5% impact per $1B
        redemption_shock=0.11,  # 11% daily shock (SVB peak)
        noise_std=0.05,  # 5% signal noise
        psm_buffer=0.0,  # No PSM initially
        redemption_gate_cost=0.0,  # No gates
        lcr_floor=1.0,  # 100% LCR minimum
    )


def analyze_reserve_mix_sensitivity(base_params: StablecoinParameters,
                                   tbill_shares: np.ndarray) -> Dict:
    """
    Analyze how run cutoff varies with T-bill share of reserves.
    
    Args:
        base_params: Baseline parameters
        tbill_shares: Array of T-bill shares to test (0 to 1)
    
    Returns:
        Dictionary with results for each share level
    """
    results = {
        'tbill_share_pct': [],
        'run_cutoff_bps': [],
        'fire_sale_impact_bps': [],
        'lcr_ratio': [],
        'run_probability': [],
    }
    
    total_reserves = base_params.total_reserves
    
    for tbill_share in tbill_shares:
        # Adjust reserve mix
        tbill_amount = tbill_share * total_reserves
        other_amount = (1 - tbill_share) * total_reserves
        
        # Distribute other amount across deposits, repo, MMF equally
        other_per_class = other_amount / 3
        
        adjusted_reserves = [
            ReserveAsset(
                asset_class=AssetClass.TBILLS,
                amount=tbill_amount,
                yield_rate=0.0465,
                fire_sale_haircut_bps=2,
                liquidity_weight=1.0,
            ),
            ReserveAsset(
                asset_class=AssetClass.DEPOSITS,
                amount=other_per_class,
                yield_rate=0.035,
                fire_sale_haircut_bps=500,
                liquidity_weight=0.0,
            ),
            ReserveAsset(
                asset_class=AssetClass.REPO,
                amount=other_per_class,
                yield_rate=0.048,
                fire_sale_haircut_bps=10,
                liquidity_weight=0.85,
            ),
            ReserveAsset(
                asset_class=AssetClass.MMF,
                amount=other_per_class,
                yield_rate=0.0442,
                fire_sale_haircut_bps=10,
                liquidity_weight=0.95,
            ),
        ]
        
        # Create adjusted parameters
        params = StablecoinParameters(
            total_supply=base_params.total_supply,
            reserves=adjusted_reserves,
            market_impact_coeff=base_params.market_impact_coeff,
            redemption_shock=base_params.redemption_shock,
            noise_std=base_params.noise_std,
            psm_buffer=base_params.psm_buffer,
            redemption_gate_cost=base_params.redemption_gate_cost,
            lcr_floor=base_params.lcr_floor,
        )
        
        # Run analysis
        model = StablecoinFragilityModel(params)
        run_cutoff_bps, diag = model.compute_run_threshold()
        lcr, _ = model.compute_lcr()
        run_prob = model.run_probability()
        
        results['tbill_share_pct'].append(tbill_share * 100)
        results['run_cutoff_bps'].append(run_cutoff_bps)
        results['fire_sale_impact_bps'].append(diag['fire_sale_impact_bps'])
        results['lcr_ratio'].append(lcr * 100)
        results['run_probability'].append(run_prob * 100)
    
    return results


def analyze_lcr_floors(base_params: StablecoinParameters,
                      lcr_floors: np.ndarray) -> Dict:
    """
    Analyze impact of LCR floor requirements on fragility.
    
    Args:
        base_params: Baseline parameters
        lcr_floors: Array of LCR floor values to test (e.g., 1.0, 1.5, 2.0)
    
    Returns:
        Dictionary with results for each floor level
    """
    results = {
        'lcr_floor_pct': [],
        'actual_lcr_pct': [],
        'run_probability_pct': [],
        'hqla': [],
        'max_sunspot_prob_pct': [],
    }
    
    for lcr_floor in lcr_floors:
        # Adjust parameters
        params = StablecoinParameters(
            total_supply=base_params.total_supply,
            reserves=base_params.reserves.copy(),
            market_impact_coeff=base_params.market_impact_coeff,
            redemption_shock=base_params.redemption_shock,
            noise_std=base_params.noise_std,
            psm_buffer=base_params.psm_buffer,
            redemption_gate_cost=base_params.redemption_gate_cost,
            lcr_floor=lcr_floor,
        )
        
        model = StablecoinFragilityModel(params)
        lcr, lcr_diag = model.compute_lcr()
        run_prob = model.run_probability()
        max_sunspot = model.max_run_probability()
        
        results['lcr_floor_pct'].append(lcr_floor * 100)
        results['actual_lcr_pct'].append(lcr * 100)
        results['run_probability_pct'].append(run_prob * 100)
        results['hqla'].append(lcr_diag['hqla'])
        results['max_sunspot_prob_pct'].append(max_sunspot * 100)
    
    return results


def analyze_backstop_fragility(base_params: StablecoinParameters,
                               backstop_sizes: np.ndarray) -> Dict:
    """
    Analyze robust fragility vs PSM/backstop size.
    
    Args:
        base_params: Baseline parameters
        backstop_sizes: Array of backstop sizes to test (millions USD)
    
    Returns:
        Dictionary with results for each backstop size
    """
    results = {
        'backstop_size': [],
        'max_run_probability_pct': [],
        'critical_threshold_pct': [],
        'expected_loss': [],
    }
    
    model = StablecoinFragilityModel(base_params)
    
    for backstop in backstop_sizes:
        max_prob = model.max_run_probability(backstop_size=backstop)
        
        # Critical threshold with backstop
        params_with_backstop = StablecoinParameters(
            total_supply=base_params.total_supply,
            reserves=base_params.reserves.copy(),
            market_impact_coeff=base_params.market_impact_coeff,
            redemption_shock=base_params.redemption_shock,
            noise_std=base_params.noise_std,
            psm_buffer=backstop,
            redemption_gate_cost=base_params.redemption_gate_cost,
            lcr_floor=base_params.lcr_floor,
        )
        
        model_with_backstop = StablecoinFragilityModel(params_with_backstop)
        critical = model_with_backstop.critical_redemption_rate()
        exp_loss, _ = model_with_backstop.expected_loss()
        
        results['backstop_size'].append(backstop)
        results['max_run_probability_pct'].append(max_prob * 100)
        results['critical_threshold_pct'].append(critical)
        results['expected_loss'].append(exp_loss)
    
    return results


def policy_counterfactuals(base_params: StablecoinParameters) -> Dict:
    """
    Compare policy interventions: more T-bills, gates, PSM, LCR floors.
    
    Returns:
        Dictionary with results for each policy scenario
    """
    scenarios = {}
    
    # 1. Current mix (baseline)
    model_baseline = StablecoinFragilityModel(base_params)
    exp_loss_base, diag_base = model_baseline.expected_loss()
    lcr_base, _ = model_baseline.compute_lcr()
    peg_spread_base = model_baseline.peg_deviation(
        base_params.redemption_shock * base_params.total_supply
    )
    
    scenarios['Current Mix'] = {
        'expected_loss': exp_loss_base,
        'lcr_pct': lcr_base * 100,
        'peg_spread_pct': peg_spread_base,
        'run_probability': diag_base['run_probability'] * 100,
    }
    
    # 2. More T-bills (+50% shift from deposits)
    total_reserves = base_params.total_reserves
    tbill_amount = 600 * 1.5  # +50%
    deposit_amount = max(0, 200 - 300)  # Reduce by 300
    additional_tbills = 300  # Make up difference
    
    reserves_more_tbills = [
        ReserveAsset(AssetClass.TBILLS, 900.0, 0.0465, 2, 1.0),
        ReserveAsset(AssetClass.DEPOSITS, 0.0, 0.035, 500, 0.0),
        ReserveAsset(AssetClass.REPO, 50.0, 0.048, 10, 0.85),
        ReserveAsset(AssetClass.MMF, 50.0, 0.0442, 10, 0.95),
    ]
    
    params_more_tbills = StablecoinParameters(
        total_supply=base_params.total_supply,
        reserves=reserves_more_tbills,
        market_impact_coeff=base_params.market_impact_coeff,
        redemption_shock=base_params.redemption_shock,
        noise_std=base_params.noise_std,
    )
    
    model_tbills = StablecoinFragilityModel(params_more_tbills)
    exp_loss_tbills, diag_tbills = model_tbills.expected_loss()
    lcr_tbills, _ = model_tbills.compute_lcr()
    peg_spread_tbills = model_tbills.peg_deviation(
        base_params.redemption_shock * base_params.total_supply
    )
    
    scenarios['More T-bills'] = {
        'expected_loss': exp_loss_tbills,
        'lcr_pct': lcr_tbills * 100,
        'peg_spread_pct': peg_spread_tbills,
        'run_probability': diag_tbills['run_probability'] * 100,
    }
    
    # 3. Add redemption gates (+50% cost)
    params_gates = StablecoinParameters(
        total_supply=base_params.total_supply,
        reserves=base_params.reserves.copy(),
        market_impact_coeff=base_params.market_impact_coeff,
        redemption_shock=base_params.redemption_shock * 0.5,  # Gates reduce shock
        noise_std=base_params.noise_std,
        redemption_gate_cost=0.005,  # 0.5% cost
    )
    
    model_gates = StablecoinFragilityModel(params_gates)
    exp_loss_gates, diag_gates = model_gates.expected_loss()
    lcr_gates, _ = model_gates.compute_lcr()
    peg_spread_gates = model_gates.peg_deviation(
        params_gates.redemption_shock * base_params.total_supply
    )
    
    scenarios['Add Gates'] = {
        'expected_loss': exp_loss_gates,
        'lcr_pct': lcr_gates * 100,
        'peg_spread_pct': peg_spread_gates,
        'run_probability': diag_gates['run_probability'] * 100,
    }
    
    # 4. Add PSM (+5% buffer)
    params_psm = StablecoinParameters(
        total_supply=base_params.total_supply,
        reserves=base_params.reserves.copy(),
        market_impact_coeff=base_params.market_impact_coeff,
        redemption_shock=base_params.redemption_shock,
        noise_std=base_params.noise_std,
        psm_buffer=50.0,  # $50M (5% of supply)
    )
    
    model_psm = StablecoinFragilityModel(params_psm)
    exp_loss_psm, diag_psm = model_psm.expected_loss()
    lcr_psm, _ = model_psm.compute_lcr()
    peg_spread_psm = model_psm.peg_deviation(
        base_params.redemption_shock * base_params.total_supply
    )
    
    scenarios['Add PSM'] = {
        'expected_loss': exp_loss_psm,
        'lcr_pct': lcr_psm * 100,
        'peg_spread_pct': peg_spread_psm,
        'run_probability': diag_psm['run_probability'] * 100,
    }
    
    # 5. LCR 150%
    # Need to adjust reserves to meet 150% LCR
    params_lcr150 = StablecoinParameters(
        total_supply=base_params.total_supply,
        reserves=base_params.reserves.copy(),
        market_impact_coeff=base_params.market_impact_coeff,
        redemption_shock=base_params.redemption_shock,
        noise_std=base_params.noise_std,
        lcr_floor=1.5,
    )
    
    model_lcr150 = StablecoinFragilityModel(params_lcr150)
    exp_loss_lcr150, diag_lcr150 = model_lcr150.expected_loss()
    lcr_lcr150, _ = model_lcr150.compute_lcr()
    peg_spread_lcr150 = model_lcr150.peg_deviation(
        base_params.redemption_shock * base_params.total_supply
    )
    
    scenarios['LCR 150%'] = {
        'expected_loss': exp_loss_lcr150,
        'lcr_pct': lcr_lcr150 * 100,
        'peg_spread_pct': peg_spread_lcr150,
        'run_probability': diag_lcr150['run_probability'] * 100,
    }
    
    # 6. LCR 200%
    params_lcr200 = StablecoinParameters(
        total_supply=base_params.total_supply,
        reserves=base_params.reserves.copy(),
        market_impact_coeff=base_params.market_impact_coeff,
        redemption_shock=base_params.redemption_shock,
        noise_std=base_params.noise_std,
        lcr_floor=2.0,
    )
    
    model_lcr200 = StablecoinFragilityModel(params_lcr200)
    exp_loss_lcr200, diag_lcr200 = model_lcr200.expected_loss()
    lcr_lcr200, _ = model_lcr200.compute_lcr()
    peg_spread_lcr200 = model_lcr200.peg_deviation(
        base_params.redemption_shock * base_params.total_supply
    )
    
    scenarios['LCR 200%'] = {
        'expected_loss': exp_loss_lcr200,
        'lcr_pct': lcr_lcr200 * 100,
        'peg_spread_pct': peg_spread_lcr200,
        'run_probability': diag_lcr200['run_probability'] * 100,
    }
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("STABLECOIN ROBUST FRAGILITY MODEL")
    print("Diamond-Dybvig + Izumi-Li Framework")
    print("=" * 70)
    
    # Create baseline USDC model
    params = create_usdc_baseline()
    model = StablecoinFragilityModel(params)
    
    print("\n📊 BASELINE PARAMETERS")
    print(f"Total Supply: ${params.total_supply:.1f}M")
    print(f"Total Reserves: ${params.total_reserves:.1f}M")
    print(f"Reserve Ratio: {params.reserve_ratio*100:.1f}%")
    print(f"\nReserve Composition:")
    for asset in params.reserves:
        print(f"  {asset.asset_class.value}: ${asset.amount:.1f}M "
              f"({asset.amount/params.total_reserves*100:.1f}%) "
              f"- {asset.fire_sale_haircut_bps}bps haircut")
    
    print("\n🎯 RUN THRESHOLD ANALYSIS")
    run_cutoff, diag = model.compute_run_threshold()
    print(f"Run Cutoff: {run_cutoff:.0f} bps")
    print(f"Fire-Sale Impact: {diag['fire_sale_impact_bps']:.1f} bps")
    print(f"Available Liquidity: ${diag['available_liquidity']:.1f}M")
    print(f"Liquidity Gap: ${diag['liquidity_gap']:.1f}M")
    print(f"Run Probability: {model.run_probability()*100:.1f}%")
    
    print("\n💧 LIQUIDITY COVERAGE RATIO")
    lcr, lcr_diag = model.compute_lcr()
    print(f"LCR: {lcr*100:.0f}%")
    print(f"HQLA: ${lcr_diag['hqla']:.1f}M")
    print(f"Net Cash Outflows (30d): ${lcr_diag['net_cash_outflows_30d']:.1f}M")
    print(f"Meets {params.lcr_floor*100:.0f}% Requirement: {lcr_diag['meets_requirement']}")
    
    print("\n💥 EXPECTED LOSS ANALYSIS")
    exp_loss, loss_diag = model.expected_loss()
    print(f"Expected Loss: ${exp_loss:.2f}M")
    print(f"Loss if Run Occurs: ${loss_diag['total_loss_if_run']:.2f}M "
          f"({loss_diag['loss_as_pct_supply']:.2f}% of supply)")
    
    print("\n🎲 ROBUST FRAGILITY")
    for backstop in [0, 100, 200, 500]:
        max_prob = model.max_run_probability(backstop_size=backstop)
        print(f"Backstop ${backstop}M → Max Run Probability: {max_prob*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("Models implemented successfully!")
    print("See functions for detailed analysis capabilities.")

