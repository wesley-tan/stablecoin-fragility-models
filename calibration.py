"""
Calibration Module for Stablecoin Fragility Models
==================================================

Historical data and calibrations for major stablecoin episodes:
- USDC/SVB Crisis (March 2023)
- UST Collapse (May 2022)
- USDT Mini-Depegs (various)
- Circle Reserve Fund data

Data sources:
- Circle Reserve Fund (BlackRock)
- Federal Reserve (T-bill yields, MMF data)
- Curve/Uniswap pool states (on-chain)
- Dune Analytics (mints/burns)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from stablecoin_fragility import (
    StablecoinParameters, ReserveAsset, AssetClass
)


@dataclass
class HistoricalEpisode:
    """Historical stablecoin stress episode"""
    name: str
    date: datetime
    stablecoin: str
    
    # Market conditions
    total_supply_billions: float
    peak_redemption_amount_billions: float
    peak_redemption_rate_pct: float
    duration_hours: float
    
    # Peg deviation
    min_price: float  # Lowest peg price (e.g., 0.88 for 12% depeg)
    max_discount_pct: float  # Maximum discount from $1.00
    
    # Reserve composition (if available)
    reserve_composition: Optional[Dict[str, float]] = None
    
    # Yields at time
    tbill_3m_yield: Optional[float] = None
    mmf_yield: Optional[float] = None
    
    # Outcome
    recovered: bool = True
    recovery_time_hours: Optional[float] = None
    
    # Additional notes
    trigger_event: Optional[str] = None
    contagion_effects: Optional[List[str]] = None


# Historical Episodes Database
HISTORICAL_EPISODES = {
    'usdc_svb_2023': HistoricalEpisode(
        name="USDC/SVB Crisis",
        date=datetime(2023, 3, 10),
        stablecoin="USDC",
        total_supply_billions=43.5,
        peak_redemption_amount_billions=5.0,
        peak_redemption_rate_pct=11.5,  # Peak daily rate
        duration_hours=72,  # Main stress period
        min_price=0.88,
        max_discount_pct=12.0,
        reserve_composition={
            'T-bills': 0.80,
            'Bank Deposits': 0.20,
        },
        tbill_3m_yield=0.0465,  # 4.65%
        mmf_yield=0.0442,  # 4.42%
        recovered=True,
        recovery_time_hours=96,
        trigger_event="SVB collapse; $3.3B USDC reserves trapped",
        contagion_effects=["DAI depeg", "USDT temporary depeg", "DeFi liquidity crisis"],
    ),
    
    'ust_collapse_2022': HistoricalEpisode(
        name="UST/Luna Collapse",
        date=datetime(2022, 5, 7),
        stablecoin="UST",
        total_supply_billions=18.7,
        peak_redemption_amount_billions=12.0,
        peak_redemption_rate_pct=64.2,  # Massive run
        duration_hours=168,  # One week collapse
        min_price=0.01,  # Effectively zero
        max_discount_pct=99.0,
        reserve_composition={
            'Bitcoin': 0.40,  # Luna Foundation Guard reserves
            'Other Crypto': 0.60,
        },
        tbill_3m_yield=0.0089,  # 0.89%
        mmf_yield=0.0025,  # 0.25%
        recovered=False,
        recovery_time_hours=None,
        trigger_event="Curve pool imbalance → algorithmic death spiral",
        contagion_effects=["LUNA collapse", "$60B market cap destroyed", "Crypto bear market"],
    ),
    
    'usdt_2022_05': HistoricalEpisode(
        name="USDT Mini-Depeg May 2022",
        date=datetime(2022, 5, 12),
        stablecoin="USDT",
        total_supply_billions=83.0,
        peak_redemption_amount_billions=7.0,
        peak_redemption_rate_pct=8.4,
        duration_hours=48,
        min_price=0.95,
        max_discount_pct=5.0,
        reserve_composition={
            'Commercial Paper': 0.35,
            'T-bills': 0.40,
            'Other': 0.25,
        },
        tbill_3m_yield=0.0089,
        mmf_yield=0.0025,
        recovered=True,
        recovery_time_hours=72,
        trigger_event="UST collapse contagion; fear of reserve quality",
        contagion_effects=["General stablecoin distrust", "Flight to USDC"],
    ),
    
    'usdc_march_2021': HistoricalEpisode(
        name="USDC Growth Phase",
        date=datetime(2021, 3, 1),
        stablecoin="USDC",
        total_supply_billions=10.0,
        peak_redemption_amount_billions=0.2,
        peak_redemption_rate_pct=2.0,  # Normal volatility
        duration_hours=24,
        min_price=0.999,
        max_discount_pct=0.1,
        reserve_composition={
            'Cash': 0.61,
            'Short-term Treasuries': 0.39,
        },
        tbill_3m_yield=0.0002,  # 0.02% (near zero)
        mmf_yield=0.0001,  # 0.01%
        recovered=True,
        recovery_time_hours=12,
        trigger_event="Normal market operations",
        contagion_effects=None,
    ),
}


class CalibrationEngine:
    """
    Engine for calibrating model parameters to historical episodes.
    """
    
    def __init__(self):
        self.episodes = HISTORICAL_EPISODES
    
    def calibrate_to_episode(self, episode_key: str,
                           scale_to_millions: float = 1000.0) -> StablecoinParameters:
        """
        Create model parameters calibrated to historical episode.
        
        Args:
            episode_key: Key from HISTORICAL_EPISODES
            scale_to_millions: Scale supply to this size (default 1000M = $1B)
        
        Returns:
            Calibrated StablecoinParameters
        """
        episode = self.episodes[episode_key]
        
        # Scaling factor
        scale = scale_to_millions / (episode.total_supply_billions * 1000)
        
        # Build reserves from composition
        reserves = []
        
        if episode.reserve_composition:
            for asset_type, share in episode.reserve_composition.items():
                amount = scale_to_millions * share
                
                # Map to asset classes
                if 'T-bill' in asset_type or 'Treasur' in asset_type:
                    asset_class = AssetClass.TBILLS
                    yield_rate = episode.tbill_3m_yield or 0.04
                    haircut_bps = 2
                    liquidity_weight = 1.0
                    
                elif 'Deposit' in asset_type or 'Cash' in asset_type:
                    asset_class = AssetClass.DEPOSITS
                    yield_rate = (episode.tbill_3m_yield or 0.04) * 0.75
                    haircut_bps = 500 if 'SVB' in episode.name else 50
                    liquidity_weight = 0.0
                    
                elif 'MMF' in asset_type or 'Money Market' in asset_type:
                    asset_class = AssetClass.MMF
                    yield_rate = episode.mmf_yield or 0.04
                    haircut_bps = 10
                    liquidity_weight = 0.95
                    
                elif 'Repo' in asset_type:
                    asset_class = AssetClass.REPO
                    yield_rate = (episode.tbill_3m_yield or 0.04) * 1.05
                    haircut_bps = 10
                    liquidity_weight = 0.85
                    
                elif 'Commercial Paper' in asset_type:
                    # Treat as lower-grade MMF
                    asset_class = AssetClass.MMF
                    yield_rate = (episode.tbill_3m_yield or 0.04) * 1.5
                    haircut_bps = 50
                    liquidity_weight = 0.5
                    
                else:
                    # Unknown asset type - conservative assumption
                    asset_class = AssetClass.DEPOSITS
                    yield_rate = 0.02
                    haircut_bps = 200
                    liquidity_weight = 0.3
                
                reserves.append(ReserveAsset(
                    asset_class=asset_class,
                    amount=amount,
                    yield_rate=yield_rate,
                    fire_sale_haircut_bps=haircut_bps,
                    liquidity_weight=liquidity_weight,
                ))
        else:
            # Default reserve mix if not specified
            reserves = [
                ReserveAsset(AssetClass.TBILLS, scale_to_millions * 0.6, 0.04, 2, 1.0),
                ReserveAsset(AssetClass.DEPOSITS, scale_to_millions * 0.2, 0.03, 100, 0.0),
                ReserveAsset(AssetClass.REPO, scale_to_millions * 0.1, 0.042, 10, 0.85),
                ReserveAsset(AssetClass.MMF, scale_to_millions * 0.1, 0.04, 10, 0.95),
            ]
        
        # Redemption shock from episode
        redemption_shock = episode.peak_redemption_rate_pct / 100
        
        # Noise calibration from price deviation
        # Higher depegs suggest more coordination/panic
        noise_std = 0.05  # Default
        if episode.max_discount_pct > 10:
            noise_std = 0.02  # High panic, low noise (strong coordination)
        elif episode.max_discount_pct < 2:
            noise_std = 0.10  # Low stress, high noise (weak coordination)
        
        # Market impact coefficient
        # Calibrate to observed depeg
        # P = 1 - κ*Q/C → κ = (1-P)*C/Q
        if episode.max_discount_pct > 0:
            observed_price = 1.0 - episode.max_discount_pct / 100
            Q = episode.peak_redemption_amount_billions * 1000  # Forced sales
            C = scale_to_millions * 0.6  # Assume 60% in liquid reserves
            
            # Implied market impact
            if Q > 0 and C > 0:
                implied_kappa = (1 - observed_price) * C / Q
                market_impact_coeff = max(0.01, min(0.20, implied_kappa))
            else:
                market_impact_coeff = 0.05
        else:
            market_impact_coeff = 0.05
        
        # Create parameters
        params = StablecoinParameters(
            total_supply=scale_to_millions,
            reserves=reserves,
            market_impact_coeff=market_impact_coeff,
            redemption_shock=redemption_shock,
            noise_std=noise_std,
            psm_buffer=0.0,
            redemption_gate_cost=0.0,
            lcr_floor=1.0,
        )
        
        return params
    
    def compare_episodes(self, episode_keys: List[str]) -> Dict:
        """
        Compare multiple historical episodes.
        
        Args:
            episode_keys: List of episode keys to compare
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'episodes': [],
            'metrics': {
                'supply_billions': [],
                'redemption_pct': [],
                'max_discount_pct': [],
                'duration_hours': [],
                'recovered': [],
                'market_impact_coeff': [],
            }
        }
        
        for key in episode_keys:
            if key not in self.episodes:
                continue
            
            episode = self.episodes[key]
            params = self.calibrate_to_episode(key)
            
            comparison['episodes'].append(episode.name)
            comparison['metrics']['supply_billions'].append(episode.total_supply_billions)
            comparison['metrics']['redemption_pct'].append(episode.peak_redemption_rate_pct)
            comparison['metrics']['max_discount_pct'].append(episode.max_discount_pct)
            comparison['metrics']['duration_hours'].append(episode.duration_hours)
            comparison['metrics']['recovered'].append(episode.recovered)
            comparison['metrics']['market_impact_coeff'].append(params.market_impact_coeff)
        
        return comparison
    
    def get_yield_curve(self, date: datetime) -> Dict[str, float]:
        """
        Get approximate yield curve for a date (based on episodes).
        
        Args:
            date: Date to get yields for
        
        Returns:
            Dictionary of yields by instrument
        """
        # Find closest episode
        closest_episode = None
        min_days = float('inf')
        
        for episode in self.episodes.values():
            days_diff = abs((episode.date - date).days)
            if days_diff < min_days:
                min_days = days_diff
                closest_episode = episode
        
        if closest_episode and closest_episode.tbill_3m_yield:
            return {
                'tbill_3m': closest_episode.tbill_3m_yield,
                'mmf': closest_episode.mmf_yield or closest_episode.tbill_3m_yield * 0.9,
                'repo': closest_episode.tbill_3m_yield * 1.05,
                'deposit': closest_episode.tbill_3m_yield * 0.75,
            }
        else:
            # Default yields
            return {
                'tbill_3m': 0.04,
                'mmf': 0.038,
                'repo': 0.042,
                'deposit': 0.03,
            }


def create_scenario_variants() -> Dict[str, StablecoinParameters]:
    """
    Create multiple scenario variants for analysis.
    
    Returns:
        Dictionary of scenario name → parameters
    """
    calibrator = CalibrationEngine()
    
    scenarios = {}
    
    # Baseline: USDC pre-crisis
    scenarios['baseline'] = calibrator.calibrate_to_episode('usdc_svb_2023')
    
    # High stress: UST-like collapse
    scenarios['high_stress'] = calibrator.calibrate_to_episode('ust_collapse_2022')
    
    # Moderate stress: USDT mini-depeg
    scenarios['moderate_stress'] = calibrator.calibrate_to_episode('usdt_2022_05')
    
    # Low stress: Normal operations
    scenarios['low_stress'] = calibrator.calibrate_to_episode('usdc_march_2021')
    
    # Conservative reserve mix (90% T-bills)
    base = calibrator.calibrate_to_episode('usdc_svb_2023')
    scenarios['conservative_reserves'] = StablecoinParameters(
        total_supply=1000.0,
        reserves=[
            ReserveAsset(AssetClass.TBILLS, 900.0, 0.0465, 2, 1.0),
            ReserveAsset(AssetClass.REPO, 50.0, 0.048, 10, 0.85),
            ReserveAsset(AssetClass.MMF, 50.0, 0.0442, 10, 0.95),
        ],
        market_impact_coeff=base.market_impact_coeff,
        redemption_shock=base.redemption_shock * 0.5,  # Lower due to better reserves
        noise_std=0.07,  # Higher confidence
    )
    
    # Risky reserve mix (50% deposits)
    scenarios['risky_reserves'] = StablecoinParameters(
        total_supply=1000.0,
        reserves=[
            ReserveAsset(AssetClass.TBILLS, 400.0, 0.0465, 2, 1.0),
            ReserveAsset(AssetClass.DEPOSITS, 500.0, 0.035, 500, 0.0),
            ReserveAsset(AssetClass.MMF, 100.0, 0.0442, 10, 0.95),
        ],
        market_impact_coeff=base.market_impact_coeff * 1.5,  # Worse liquidity
        redemption_shock=base.redemption_shock * 1.5,  # Higher stress
        noise_std=0.03,  # Lower confidence, higher panic
    )
    
    return scenarios


if __name__ == "__main__":
    print("=" * 70)
    print("CALIBRATION MODULE")
    print("Historical Stablecoin Episodes")
    print("=" * 70)
    
    calibrator = CalibrationEngine()
    
    print("\n📚 AVAILABLE EPISODES")
    for key, episode in calibrator.episodes.items():
        print(f"\n{episode.name} ({episode.date.strftime('%Y-%m-%d')})")
        print(f"  Stablecoin: {episode.stablecoin}")
        print(f"  Supply: ${episode.total_supply_billions:.1f}B")
        print(f"  Peak Redemption: {episode.peak_redemption_rate_pct:.1f}% "
              f"(${episode.peak_redemption_amount_billions:.1f}B)")
        print(f"  Max Depeg: {episode.max_discount_pct:.1f}%")
        print(f"  Recovered: {episode.recovered}")
        if episode.trigger_event:
            print(f"  Trigger: {episode.trigger_event}")
    
    print("\n" + "=" * 70)
    print("\n🎯 CALIBRATION EXAMPLE: USDC/SVB Crisis")
    params = calibrator.calibrate_to_episode('usdc_svb_2023')
    
    print(f"\nScaled Model (${params.total_supply:.0f}M):")
    print(f"  Total Reserves: ${params.total_reserves:.0f}M")
    print(f"  Reserve Ratio: {params.reserve_ratio*100:.1f}%")
    print(f"  Market Impact κ: {params.market_impact_coeff:.3f}")
    print(f"  Redemption Shock: {params.redemption_shock*100:.1f}%")
    print(f"  Signal Noise σ: {params.noise_std:.3f}")
    
    print(f"\nReserve Breakdown:")
    for asset in params.reserves:
        print(f"  {asset.asset_class.value}: ${asset.amount:.0f}M "
              f"({asset.amount/params.total_reserves*100:.1f}%) "
              f"@ {asset.yield_rate*100:.2f}% yield")
    
    print("\n" + "=" * 70)
    print("\n📊 EPISODE COMPARISON")
    comparison = calibrator.compare_episodes([
        'usdc_svb_2023',
        'ust_collapse_2022',
        'usdt_2022_05',
        'usdc_march_2021'
    ])
    
    print(f"\n{'Episode':<30} {'Supply($B)':<12} {'Redemption%':<14} {'MaxDepeg%':<12} {'Recovered':<10}")
    print("-" * 80)
    for i, name in enumerate(comparison['episodes']):
        supply = comparison['metrics']['supply_billions'][i]
        redemption = comparison['metrics']['redemption_pct'][i]
        depeg = comparison['metrics']['max_discount_pct'][i]
        recovered = "Yes" if comparison['metrics']['recovered'][i] else "No"
        
        print(f"{name:<30} {supply:<12.1f} {redemption:<14.1f} {depeg:<12.1f} {recovered:<10}")
    
    print("\n" + "=" * 70)
    print("Calibration module complete!")

