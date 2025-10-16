# Stablecoin Fragility Research Framework

This repository provides a toolkit for analyzing stablecoin fragility across multiple dimensions:

| Module | Purpose | Lines |
|--------|---------|-------|
| `stablecoin_fragility.py` | Core fragility + Diamond-Dybvig | 501 |
| `run_cutoff.py` | Two-tier ladder (cash/MMF, T-bills) | 416 |
| `run_cutoff_corrected.py` | Fixed fire-sale logic & units | 366 |
| `calibration.py` | USDC/SVB, UST/Luna episodes | 470 |
| `onchain_frictions.py` | Gas spikes, block time, Curve pools | 338 |

### Risk Management Tools

| Module | Purpose | Lines |
|--------|---------|-------|
| **`liquidity_risk.py`** | Liquidity-at-Risk (LaR) | 382 |
| **`firesale_var.py`** | Fire-Sale VaR/ES Monte Carlo | 453 |
| **`policy_levers.py`** | LCR, PSM, gates, disclosure | 481 |

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/stablecoin_fragility_model.git
cd stablecoin_fragility_model

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Run Cutoff Analysis (Two-Tier Ladder)

```python
from run_cutoff_corrected import grid_run_cutoff_corrected

# Compute run cutoff curve
result = grid_run_cutoff_corrected(
    pi=0.08,      # 8% impatient investors
    kappa=0.02,   # 2% price impact coefficient
    h_max=0.10    # 10% max haircut cap
)

print(f"Lambda cutoff: {result.lambda_cutoff*100:.1f}% liquid reserves")
print(f"Vulnerable range: {result.vulnerable.sum()} of {len(result.lambda_grid)} points")
```

#### 2. Liquidity-at-Risk (LaR)

```python
from liquidity_risk import liquidity_at_risk

lar = liquidity_at_risk(
    lam=0.20,           # 20% liquid reserves
    pi=0.115,           # 11.5% baseline redemptions
    kappa=0.10,         # 10% impact coefficient
    max_depeg_bps=100   # 100bp depeg threshold
)

print(f"LaR: {lar['LaR_pct']:.1f}% of supply")
print(f"Buffer: {lar['buffer_pct']:.1f}% cushion")
print(f"Time to LaR: {lar['time_to_breach_hours']:.1f} hours")
```

#### 3. Fire-Sale VaR/ES

```python
from firesale_var import monte_carlo_FS_VaR
import numpy as np

# Define shock distributions
pi_dist = np.random.uniform(0.05, 0.15, 10000)
kappa_dist = np.random.uniform(0.05, 0.15, 10000)

result = monte_carlo_FS_VaR(
    lam=0.20,
    pi_dist=pi_dist,
    kappa_dist=kappa_dist,
    n_sims=10000
)

print(f"VaR₉₅: {result['VaR_metrics']['95%']['VaR_bps']:.0f} bps")
print(f"ES₉₅: {result['VaR_metrics']['95%']['ES_bps']:.0f} bps")
print(f"VaR₉₉: {result['VaR_metrics']['99%']['VaR_bps']:.0f} bps")
```

#### 4. Fire-Sale Externalities (Multi-Issuer)

```python
from firesale_externalities import run_izumi_li_analysis

# Setup issuers
issuer_names = ["USDC", "DAI", "USDT"]
initial_holdings = {
    "USDC": 34.8e9,  # $34.8B T-bills
    "DAI": 1.04e9,   # $1.04B T-bills
    "USDT": 33.2e9   # $33.2B T-bills
}
liquidity_needs = {k: v * 0.10 for k, v in initial_holdings.items()}

results = run_izumi_li_analysis(
    issuer_names=issuer_names,
    initial_holdings=initial_holdings,
    liquidity_needs=liquidity_needs,
    market_depth=50e9,  # $50B market
    kappa=0.05
)

nash_loss = results['baseline_nash']['total_loss']
coord_loss = results['coordinated']['total_loss']
print(f"Nash loss: ${nash_loss/1e6:.1f}M")
print(f"Coordinated loss: ${coord_loss/1e6:.1f}M")
print(f"Externality: ${(nash_loss - coord_loss)/1e6:.1f}M")
```

#### 5. Cross-Issuer Contagion

```python
from contagion_network import create_usdc_dai_usdt_network, simulate_contagion

# Create network
network = create_usdc_dai_usdt_network()

# Simulate USDC shock
history = simulate_contagion(
    network,
    initial_shock_coin="USDC",
    shock_pi=0.115,  # 11.5% redemptions
    max_rounds=10
)

print(f"Rounds to stabilize: {len(history)}")
print(f"Issuers affected: {sum(1 for h in history[-1]['runs'] if h['n_star'] > 0)}")
print(f"System depeg: {history[-1]['tbill_price']:.4f}")
```

### Generate All Figures

```bash
# Corrected run cutoff plots
python plot_corrected.py

# Contagion heatmap and IRFs
python contagion_heatmap.py

# Comprehensive analysis
python run_analysis.py
```

---

## 📊 Key Results

### 1. Reserve Composition → Run Thresholds

**Finding**: Run cutoff **decreases** as liquid reserves (cash/MMF) **increase**

| Liquid % | T-bill % | n* (run %) | Depeg (bps) | Status |
|----------|----------|------------|-------------|---------|
| 0-7% | 93-100% | 100% | 1,000+ | 🔴 Full run |
| 8-11% | 89-92% | 50-100% | 500-1,000 | 🟡 Unstable |
| 12-20% | 80-88% | 10-50% | 100-500 | 🟠 Threshold |
| 20%+ | <80% | 0-10% | 0-100 | 🟢 Stable |

**Policy Recommendation**: LCR-style floor at **15-20% liquid reserves**

**Sensitivity**:
- π (impatient share): 5% → 10% shifts cutoff by **~3pp**
- κ (price impact): 1% → 3% shifts cutoff by **~5pp**
- PSM ($200M on $1B supply): shifts cutoff **left by 2-3pp**

### 2. Fire-Sale Externalities (Izumi-Li)

**Setup**: USDC, DAI, USDT with shared T-bill exposure ($69B combined holdings)

| Scenario | Total Sales | Price | Depeg (bps) | Total Loss |
|----------|------------|-------|-------------|------------|
| **Nash (uncoordinated)** | $7.0B | 0.9930 | 70 | $49M |
| **Coordinated** | $6.9B | 0.9935 | 65 | $45M |
| **Diversified (20%)** | $5.5B | 0.9950 | 50 | $28M |
| **Position limits (40%)** | $4.2B | 0.9965 | 35 | $15M |

**Key Insight**: Coordination gains = **$4M** (9% improvement). Diversification gains = **$21M** (43% improvement).

**Externality Breakdown**:
- USDC sales impose **$18M** externality on DAI/USDT
- USDT sales impose **$16M** externality on USDC/DAI
- DAI sales impose **$0.5M** externality (smaller player)

### 3. Liquidity-at-Risk (LaR)

**Metric**: Maximum redemptions before 100bp depeg

**Example** (λ=20%, π=8%, κ=10%):
```
LaR = 28% of supply
Buffer = 20pp above baseline
Time to LaR = 39.5 hours @ 10%/day redemptions
Circuit breaker window = 60 seconds (2 blocks)
```

**LaR vs. Reserve Mix**:

| λ (liquid %) | LaR (%) | Buffer (pp) | Time (hours) |
|--------------|---------|-------------|--------------|
| 5% | 8% | 3 | 11.4 |
| 10% | 15% | 5 | 21.4 |
| 20% | 28% | 20 | 39.5 |
| 30% | 42% | 34 | 60.0 |

**Application**: Daily monitoring dashboard, stress test capacity

### 4. Fire-Sale VaR/ES

**Monte Carlo** (10,000 sims, π ~ U[5%, 15%], κ ~ U[5%, 15%])

**Example** (λ=20%):
```
VaR₉₅ = 145 bps
ES₉₅ = 203 bps
VaR₉₉ = 289 bps
ES₉₉ = 378 bps
Max loss = 621 bps
```

**Reserve Mix Comparison**:

| λ | VaR₉₅ (bps) | ES₉₅ (bps) | VaR₉₉ (bps) | Run Prob |
|---|------------|-----------|-----------|----------|
| 5% | 450 | 620 | 780 | 85% |
| 10% | 320 | 445 | 590 | 72% |
| 20% | 145 | 203 | 289 | 18% |
| 30% | 45 | 67 | 95 | 3% |

**Policy**: Size capital buffer = **VaR₉₉ + 50bp safety margin**

### 5. Cross-Issuer Contagion

**Transmission Channels**:
1. **DEX arbitrage** (immediate): USDC depeg → Curve 3pool imbalance → DAI/USDT pressure
2. **Fire-sale cascade** (1-2 hours): Joint T-bill sales → price impact feedback
3. **Information spillover** (2-6 hours): USDC run → DAI/USDT π increases

**March 2023 USDC/SVB Calibration**:

| Issuer | Direct Shock | DEX Spillover | Fire-Sale | Info | Total Depeg |
|--------|-------------|---------------|-----------|------|-------------|
| **USDC** | -1200bp | 0bp | 0bp | 0bp | **-1200bp** |
| **DAI** | 0bp | -200bp | -150bp | -150bp | **-500bp** |
| **USDT** | 0bp | -50bp | -100bp | -50bp | **-200bp** |

**Note**: Current model captures ~60% of observed DAI spillover. Refinements needed:
- Stronger Curve/Uniswap linkages
- PSM redemption flows (MakerDAO)
- Confidence cascades

**Network Metrics**:
- **Centrality**: USDC (0.65), USDT (0.58), DAI (0.42)
- **Contagion amplification**: 1.8x (1% USDC shock → 1.8% system stress)
- **Critical threshold**: 3+ simultaneous runs → systemic crisis

### 6. Policy Levers

**Cost-Benefit Analysis**:

| Policy | Annual Cost | Risk Reduction | Ratio | Recommendation |
|--------|------------|----------------|-------|----------------|
| **Real-time disclosure** | ~0bp | -10pp sunspot prob | ∞ | ✅ Implement |
| **LCR 150%** | 20bp | Eliminates run risk | 10x | ✅ Core requirement |
| **PSM $200M** | 0-5bp | -50bp VaR₉₉ | 10x+ | ✅ Tail protection |
| **Redemption gates (50bp)** | 25bp adverse selection | -30bp expected loss | 1.2x | ⚠️ Emergency only |
| **Reserve diversification (20%)** | 15bp yield drag | -43% fire-sale loss | 3x | ✅ Strongly recommend |

**Optimal Bundle**: Real-time disclosure + LCR 150% + PSM $200M + diversification
- **Total cost**: ~40bp/year
- **Total benefit**: Eliminates run risk, -65bp VaR₉₉, -43% fire-sale externality
- **Net benefit**: **>$50M/year** on $10B supply

---

## 📈 Model Specifications

### Two-Tier Liquidity Ladder

**Setup**:
- **Tier-1** (λ): Cash + MMFs (redeemable at par, $1.00)
- **Tier-2** (1-λ): T-bills (fire-sale price p < 1.00 if sold early)

**Redemption Sequencing**:
1. Use all Tier-1 assets (λ)
2. If R(n) > λ, sell Tier-2: Q = R(n) - λ

**Fire-Sale Pricing** (Corrected):
```
Q(n) = max(0, R(n) - λ)                    # Shortfall only
p(n) = 1 - min(κ·Q(n)/(1-λ), h_max)       # Stack-fraction impact, capped
R(n) = π + n·(1-π)                         # Total redemptions
```

**Patient Indifference** (Equilibrium):
```
λ + p(n*)·(R(n*) - λ) = R(n*)              # Wait = Run
```

**Parameters**:
- **π**: Impatient share (baseline: 5-10%, stress: 10-15%)
- **κ**: Price impact coefficient (calibrated: 1-3%, stress: 5-10%)
- **h_max**: Maximum haircut cap (10-15% for T-bills)

**Corrections Applied** (October 2025):
1. ✅ Fire-sale applies to **shortfall only** (not full R)
2. ✅ Price impact uses **stack fraction** Q/(1-λ), not Q/M
3. ✅ Haircut **capped** at h_max (no negative prices)
4. ✅ Units: cutoffs in **%**, converted to bps once (×100)
5. ✅ Direction: More liquid reserves → **lower** run incentives

### Fire-Sale Externalities (Izumi-Li)

**Nash Equilibrium** (Uncoordinated):
```
Each issuer i solves:
  max   q_i · P(Q_total)
  s.t.  q_i · P(Q_total) ≥ L_i  (liquidity need)
  
where Q_total = Σ q_j  (aggregate sales)
      P(Q) = 1 - κ·Q/M  (market price)
```

**Social Optimum** (Coordinated):
```
Social planner solves:
  min   Q_total · (1 - P(Q_total))  (total loss)
  s.t.  Q_total · P(Q_total) ≥ Σ L_i  (aggregate liquidity)
```

**Externality**:
```
Externality_i = dP/dq_i · Σ_{j≠i} holdings_j
              = -(κ/M) · q_i · (Holdings_total - holdings_i)
```

**Welfare Loss**:
```
Δ Welfare = Loss(Nash) - Loss(Social Optimum)
```

### Contagion Network

**Nodes**: Stablecoins (USDC, DAI, USDT, ...)

**Edges**:
1. **DEX pools**: Curve 3pool (USDC-DAI-USDT), Uniswap pairs
2. **Common reserves**: Shared T-bill exposure
3. **PSM channels**: Direct redemption (e.g., DAI ↔ USDC)

**Dynamics**:
```
Round t:
1. Update π_i based on spillovers:
   π_i(t) = π_i(t-1) + Σ_j w_ij · I(j running)
   
2. Update T-bill price:
   p_T(t) = 1 - κ · (Σ_i Q_i(t)) / M
   
3. Solve each issuer's run cutoff:
   n_i*(t) = solve(λ_i + p_T(t)·(R_i - λ_i) = R_i)
   
4. Update DEX pools via arbitrage flows
```

