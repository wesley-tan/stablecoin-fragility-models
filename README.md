# Stablecoin Fragility Research Framework

**Reserve Composition, Fire-Sale Externalities, and Cross-Issuer Contagion**

A comprehensive research framework for analyzing stablecoin fragility using Diamond-Dybvig coordination games with Izumi-Li fire-sale pricing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Findings

### Main Result

**Optimal reserve composition**: **30-40% Tier-1 liquid** (MMF 20-25%, Repo 10-15%, Cash 0-5%)

**Performance**:
- **Internal run probability**: ~0% (conditional on fundamentals)
- **Residual tail risk**: ~14% annually (operational + exogenous shocks)
- **VaR₉₉**: ~1,500 bps (custody/regulatory/cyber risk)
- **Portfolio yield**: 491 bps (exceeds 450 bps target)

### Core Insight

> **Reserve composition dominates policy interventions.** With λ ∈ [30%, 40%], internal fire-sale fragility is eliminated at minimal cost (3-14 bps). Policy band-aids (PSM, gates, high LCR) are expensive substitutes for poor liquidity management.

**Caveat**: Eliminates *endogenous* run risk but NOT exogenous shocks (regulatory freeze, custody failure, smart contract exploits).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/wesley-tan/stablecoin-fragility-models.git
cd stablecoin-fragility-models
pip install -r requirements.txt
```

### Run Analysis

```python
# 1. Find optimal reserve mix + policy bundle
python robust_optimization.py

# 2. Generate research figures
python plot_research_questions.py

# 3. Run robustness checks
python robustness_checks.py
```

### Basic Example

```python
from run_cutoff_corrected import grid_run_cutoff_corrected

# Compute run cutoff curve
result = grid_run_cutoff_corrected(pi=0.08, kappa=0.02)
print(f"Stability threshold: λ ≥ {result.lambda_cutoff*100:.0f}%")
```

---

## 📊 Main Modules

### Core Models (4 modules)

| Module | Purpose | Key Output |
|--------|---------|------------|
| `robust_optimization.py` | Optimize reserve mix + policies | Optimal λ = 30-40% band |
| `run_cutoff_corrected.py` | Two-tier liquidity ladder | Run cutoff n* vs. λ |
| `firesale_externalities.py` | Izumi-Li multi-issuer framework | Nash vs. Social Optimum |
| `contagion_network.py` | Cross-issuer spillovers | Network heatmap |

### Risk Management (3 modules)

| Module | Purpose | Key Metric |
|--------|---------|------------|
| `liquidity_risk.py` | Liquidity-at-Risk (LaR) | Max redemptions before depeg |
| `firesale_var.py` | Fire-Sale VaR/ES | Loss distribution |
| `policy_levers.py` | Policy cost-benefit | LCR, PSM, gates, disclosure |

### Analysis & Validation (2 modules)

| Module | Purpose |
|--------|---------|
| `robustness_checks.py` | Parameter uncertainty, operational frictions, historical backtest |
| `plot_research_questions.py` | Paper-ready figures (6 plots, 300dpi) |

**Total**: ~8,000 lines of production code

---

## 📈 Research Questions Answered

### Q1: Reserve Mix → Fragility

**How do T-bills, MMFs, repo, deposits affect run cutoff n* and expected loss?**

**Answer**: 
- **Sharp threshold** at λ ≈ 30-40% (robust band, not point)
- Below threshold: Vulnerable equilibria, convex losses
- Above threshold: Unique stable equilibrium, zero internal run risk
- **Operational reality**: Effective λ can be 60-70% lower in stress (MMF cutoffs, settlement lags)

### Q2: Policies → Robust Stability

**What policy bundle minimizes max run probability at lowest cost?**

**Answer**:
- **Minimal interventions needed** if reserves well-composed
- LCR 100-120%, 24-48h disclosure, modest PSM (0-2%)
- **Total cost**: 3-14 bps
- **Avoid**: High LCR floors (>150%), redemption fees/gates (adverse selection)

---

## 🔬 Validation

### Historical Backtest (3/3 Correct)

| Episode | λ | π | Predicted | Actual | ✓ |
|---------|---|---|-----------|--------|---|
| USDC/SVB '23 | 23.6% | 11.5% | FRAGILE | Depeg -12% | ✓ |
| UST/Luna '22 | 15.0% | 80.0% | FRAGILE | Collapse | ✓ |
| USDT '24 | 85.0% | 2.0% | STABLE | Stable | ✓ |

### Robustness Checks

- ✅ **Parameter uncertainty**: Threshold band [30%, 40%] across 100 Monte Carlo samples
- ✅ **Operational frictions**: Effective λ 60-70% lower in stress scenarios
- ✅ **Sector upscaling**: Threshold rises to 38-40% at 3× current size
- ✅ **Exogenous shocks**: Residual 14% annual tail risk quantified

---

## 📖 Documentation

| Document | Purpose | Lines |
|----------|---------|-------|
| **README.md** | **Quick start (this file)** | **Short** |
| **RESEARCH_ANSWERS.md** | **Complete research findings** | **545** |
| `requirements.txt` | Dependencies | — |

---

## 📁 Generated Outputs

### Figures (9 PNG, 300dpi)

1. `figure_rq1_reserve_mix.png` — Reserve mix sensitivity (6-panel)
2. `figure_rq2_policy_frontier.png` — Policy cost-benefit analysis
3. `figure_rq_robust_frontier.png` — Yield-stability-cost trade-off
4. `figure1_run_cutoff.png` — Run cutoff vs. T-bill share
5. `figure_corrected_*.png` — Corrected models with PSM overlay
6. `contagion_heatmap.png` — Network spillover matrix

### Data (4 JSON)

- `optimal_solution.json` — Optimal reserve mix and policy bundle
- `reserve_mix_sweeps.json` — Sensitivity analysis
- `robustness_checks.json` — Robustness results
- `run_cutoff_analysis_*.json` — Episode-specific analysis

---

## 🎓 Citation

```bibtex
@software{stablecoin_fragility_2025,
  title = {Stablecoin Fragility Research Framework},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/wesley-tan/stablecoin-fragility-models}
}
```

---

## 📊 Key Parameters

| Parameter | Symbol | Baseline | Stress | Interpretation |
|-----------|--------|----------|--------|----------------|
| Liquid reserves | λ | 30-40% | 20-30% | Tier-1 share |
| Impatient share | π | 5-10% | 10-20% | Redemption intensity |
| Price impact | κ | 1-3% | 5-15% | Fire-sale coefficient |
| Max haircut | h_max | 10% | 15% | Price floor |

---

## 💼 Policy Recommendations

### For Issuers
- ✅ Target **35-40% Tier-1** liquid reserves
- ✅ **24-48h disclosure** (balance transparency vs synchronization)
- ✅ Modest **PSM buffer** (1-2%) for tail events
- ❌ Avoid bank deposit concentration (>10%)

### For Regulators
- ✅ Mandate **minimum 30% Tier-1** reserves
- ✅ Industry-wide **24-48h disclosure** standard
- ✅ Limit single asset class to **<60%** of reserves
- ✅ Contingent systemic **PSM facility** ($500M-$1B)

### For DeFi
- ✅ Monitor Tier-1 % for integrated stablecoins
- ✅ Alert/circuit-breaker if **λ < 35%**
- ✅ Diversify across multiple stablecoins

---

## ⚠️ Caveats & Limitations

1. **Threshold is a band** [30%, 40%], not constant (34%)
2. **Conditional stability**: Internal run prob ≈ 0%, BUT **excludes exogenous shocks**
3. **Residual tail risk**: ~14% annually (regulatory freeze, custody failure, smart contract exploits)
4. **Operational frictions**: Effective λ 60-70% lower in stress (MMF cutoffs, settlement lags)
5. **Depth-limited**: T-bill liquidity assumption may fail at 3-5× sector scale

**Proper claim**: "λ ∈ [30%, 40%] eliminates endogenous fire-sale fragility, conditional on fundamentals. Does NOT eliminate exogenous shocks. VaR₉₉ ≈ 1,500 bps (non-zero)."

---

## 🔗 References

- Diamond & Dybvig (1983): Bank Runs, Deposit Insurance, and Liquidity
- Izumi & Li (JMCB forthcoming): Financial Stability with Fire Sale Externalities
- Morris & Shin (1998): Unique Equilibrium in Self-Fulfilling Currency Attacks

---

## 📧 Contact

- **GitHub**: https://github.com/wesley-tan/stablecoin-fragility-models
- **Issues**: Report bugs or request features via GitHub Issues
- **Contributions**: Pull requests welcome

---

**Last Updated**: October 16, 2025  
**Version**: 2.1 (With Robustness Checks)  
**Status**: ✅ Production Ready, Academically Rigorous

