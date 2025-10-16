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

## 📊 Key Parameters

| Parameter | Symbol | Baseline | Stress | Interpretation |
|-----------|--------|----------|--------|----------------|
| Liquid reserves | λ | 30-40% | 20-30% | Tier-1 share |
| Impatient share | π | 5-10% | 10-20% | Redemption intensity |
| Price impact | κ | 1-3% | 5-15% | Fire-sale coefficient |
| Max haircut | h_max | 10% | 15% | Price floor |