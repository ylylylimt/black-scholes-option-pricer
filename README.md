# Options Pricing & Greeks Explorer

A from-scratch implementation of the **Black-Scholes-Merton** options pricing model with full Greeks, an implied volatility solver, and live market data analysis via Yahoo Finance.

---

## What it does

| Module | Description |
|---|---|
| `black_scholes.py` | BSM pricer, all 5 Greeks, IV solver (Newton-Raphson + Brent fallback), put-call parity |
| `market_data.py` | Fetches live options chains, computes IV for every contract |
| `visualise.py` | Vol smile, IV surface, Greeks profiles, model fit diagnostics |
| `main.py` | End-to-end pipeline; runs demos + live analysis |
| `vol_analysis.py` | Realized vol (trailing / forward), variance risk premium, Newey-West (HAC) inference, forecast regressions |
| `hedging.py` | Daily delta-hedged short-option simulator and trade statistics |
| `history.py` | Cached SPY / VIX / T-bill history loader, plus a synthetic market for offline checks |
| `run_vrp_backtest.py` | End-to-end VRP study and hedged-P&L backtest |
| `test_black_scholes.py` | Unit tests covering pricing accuracy, Greek properties, IV round-trips |
| `test_vol_analysis.py`, `test_hedging.py`, `test_history.py` | Tests for the volatility study and the hedging simulator (see below) |

---

## Quickstart

```bash
pip install -r requirements.txt

# Run standalone demos (no internet needed)
python main.py --demo-only

# Full live analysis on SPY (default)
python main.py

# Different ticker
python main.py --ticker AAPL --expirations 3
```

---

## The Maths

### Black-Scholes formula

For a European call and put on a non-dividend-paying asset:

```
C = S·N(d₁) − K·e^(−rT)·N(d₂)
P = K·e^(−rT)·N(−d₂) − S·N(−d₁)

d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
d₂ = d₁ − σ√T
```

Where:
- `S` = spot price, `K` = strike, `T` = time to expiry (years)
- `r` = risk-free rate (continuous), `σ` = volatility (annualised)
- `N(·)` = standard normal CDF

### Greeks

| Greek | Definition | Interpretation |
|---|---|---|
| **Delta** (Δ) | ∂V/∂S | Sensitivity to spot price; hedge ratio |
| **Gamma** (Γ) | ∂²V/∂S² | Rate of change of delta; convexity |
| **Theta** (Θ) | ∂V/∂t | Time decay per calendar day |
| **Vega** (ν) | ∂V/∂σ | Sensitivity to volatility (per 1% move) |
| **Rho** (ρ) | ∂V/∂r | Sensitivity to interest rate (per 1% move) |

Key Greek relationships:
- Δ_call − Δ_put = 1 (put-call delta parity)
- Γ_call = Γ_put (same strike, same expiry)
- ν_call = ν_put (same strike, same expiry)

### Implied Volatility

The market-observed option price is inverted to solve for σ — the volatility the market is *implying*. Since BSM has no closed-form inverse in σ, this is solved numerically:

1. **Newton-Raphson** with vega as the derivative — fast convergence near ATM
2. **Brent's method** fallback over [0.001, 10.0] — robust for deep ITM/OTM

### Volatility Smile

Under BSM, IV should be constant across strikes. In practice it is not — low strikes (OTM puts) trade at a premium due to:
- **Crash risk / skew**: demand for downside protection
- **Fat tails**: the log-normal assumption understates extreme moves

The deviation from flat IV is the *volatility smile*.

---

## Sample outputs

```
============================================================
  Black-Scholes Pricer — Single Contract
============================================================
  S=100  K=105  T=0.25y  r=5.0%  σ=20%

         Price      Delta     Gamma          Theta          Vega          Rho
Call   $2.7235   +0.3894  0.000038  -0.051423/day  0.1937/1%σ   +0.0369/1%r
Put    $6.4934   -0.6106  0.000038  -0.037916/day  0.1937/1%σ   -0.0568/1%r

  Put-call parity check: C-P=-3.7699  S-Ke^(-rT)=-3.7699  ✓ holds

============================================================
  Implied Volatility Solver — Round-trip Test
============================================================
 Type True IV  Mkt Price  Solved IV        Error
 Call     10%    $0.9054     10.00%  0.000000%
  Put     10%    $0.6617     10.00%  0.000000%
 Call     20%    $3.0634     20.00%  0.000000%
  Put     20%    $1.8254     20.00%  0.000000%
```

---

## Tests

```bash
pytest test_black_scholes.py -v
```

62 tests in total. The Black-Scholes suite covers:
- Analytical price accuracy vs known values
- Put-call parity (ATM and ITM)
- Greek signs and relationships (delta bounds, gamma positivity, theta negativity)
- Greek symmetry (Γ_call = Γ_put, ν_call = ν_put)
- Numerical vega vs finite-difference approximation
- IV round-trip to < 0.0001 accuracy across 5 vol levels
- Edge cases: intrinsic-only prices, expired options

The volatility and hedging suites check, on synthetic data where the true answer is known:
- No look-ahead: forward realized vol uses only returns *after* the entry date
- HAC standard errors match classical ones for i.i.d. data and are several times larger for autocorrelated or overlapping-window data
- A delta-hedged option earns about zero when realized vol equals implied vol, independent of the real-world drift
- Short-vol profits when implied > realized and loses when implied < realized
- Simulated P&L matches the analytic gamma P&L formula trade by trade
- Transaction costs can only reduce P&L

---

## Project structure

```
options_pricer/
├── black_scholes.py      # Core BSM engine
├── market_data.py        # Live data via yfinance
├── visualise.py          # Plotting
├── vol_analysis.py       # Realized vol, VRP, HAC statistics
├── hedging.py            # Delta-hedged P&L simulator
├── history.py            # Historical data loader (+ synthetic market)
├── run_vrp_backtest.py   # VRP study + hedged-P&L backtest entry point
├── main.py               # Pipeline entry point
├── test_black_scholes.py # Unit tests: pricer
├── test_vol_analysis.py  # Unit tests: realized vol / HAC
├── test_hedging.py       # Unit tests: hedging simulator
├── test_history.py       # Unit tests: data loader
├── requirements.txt
└── plots/                # Generated on first run
    ├── vol_smile.png
    ├── vol_surface.png
    ├── greeks_profile.png
    ├── model_fit.png
    ├── iv_vs_rv.png       # from run_vrp_backtest.py
    ├── vrp.png
    └── hedged_pnl.png
```

---

## Implied vs realized volatility and delta-hedged P&L

Does SPY implied volatility overstate the volatility that is subsequently realized, and can that gap be harvested by selling options and delta-hedging them? `run_vrp_backtest.py` answers both parts.

```bash
python run_vrp_backtest.py                          # SPY + VIX, 2010 -> today (needs internet)
python run_vrp_backtest.py --start 2015-01-01 --tc-bps 1
python run_vrp_backtest.py --hedge-vol daily --moneyness 1.05
python run_vrp_backtest.py --synthetic              # offline pipeline check on simulated data
```

### Method

**Variance risk premium.** At each date, implied vol is the VIX (30-day implied vol on the S&P 500). Realized vol is the annualised root-mean-square of daily log returns over the *next* 21 trading days, which is the period an option bought that day would live through. The VRP is IV − RV in vol points and IV² − RV² in variance points. SPY prices are dividend-adjusted, so the series behaves like a total-return index.

**Inference.** Daily observations of a 21-day window overlap by 20/21, so residuals are strongly autocorrelated and naive t-stats are wildly overstated. All t-stats use Newey-West (HAC) standard errors with 21 lags. Two forecast regressions, `RV = a + b·IV` and `RV = a + b·IV + c·RV_trailing`, test whether implied vol is an unbiased forecast (a = 0, b = 1) and whether past realized vol adds information.

**Delta-hedged short call.** Sell a 21-day call at the prevailing implied vol, hedge with the underlying each day using BSM delta, hold to expiry, and measure P&L at expiry as a percentage of spot. Cash earns the T-bill rate, and optional proportional costs are charged on every hedge trade. Headline statistics use non-overlapping trades. The mean is cross-checked on all overlapping start dates with a Newey-West standard error.

### Why you can trust the engine

For a continuously hedged option, P&L is approximately ∫ ½ Γ S² (σ²_implied − σ²_realized) dt. The test suite verifies that the simulator reproduces this formula trade by trade, earns about zero when σ_implied = σ_realized regardless of drift, and has the right sign when they differ.

### Limitations of this study

- **VIX is not an at-the-money implied vol.** It is a model-free variance measure that includes out-of-the-money puts, so it typically sits above ATM implied vol. Using it to price an ATM option overstates the premium collected.
- **No option bid/ask.** Options are priced at the VIX-implied BSM value, so results are an upper bound on what a real desk could capture. `--tc-bps` covers hedge costs in the underlying only.
- **Timing.** Daily closes only. The VIX is computed from option prices at a slightly different time than the SPY close.
- **Constant r per trade**, set at entry, and a single underlying.
- **Sample dependence.** A short-vol strategy looks best in calm regimes. Results depend heavily on whether the sample includes volatility spikes, so look at the worst trades the script prints, not only the mean.
- **Overlapping observations** reduce the effective sample size even after HAC corrections.

---

## Limitations 

This implements the **vanilla BSM model** as starting point. Here are the limitations:

- **No dividends** — extend to continuous dividend yield (BSM-Merton)
- **European options only** — American options require binomial trees / PDE methods 
- **Constant volatility** — real markets show stochastic vol (Heston model) and local vol (Dupire)
- **No jumps** — Merton jump-diffusion extends BSM to handle discontinuous price moves
- **Flat term structure** — a full implementation would use bootstrapped zero curves
