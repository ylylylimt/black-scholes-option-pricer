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
| `test_black_scholes.py` | 25 unit tests covering pricing accuracy, Greek properties, IV round-trips |

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

25 tests covering:
- Analytical price accuracy vs known values
- Put-call parity (ATM and ITM)
- Greek signs and relationships (delta bounds, gamma positivity, theta negativity)
- Greek symmetry (Γ_call = Γ_put, ν_call = ν_put)
- Numerical vega vs finite-difference approximation
- IV round-trip to < 0.0001 accuracy across 5 vol levels
- Edge cases: intrinsic-only prices, expired options

---

## Project structure

```
options_pricer/
├── black_scholes.py      # Core BSM engine
├── market_data.py        # Live data via yfinance
├── visualise.py          # Plotting
├── main.py               # Pipeline entry point
├── test_black_scholes.py # Unit tests
├── requirements.txt
└── plots/                # Generated on first run
    ├── vol_smile.png
    ├── vol_surface.png
    ├── greeks_profile.png
    └── model_fit.png
```

---

## Limitations 

This implements the **vanilla BSM model** as starting point. Here are the limitations:

- **No dividends** — extend to continuous dividend yield (BSM-Merton)
- **European options only** — American options require binomial trees / PDE methods 
- **Constant volatility** — real markets show stochastic vol (Heston model) and local vol (Dupire)
- **No jumps** — Merton jump-diffusion extends BSM to handle discontinuous price moves
- **Flat term structure** — a full implementation would use bootstrapped zero curves
