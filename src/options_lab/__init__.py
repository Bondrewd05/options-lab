"""Precificação de opções europeias sobre um ativo sem dividendos."""

from .pricing import binomial_crr_price, black_scholes_price, european_payoff
from .greeks import Greeks, black_scholes_greeks
from .simulation import MonteCarloEstimate, monte_carlo_price, simulate_gbm_path
from .hedging import HedgeRecord, HedgeResult, delta_hedge_path
from .experiments import run_hedging_experiments

__all__ = [
    "black_scholes_price", "binomial_crr_price", "european_payoff",
    "Greeks", "black_scholes_greeks",
    "MonteCarloEstimate", "monte_carlo_price",
    "simulate_gbm_path", "HedgeRecord", "HedgeResult", "delta_hedge_path",
    "run_hedging_experiments",
]
