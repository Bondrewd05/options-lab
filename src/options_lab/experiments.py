"""Experimentos pareados de frequência, custos e volatilidade do delta hedge.

Os parâmetros são fixos e didáticos. Não há calibração a dados de mercado nem
procura de uma estratégia ótima. A simulação ocorre sob a medida Q.
"""

import math
import platform
import statistics
from importlib.metadata import version
from numbers import Integral

from .hedging import delta_hedge_path
from .pricing import black_scholes_price
from .simulation import simulate_gbm_path


FREQUENCIES = (12, 52, 252)
COST_RATES = (0.0, 0.0005, 0.001, 0.002)
HEDGE_VOLATILITIES = (0.15, 0.20, 0.25)


def _summary(values):
    count = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    se = std / math.sqrt(count)
    margin = statistics.NormalDist().inv_cdf(0.975) * se
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "rmse_pnl": math.sqrt(math.fsum(value * value for value in values) / count),
        "mean_standard_error": se,
        "mean_confidence_interval_95_approx": [mean - margin, mean + margin],
        "percentile_5": quantiles[4],
        "median": statistics.median(values),
        "percentile_95": quantiles[94],
        "negative_fraction": sum(value < 0.0 for value in values) / count,
    }


def run_hedging_experiments(n_paths: int = 2000, *, seed_start: int = 2026) -> dict:
    """Compara estratégias nas mesmas trajetórias, retornando dados JSON-safe.

    n_paths é inteiro >=2; seed_start é inteiro >=0. Booleanos são rejeitados.
    Seeds consecutivas alimentam geradores locais. Sementes iniciais negativas
    são rejeitadas: sementes opostas geram a mesma sequência em random.Random,
    e atravessar zero repetiria trajetórias na amostra. A grade-mestra tem 3276
    intervalos, o MMC de 12, 52 e 252. Cada estratégia observa uma subgrade
    uniforme dessa mesma trajetória: mesmo início, mesmo vencimento e mesmos
    preços nos tempos compartilhados. As três grades não são aninhadas entre si.

    Frequência/custos: sigma=20%, prêmio BS(20%), 12/52/252 intervalos e custos
    0/5/10/20 pontos-base do valor de cada negociação em ações. Inclui compra
    inicial e liquidação. O delta permanece o BS sem ajuste para custos.

    Volatilidade: ativo com sigma=20%, hedge com 15/20/25%, 252 intervalos e
    custo zero. Compara prêmio fixo BS(20%) e prêmio calculado com sigma do
    hedge. Somente mudar o prêmio desloca o PnL por sua diferença capitalizada;
    não modifica os trades. Todos os valores de PnL são na data de vencimento.

    O relatório guarda as amostras na mesma ordem e diferenças pareadas.
    As estatísticas têm finalidade didática; não são previsões sob a medida P.
    """
    for name, value in (("n_paths", n_paths), ("seed_start", seed_start)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} deve ser inteiro.")
    n_paths, seed_start = int(n_paths), int(seed_start)
    if n_paths < 2:
        raise ValueError("n_paths deve ser maior ou igual a 2.")
    if seed_start < 0:
        raise ValueError("seed_start deve ser maior ou igual a zero.")

    parameters = dict(spot=100.0, strike=100.0, time_to_maturity=1.0,
                      rate=0.05, simulation_volatility=0.20)
    spot, strike, maturity, rate, sigma = parameters.values()
    premium = black_scholes_price(spot, strike, maturity, rate, sigma)
    master_steps = math.lcm(*FREQUENCIES)
    scenarios = []

    def scenario(identifier, family, steps, cost, hedge_sigma, premium_rule):
        received = premium if premium_rule == "fixed_true_model" else black_scholes_price(
            spot, strike, maturity, rate, hedge_sigma)
        return {
            "id": identifier, "family": family, "steps": steps,
            "transaction_cost_rate": cost, "hedge_volatility": hedge_sigma,
            "premium_convention": premium_rule, "initial_premium": received,
            "theoretical_mean_without_costs": (received - premium) * math.exp(rate * maturity),
            "terminal_pnls": [], "nominal_costs": [], "terminal_costs": [],
        }

    for steps in FREQUENCIES:
        for cost in COST_RATES:
            scenarios.append(scenario(
                f"n{steps}_cost{round(cost * 10000)}bp", "frequency_costs", steps,
                cost, sigma, "fixed_true_model"))
    for hedge_sigma in HEDGE_VOLATILITIES:
        for rule in ("fixed_true_model", "hedge_model"):
            scenarios.append(scenario(
                f"vol{round(hedge_sigma * 100)}_{rule}", "volatility", 252,
                0.0, hedge_sigma, rule))

    terminal_spots = []
    example_paths = None
    max_cost_identity_error = 0.0
    max_premium_identity_error = 0.0
    for index in range(n_paths):
        master = simulate_gbm_path(spot, maturity, rate, sigma,
                                   steps=master_steps, seed=seed_start + index)
        paths = {steps: master[::master_steps // steps] for steps in FREQUENCIES}
        if index == 0:
            example_paths = {str(steps): list(path) for steps, path in paths.items()}
        terminal_spots.append(master[-1])
        cache = {}
        for item in scenarios:
            key = (item["steps"], item["transaction_cost_rate"],
                   item["hedge_volatility"], item["initial_premium"])
            if key not in cache:
                cache[key] = delta_hedge_path(
                    paths[item["steps"]], strike, maturity, rate,
                    item["hedge_volatility"], initial_premium=item["initial_premium"],
                    transaction_cost_rate=item["transaction_cost_rate"],
                )
            result = cache[key]
            item["terminal_pnls"].append(result.terminal_pnl)
            item["nominal_costs"].append(result.total_transaction_cost)
            item["terminal_costs"].append(result.terminal_transaction_cost)
            if item["transaction_cost_rate"] > 0:
                no_cost = cache[(item["steps"], 0.0, sigma, premium)]
                error = abs(no_cost.terminal_pnl - result.terminal_pnl - result.terminal_transaction_cost)
                max_cost_identity_error = max(max_cost_identity_error, error)
            if item["family"] == "volatility" and item["premium_convention"] == "hedge_model":
                fixed = cache[(item["steps"], 0.0, item["hedge_volatility"], premium)]
                error = abs(result.terminal_pnl - fixed.terminal_pnl
                            - item["theoretical_mean_without_costs"])
                max_premium_identity_error = max(max_premium_identity_error, error)

    for item in scenarios:
        item["summary"] = _summary(item["terminal_pnls"])
        item["summary"].update({
            "mean_nominal_cost": statistics.mean(item["nominal_costs"]),
            "mean_terminal_cost": statistics.mean(item["terminal_costs"]),
            "mean_pnl_plus_terminal_cost": statistics.mean(
                pnl + cost for pnl, cost in zip(item["terminal_pnls"], item["terminal_costs"])),
        })

    by_id = {item["id"]: item for item in scenarios}
    paired_comparisons = []
    for left_id, right_id, quantity in (
        ("n252_cost0bp", "n12_cost0bp", "squared_pnl"),
        ("n252_cost10bp", "n12_cost10bp", "squared_pnl"),
        ("n52_cost10bp", "n52_cost0bp", "pnl"),
    ):
        left, right = by_id[left_id]["terminal_pnls"], by_id[right_id]["terminal_pnls"]
        differences = [a*a-b*b if quantity == "squared_pnl" else a-b for a, b in zip(left, right)]
        stats = _summary(differences)
        paired_comparisons.append({
            "left_id": left_id, "right_id": right_id,
            "quantity": quantity, "direction": "left minus right",
            "mean_difference": stats["mean_pnl"],
            "standard_error": stats["mean_standard_error"],
            "confidence_interval_95_approx": stats["mean_confidence_interval_95_approx"],
        })

    return {
        "schema_version": 1, "package_version": version("options-lab"),
        "python_version": platform.python_version(), "parameters": parameters,
        "measure": "Q; drift do ativo igual a rate",
        "n_paths": n_paths, "seed_start": seed_start,
        "seed_recipe": "seed_start + índice (0 até n_paths-1), na grade-mestra",
        "master_steps": master_steps, "frequencies": list(FREQUENCIES),
        "cost_rates": list(COST_RATES), "hedge_volatilities": list(HEDGE_VOLATILITIES),
        "true_model_premium": premium,
        "conventions": {
            "position": "Vendedor de uma call europeia, multiplicador 1, sem dividendos.",
            "cost": "kappa * abs(trade) * spot; inicial, ajustes e liquidação das ações.",
            "cash": "Mesmo juro contínuo para aplicação e empréstimo; nenhum aporte externo.",
            "pnl": "Dinheiro no vencimento após fechar ações e pagar a call.",
            "premium": "Fixo BS(sigma real), salvo cenários identificados hedge_model.",
            "marking": "option_value usa sigma do hedge; comparações usam somente PnL terminal.",
            "pairing": "Mesma trajetória-mestra por índice; subgrades uniformes, não aninhadas entre si.",
            "scope": "Parâmetros didáticos, sem calibração de mercado nem otimização da política.",
        },
        "terminal_spots": terminal_spots, "example_paths": example_paths,
        "scenarios": scenarios, "paired_comparisons": paired_comparisons,
        "checks": {"max_cost_identity_error": max_cost_identity_error,
                   "max_premium_identity_error": max_premium_identity_error},
    }
