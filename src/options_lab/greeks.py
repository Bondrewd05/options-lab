"""Derivadas do preço de Black–Scholes no domínio de prazo e vol positivos."""

import math
from dataclasses import dataclass

from .pricing import OptionType, _log_moneyness, _normal_cdf, _validate_model_inputs


@dataclass(frozen=True)
class Greeks:
    """Sensibilidades da posição comprada em uma opção sobre uma unidade do ativo.

    delta = dV/dS; gamma = d²V/dS²; vega = dV/dsigma.
    Vega usa sigma em decimal: multiplique por 0.01 para uma mudança de 1 pp.
    """

    delta: float
    gamma: float
    vega: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> Greeks:
    """Retorna delta, gamma e vega de uma call ou put europeia sem dividendos.

    Mesmas unidades e tipos de black_scholes_price. Aqui, prazo e volatilidade
    devem ser estritamente positivos. Essa restrição evita convenções especiais
    nas fronteiras, onde o preço pode apresentar pontos não diferenciáveis.

    Vega é a derivada por unidade DECIMAL de volatilidade, e não por ponto
    percentual: para ir de 20% a 21%, a aproximação é vega * 0.01.

    Raises:
        TypeError: Entrada fora dos tipos escalares aceitos.
        ValueError: Entrada não finita ou fora do domínio acima.
        OverflowError: Cálculo fora do alcance numérico de float.
    """
    spot, strike, time_to_maturity, rate, volatility = _validate_model_inputs(
        spot, strike, time_to_maturity, rate, volatility, option_type
    )
    if time_to_maturity == 0.0 or volatility == 0.0:
        raise ValueError(
            "As Greeks exigem time_to_maturity e volatility estritamente positivos."
        )

    integrated_rate = rate * time_to_maturity
    total_volatility = volatility * math.sqrt(time_to_maturity)
    if not math.isfinite(integrated_rate) or not math.isfinite(total_volatility):
        raise OverflowError("Juros ou volatilidade acumulados excedem o alcance de float.")
    if total_volatility == 0.0:
        raise OverflowError("A volatilidade acumulada foi arredondada a zero em float.")

    d1 = (
        (_log_moneyness(spot, strike) + integrated_rate) / total_volatility
        + 0.5 * total_volatility
    )
    delta = _normal_cdf(d1) if option_type == "call" else -_normal_cdf(-d1)

    # phi(d1)/(S*sigma*sqrt(tau)) e S*phi(d1)*sqrt(tau).
    # Logaritmos evitam produtos intermediários grandes ou pequenos demais.
    log_density = -0.5 * d1 * d1 - 0.5 * math.log(2.0 * math.pi)
    try:
        gamma = math.exp(log_density - math.log(spot) - math.log(total_volatility))
        vega = math.exp(log_density + math.log(spot) + 0.5 * math.log(time_to_maturity))
    except OverflowError as error:
        raise OverflowError("Uma sensibilidade excede o alcance de float.") from error
    return Greeks(delta=delta, gamma=gamma, vega=vega)
