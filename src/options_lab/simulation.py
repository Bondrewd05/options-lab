"""Preços Monte Carlo e trajetórias do movimento browniano geométrico sob Q."""

import math
import random
from dataclasses import dataclass
from numbers import Integral
from statistics import NormalDist

from .pricing import OptionType, _finite_real, _validate_model_inputs, black_scholes_price


_Z_975 = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class MonteCarloEstimate:
    """Estimativa de preço e incerteza amostral, em unidades monetárias.

    confidence_interval é o par (limite inferior, limite superior) do intervalo
    normal aproximado de 95%. Ele não é truncado em zero. standard_error mede
    a incerteza da média estimada, não a dispersão dos pagamentos individuais.
    """

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_paths: int
    seed: int | None
    confidence_level: float = 0.95


def monte_carlo_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    n_paths: int = 100_000,
    option_type: OptionType = "call",
    *,
    seed: int | None = None,
) -> MonteCarloEstimate:
    """Estima o preço de uma opção europeia sem dividendos por Monte Carlo.

    As entradas financeiras seguem black_scholes_price. n_paths é inteiro
    >= 2; seed é inteiro ou None. Booleanos não são aceitos nesses argumentos.
    Usa random.Random(seed).gauss(0, 1), sem alterar o gerador global.

    Cada sorteio representa o preço terminal exato do movimento browniano
    geométrico sob a medida neutra ao risco. A implementação calcula o payoff
    já descontado por uma expressão equivalente, evitando multiplicar um
    preço terminal muito grande por um desconto muito pequeno.

    A média e a variância amostral (denominador n_paths - 1) são atualizadas
    pelo algoritmo de Welford: tempo O(n_paths) e memória O(1). O intervalo
    normal aproximado de 95% é price +/- 1.95996... * standard_error.

    Prazo ou volatilidade zero retornam o preço determinístico, erro padrão
    zero e intervalo degenerado. As entradas são validadas antes dos atalhos.

    Com volatilidade e prazo positivos, todos os payoffs observados podem ser
    zero. Nesse caso, erro padrão e intervalo empíricos também são zero: isso
    NÃO prova que o preço verdadeiro seja zero. A aproximação normal pode ser
    ruim em amostras pequenas ou quando pagamentos positivos são muito raros.

    Reproduza resultados com seed, parâmetros e versão de Python registrados.
    O módulo random não garante todos os seus algoritmos entre versões.

    Raises:
        TypeError: Tipos inválidos para entradas, n_paths ou seed.
        ValueError: Valores não finitos, domínio financeiro inválido ou n_paths < 2.
        OverflowError: Cálculo fora do alcance numérico de float.
    """
    spot, strike, time_to_maturity, rate, volatility = _validate_model_inputs(
        spot, strike, time_to_maturity, rate, volatility, option_type
    )
    if isinstance(n_paths, bool) or not isinstance(n_paths, Integral):
        raise TypeError("n_paths deve ser um inteiro maior ou igual a 2.")
    n_paths = int(n_paths)
    if n_paths < 2:
        raise ValueError("n_paths deve ser maior ou igual a 2.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed deve ser um inteiro ou None.")
        seed = int(seed)

    if time_to_maturity == 0.0 or volatility == 0.0:
        price = black_scholes_price(
            spot, strike, time_to_maturity, rate, volatility, option_type
        )
        return MonteCarloEstimate(price, 0.0, (price, price), n_paths, seed)

    integrated_rate = rate * time_to_maturity
    total_volatility = volatility * math.sqrt(time_to_maturity)
    total_variance = total_volatility * total_volatility
    if not all(math.isfinite(value) for value in (integrated_rate, total_volatility, total_variance)):
        raise OverflowError("Juros ou volatilidade acumulados excedem o alcance de float.")
    if total_volatility == 0.0:
        raise OverflowError("A volatilidade acumulada foi arredondada a zero em float.")

    try:
        discounted_strike = math.exp(math.log(strike) - integrated_rate)
        log_discounted_spot = math.log(spot) - 0.5 * total_variance
        generator = random.Random(seed)
        mean = 0.0
        squared_deviations = 0.0
        for count in range(1, n_paths + 1):
            normal = generator.gauss(0.0, 1.0)
            # exp(-r*tau)*S_T = S*exp(-sigma²*tau/2 + sigma*sqrt(tau)*Z).
            discounted_spot = math.exp(log_discounted_spot + total_volatility * normal)
            if option_type == "call":
                payoff = max(discounted_spot - discounted_strike, 0.0)
            else:
                payoff = max(discounted_strike - discounted_spot, 0.0)
            deviation = payoff - mean
            mean += deviation / count
            squared_deviations += deviation * (payoff - mean)
    except OverflowError as error:
        raise OverflowError("Um pagamento ou fator de Monte Carlo excede o alcance de float.") from error

    if not math.isfinite(mean) or not math.isfinite(squared_deviations):
        raise OverflowError("A média ou variância amostral excede o alcance de float.")
    standard_error = math.sqrt(max(squared_deviations / n_paths / (n_paths - 1), 0.0))
    margin = _Z_975 * standard_error
    confidence_interval = (mean - margin, mean + margin)
    if not all(math.isfinite(value) for value in confidence_interval):
        raise OverflowError("O intervalo de confiança excede o alcance de float.")
    return MonteCarloEstimate(mean, standard_error, confidence_interval, n_paths, seed)


def simulate_gbm_path(
    spot: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    steps: int = 252,
    *,
    seed: int | None = None,
) -> tuple[float, ...]:
    """Sorteia uma trajetória de preços em uma grade uniforme sob a medida Q.

    Retorna steps+1 preços, incluindo spot no instante inicial. Cada incremento
    usa a transição exata do modelo, sem aproximação de Euler. Há preços apenas
    nas datas da grade; a trajetória contínua entre elas não é armazenada.

    spot e time_to_maturity devem ser positivos; volatility pode ser zero;
    rate aceita qualquer real finito. steps é inteiro >=1, seed inteiro ou None.
    Booleanos são rejeitados. O RNG local não altera o gerador global.
    Com volatilidade zero, a trajetória é determinística e dispensa sorteios.

    A distribuição descreve cenários de precificação sob Q, não uma estimativa
    do retorno real do ativo. Um preço arredondado a zero ou não representável
    gera OverflowError, pois a trajetória deve permanecer positiva e finita.
    """
    spot = _finite_real("spot", spot)
    time_to_maturity = _finite_real("time_to_maturity", time_to_maturity)
    rate = _finite_real("rate", rate)
    volatility = _finite_real("volatility", volatility)
    if spot <= 0.0 or time_to_maturity <= 0.0:
        raise ValueError("spot e time_to_maturity devem ser estritamente positivos.")
    if volatility < 0.0:
        raise ValueError("volatility deve ser não negativa.")
    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise TypeError("steps deve ser um inteiro positivo.")
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps deve ser estritamente positivo.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed deve ser um inteiro ou None.")
        seed = int(seed)

    dt = time_to_maturity / steps
    step_volatility = volatility * math.sqrt(dt)
    step_growth = rate * dt
    step_variance = step_volatility * step_volatility
    if dt == 0.0 or (volatility > 0.0 and step_volatility == 0.0):
        raise OverflowError("O intervalo ou sua volatilidade foi arredondado a zero em float.")
    if not all(math.isfinite(value) for value in (step_volatility, step_growth, step_variance)):
        raise OverflowError("Os parâmetros por intervalo excedem o alcance de float.")

    generator = random.Random(seed)
    log_initial_spot = math.log(spot)
    log_spot = log_initial_spot
    path = [spot]
    try:
        for index in range(1, steps + 1):
            if volatility == 0.0:
                time = time_to_maturity if index == steps else index * dt
                log_spot = log_initial_spot + rate * time
            else:
                log_spot += step_growth - 0.5 * step_variance + step_volatility * generator.gauss(0.0, 1.0)
            current_spot = math.exp(log_spot)
            if current_spot == 0.0 or not math.isfinite(current_spot):
                raise OverflowError("Um preço simulado saiu do intervalo positivo finito de float.")
            path.append(current_spot)
    except OverflowError as error:
        raise OverflowError("Um preço simulado excede o alcance numérico de float.") from error
    return tuple(path)
