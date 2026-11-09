"""Payoffs e preços de Black–Scholes e da árvore CRR para entradas escalares.

Tempo em anos, juros continuamente compostos e volatilidade anualizada.
Cada opção representa uma unidade do ativo, sem dividendos.
"""

import math
from numbers import Integral, Real
from sys import float_info
from typing import Literal


OptionType = Literal["call", "put"]


def _finite_real(name: str, value: float) -> float:
    """Converte um escalar real finito, preservando erros de entrada claros."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} deve ser um número real escalar.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito.")
    return value


def _validate_option_type(option_type: OptionType) -> None:
    if not isinstance(option_type, str):
        raise TypeError("option_type deve ser uma string.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type deve ser 'call' ou 'put'.")


def _normal_cdf(value: float) -> float:
    """Distribuição acumulada normal padrão, calculada pela função erfc."""
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _log_moneyness(spot: float, strike: float) -> float:
    """Calcula log(S/K), preservando a diferença quando S está perto de K."""
    difference = spot - strike
    if abs(difference) <= 0.5 * strike:
        return math.log1p(difference / strike)
    # Preços distantes podem ter uma razão fora do alcance de float.
    return math.log(spot) - math.log(strike)


def _validate_model_inputs(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType,
) -> tuple[float, float, float, float, float]:
    """Domínio comum dos preços e sensibilidades do modelo."""
    _validate_option_type(option_type)
    spot = _finite_real("spot", spot)
    strike = _finite_real("strike", strike)
    time_to_maturity = _finite_real("time_to_maturity", time_to_maturity)
    rate = _finite_real("rate", rate)
    volatility = _finite_real("volatility", volatility)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot e strike devem ser estritamente positivos.")
    if time_to_maturity < 0.0 or volatility < 0.0:
        raise ValueError("time_to_maturity e volatility devem ser não negativos.")
    return spot, strike, time_to_maturity, rate, volatility


def european_payoff(
    terminal_spot: float,
    strike: float,
    option_type: OptionType = "call",
) -> float:
    """Retorna o pagamento ao comprador no vencimento, em unidades monetárias.

    terminal_spot aceita zero. strike deve ser estritamente positivo.
    O resultado é o pagamento contratual antes de contabilizar o prêmio.
    """
    _validate_option_type(option_type)
    terminal_spot = _finite_real("terminal_spot", terminal_spot)
    strike = _finite_real("strike", strike)
    if terminal_spot < 0.0:
        raise ValueError("terminal_spot deve ser não negativo.")
    if strike <= 0.0:
        raise ValueError("strike deve ser estritamente positivo.")

    if option_type == "call":
        return max(terminal_spot - strike, 0.0)
    return max(strike - terminal_spot, 0.0)


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
) -> float:
    """Retorna o preço de uma opção europeia pelo modelo de Black–Scholes.

    Args:
        spot: Preço atual do ativo, estritamente positivo.
        strike: Preço de exercício, estritamente positivo.
        time_to_maturity: Prazo restante em anos, não negativo.
        rate: Juros anuais continuamente compostos, em decimal.
        volatility: Volatilidade anualizada, em decimal e não negativa.
        option_type: 'call' ou 'put'.

    Returns:
        Valor da posição comprada, na mesma moeda de spot e strike.

    Raises:
        TypeError: Entrada fora dos tipos escalares aceitos.
        ValueError: Entrada não finita ou fora do domínio financeiro.
        OverflowError: Um cálculo intermediário excede a representação float.

    Em prazo zero, retorna o payoff. Em volatilidade zero, desconta o
    pagamento determinístico. Juros negativos são aceitos.
    """
    spot, strike, time_to_maturity, rate, volatility = _validate_model_inputs(
        spot, strike, time_to_maturity, rate, volatility, option_type
    )

    if time_to_maturity == 0.0:
        return european_payoff(spot, strike, option_type)

    integrated_rate = rate * time_to_maturity
    total_volatility = volatility * math.sqrt(time_to_maturity)
    if not math.isfinite(integrated_rate) or not math.isfinite(total_volatility):
        raise OverflowError("Juros ou volatilidade acumulados excedem o alcance de float.")

    try:
        discount = math.exp(-integrated_rate)
    except OverflowError:
        discount = math.inf
    if not math.isfinite(discount) or discount < float_info.min:
        # Um fator isolado subnormal/nulo ou infinito ainda pode produzir um
        # strike descontado representável. Preserva-se o cálculo usual fora
        # desses extremos, combinando os logaritmos apenas como alternativa.
        try:
            discounted_strike = math.exp(math.log(strike) - integrated_rate)
        except OverflowError as error:
            raise OverflowError("O strike descontado excede o alcance de float.") from error
    else:
        discounted_strike = strike * discount
    if not math.isfinite(discounted_strike):
        raise OverflowError("O strike descontado excede o alcance de float.")

    # Inclui volatilidade nula e eventual arredondamento de seu produto a zero.
    if total_volatility == 0.0:
        if option_type == "call":
            return max(spot - discounted_strike, 0.0)
        return max(discounted_strike - spot, 0.0)

    log_moneyness = _log_moneyness(spot, strike)
    d1 = (log_moneyness + integrated_rate) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility

    if option_type == "call":
        price = spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    else:
        price = discounted_strike * _normal_cdf(-d2) - spot * _normal_cdf(-d1)

    # Pequenos resíduos negativos podem surgir ao subtrair números próximos.
    return max(price, 0.0)


def binomial_crr_price(
    spot: float,
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    steps: int = 100,
    option_type: OptionType = "call",
) -> float:
    """Preço de uma opção europeia por árvore de Cox–Ross–Rubinstein (CRR).

    As cinco entradas financeiras e option_type usam as mesmas convenções de
    black_scholes_price. steps deve ser inteiro positivo; bool não é aceito.
    A árvore é recombinante e faz indução retroativa, com tempo O(steps**2)
    e memória O(steps). Cada opção corresponde a uma unidade do ativo.

    Para dt = time_to_maturity / steps e volatilidade positiva, a grade precisa
    satisfazer -volatility*sqrt(dt) < rate*dt < volatility*sqrt(dt). Fora dessa
    condição, aumente steps para construir uma árvore sem arbitragem.
    Não há exercício antecipado: os payoffs são calculados só no vencimento.

    Prazo zero retorna o payoff; volatilidade zero usa o valor determinístico.
    Essas fronteiras também validam todos os argumentos, inclusive steps.

    Raises:
        TypeError: Entrada fora dos tipos escalares aceitos ou steps não inteiro.
        ValueError: Domínio inválido ou grade incompatível com ausência de arbitragem.
        OverflowError: Um cálculo excede o alcance numérico de float.
    """
    spot, strike, time_to_maturity, rate, volatility = _validate_model_inputs(
        spot, strike, time_to_maturity, rate, volatility, option_type
    )
    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise TypeError("steps deve ser um inteiro positivo.")
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps deve ser estritamente positivo.")

    if time_to_maturity == 0.0:
        return european_payoff(spot, strike, option_type)
    if volatility == 0.0:
        return black_scholes_price(
            spot, strike, time_to_maturity, rate, volatility, option_type
        )

    dt = time_to_maturity / steps
    log_up = volatility * math.sqrt(dt)
    log_growth = rate * dt
    if not math.isfinite(log_up) or not math.isfinite(log_growth):
        raise OverflowError("Juros ou volatilidade por passo excedem o alcance de float.")
    if log_up == 0.0:
        raise OverflowError("O tamanho do passo foi arredondado a zero em float.")
    if not -log_up < log_growth < log_up:
        raise ValueError(
            "A grade CRR viola a condição de ausência de arbitragem "
            "(d < exp(rate*dt) < u). Aumente steps."
        )

    # p = (exp(b)-exp(-a))/(exp(a)-exp(-a)), a=log_up, b=log_growth.
    # A fatorização usa expoentes não positivos; expm1 evita cancelamento.
    denominator = -math.expm1(-2.0 * log_up)
    probability_up = (
        math.exp(log_growth - log_up)
        * -math.expm1(-log_growth - log_up)
        / denominator
    )
    # Calculamos q separadamente, para preservar pequenas probabilidades de baixa.
    probability_down = -math.expm1(log_growth - log_up) / denominator
    if probability_up == 0.0 or probability_down == 0.0:
        raise OverflowError("Uma probabilidade CRR positiva foi arredondada a zero em float.")

    try:
        discount = math.exp(-log_growth)
        if discount == 0.0:
            raise OverflowError("O fator de desconto foi arredondado a zero em float.")
        log_spot = math.log(spot)
        # No vencimento há steps+1 preços, pois uma alta e uma baixa recombinam.
        values = []
        for up_moves in range(steps + 1):
            terminal_spot = math.exp(log_spot + (2 * up_moves - steps) * log_up)
            if option_type == "call":
                values.append(max(terminal_spot - strike, 0.0))
            else:
                values.append(max(strike - terminal_spot, 0.0))

        # Os valores são ordenados do menor para o maior preço do ativo.
        # Atualizar da esquerda para a direita preserva os dois filhos necessários.
        for remaining in range(steps, 0, -1):
            for node in range(remaining):
                values[node] = discount * (
                    probability_down * values[node] + probability_up * values[node + 1]
                )
            values.pop()
    except OverflowError as error:
        raise OverflowError("Um nó ou fator da árvore CRR excede o alcance de float.") from error

    if not math.isfinite(values[0]):
        raise OverflowError("O preço da árvore CRR excede o alcance de float.")
    return values[0]
