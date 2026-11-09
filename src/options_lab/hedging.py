"""Contabilidade de delta hedge discreto para o vendedor de uma call europeia."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .greeks import black_scholes_greeks
from .pricing import _finite_real, _validate_model_inputs, black_scholes_price, european_payoff


@dataclass(frozen=True)
class HedgeRecord:
    """Estado após as operações da data, inclusive liquidação no vencimento.

    trade é a quantidade comprada (negativa se vendida). option_cash_flow é
    positivo no recebimento do prêmio e negativo no pagamento do payoff.
    option_value é o valor da obrigação ainda aberta: zero após liquidá-la.
    portfolio_value = shares*spot + cash; net_value = portfolio_value-option_value.
    transaction_cost é o custo da negociação desta data, debitado do caixa.
    """

    time: float
    spot: float
    shares: float
    trade: float
    interest: float
    option_cash_flow: float
    cash: float
    option_value: float
    portfolio_value: float
    net_value: float
    transaction_cost: float = 0.0


@dataclass(frozen=True)
class HedgeResult:
    """Resultado do vendedor, em moeda do vencimento, após encerrar as posições.

    total_transaction_cost soma os custos nas datas em que foram pagos.
    terminal_transaction_cost capitaliza cada custo à taxa rate até o vencimento.
    """

    initial_premium: float
    payoff: float
    terminal_pnl: float
    records: tuple[HedgeRecord, ...]
    total_transaction_cost: float = 0.0
    terminal_transaction_cost: float = 0.0


def delta_hedge_path(
    spot_path: Sequence[float],
    strike: float,
    time_to_maturity: float,
    rate: float,
    volatility: float,
    *,
    initial_premium: float | None = None,
    transaction_cost_rate: float = 0.0,
) -> HedgeResult:
    """Executa o delta hedge de uma call vendida sobre uma trajetória fornecida.

    spot_path é uma sequência de pelo menos dois preços positivos e finitos.
    As datas são uniformes entre zero e time_to_maturity. Juros contínuos,
    prazo em anos, uma opção sobre uma unidade do ativo, sem dividendos.
    Posições fracionárias e financiamento à taxa rate são permitidos.

    Prazo e volatilidade do hedge devem ser positivos. initial_premium é um
    real finito não negativo; se None, usa o preço BS da call no instante zero.
    Com prêmio informado, option_value continua sendo o valor BS da obrigação.
    transaction_cost_rate é um real finito não negativo. O custo de cada compra
    ou venda de ações é transaction_cost_rate * abs(trade) * spot, inclusive na
    abertura e na liquidação. O delta e a avaliação da opção seguem Black–Scholes.

    Em t=0 recebe o prêmio e compra delta ações. Em cada data interior, acumula
    juros sobre o caixa e compra/vende a diferença até o novo delta. No último
    instante, acumula juros, vende todas as ações e paga o payoff. Os custos são
    debitados em cada negociação. O delta é calculado nas datas anteriores ao
    vencimento. Todos os registros descrevem o estado após operações.

    A estratégia financia os ajustes pelo próprio caixa, sem novos aportes.
    terminal_pnl é o caixa final: pode ser positivo ou negativo. A grade finita
    pode deixar erro de replicação mesmo quando o modelo e a volatilidade estão
    corretos. O método usa tempo e memória O(len(spot_path)), incluindo o registro.

    Raises:
        TypeError: Sequência ou tipos numéricos inválidos.
        ValueError: Entradas não finitas ou fora do domínio.
        OverflowError: Cálculo fora do alcance numérico de float.
    """
    if isinstance(spot_path, (str, bytes, bytearray)) or not isinstance(spot_path, Sequence):
        raise TypeError("spot_path deve ser uma sequência numérica de preços.")
    if len(spot_path) < 2:
        raise ValueError("spot_path deve conter pelo menos dois preços.")
    path = tuple(_finite_real(f"spot_path[{index}]", spot) for index, spot in enumerate(spot_path))
    if any(spot <= 0.0 for spot in path):
        raise ValueError("Todos os preços de spot_path devem ser estritamente positivos.")
    _, strike, time_to_maturity, rate, volatility = _validate_model_inputs(
        path[0], strike, time_to_maturity, rate, volatility, "call"
    )
    if time_to_maturity == 0.0 or volatility == 0.0:
        raise ValueError("O hedge exige time_to_maturity e volatility estritamente positivos.")
    if initial_premium is not None:
        initial_premium = _finite_real("initial_premium", initial_premium)
        if initial_premium < 0.0:
            raise ValueError("initial_premium deve ser não negativo.")
    transaction_cost_rate = _finite_real("transaction_cost_rate", transaction_cost_rate)
    if transaction_cost_rate < 0.0:
        raise ValueError("transaction_cost_rate deve ser não negativo.")

    initial_option_value = black_scholes_price(path[0], strike, time_to_maturity, rate, volatility)
    if initial_premium is None:
        initial_premium = initial_option_value
    steps = len(path) - 1
    dt = time_to_maturity / steps
    if dt == 0.0:
        raise OverflowError("O intervalo de rebalanceamento foi arredondado a zero em float.")
    try:
        interest_factor = math.expm1(rate * dt)
    except OverflowError as error:
        raise OverflowError("O fator de juros excede o alcance de float.") from error
    if not math.isfinite(interest_factor):
        raise OverflowError("O fator de juros excede o alcance de float.")

    payoff = european_payoff(path[-1], strike)
    cash = 0.0
    previous_shares = 0.0
    total_transaction_cost = 0.0
    terminal_transaction_cost = 0.0
    records = []
    for index, spot in enumerate(path):
        time = time_to_maturity if index == steps else index * dt
        interest = 0.0 if index == 0 else cash * interest_factor
        if index == steps:
            shares = 0.0
            option_cash_flow = -payoff
            option_value = 0.0
        else:
            remaining = time_to_maturity if index == 0 else (steps - index) * dt
            shares = black_scholes_greeks(spot, strike, remaining, rate, volatility).delta
            option_value = initial_option_value if index == 0 else black_scholes_price(
                spot, strike, remaining, rate, volatility
            )
            option_cash_flow = initial_premium if index == 0 else 0.0
        trade = shares - previous_shares
        transaction_cost = transaction_cost_rate * abs(trade) * spot
        cash = cash + interest - trade * spot + option_cash_flow - transaction_cost
        total_transaction_cost += transaction_cost
        # O saldo anterior de custos capitaliza até esta data. Em t=0, ele é zero.
        terminal_transaction_cost += terminal_transaction_cost * interest_factor + transaction_cost
        portfolio_value = shares * spot + cash
        net_value = portfolio_value - option_value
        if not all(math.isfinite(value) for value in (
            shares, trade, interest, option_cash_flow, cash, option_value,
            portfolio_value, net_value, transaction_cost,
            total_transaction_cost, terminal_transaction_cost,
        )):
            raise OverflowError("Um saldo ou resultado do hedge excede o alcance de float.")
        records.append(HedgeRecord(
            time, spot, shares, trade, interest, option_cash_flow, cash,
            option_value, portfolio_value, net_value, transaction_cost,
        ))
        previous_shares = shares

    return HedgeResult(
        initial_premium, payoff, cash, tuple(records),
        total_transaction_cost, terminal_transaction_cost,
    )
