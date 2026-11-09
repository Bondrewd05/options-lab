"""Accounting, replication, and contract checks for a short-call delta hedge."""

import math
import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import product

from options_lab import (
    HedgeRecord,
    HedgeResult,
    black_scholes_greeks,
    black_scholes_price,
    delta_hedge_path,
)


BASE_PARAMETERS = {
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
BASE_PATH = (100.0, 97.0, 106.0, 99.0, 118.0)
INVALID_NUMERIC_TYPES = (True, False, None, "100", 1j, [], {}, Decimal("1"))


class DeltaHedgePathTests(unittest.TestCase):
    def test_public_result_and_frozen_records(self):
        result = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        self.assertIsInstance(result, HedgeResult)
        self.assertTrue(is_dataclass(result))
        self.assertEqual(
            {field.name for field in fields(result)},
            {
                "initial_premium", "payoff", "terminal_pnl", "records",
                "total_transaction_cost", "terminal_transaction_cost",
            },
        )
        self.assertIsInstance(result.records, tuple)
        self.assertEqual(len(result.records), len(BASE_PATH))
        for record in result.records:
            self.assertIsInstance(record, HedgeRecord)
            self.assertTrue(is_dataclass(record))
        self.assertEqual(
            {field.name for field in fields(result.records[0])},
            {
                "time", "spot", "shares", "trade", "interest", "option_cash_flow",
                "cash", "option_value", "portfolio_value", "net_value", "transaction_cost",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            result.records[0].cash = 0.0

    def test_initial_position_is_financed_by_call_premium(self):
        expected_premium = black_scholes_price(BASE_PATH[0], **BASE_PARAMETERS)
        expected_delta = black_scholes_greeks(BASE_PATH[0], **BASE_PARAMETERS).delta
        result = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        initial = result.records[0]
        self.assertEqual(result.initial_premium, expected_premium)
        self.assertEqual(initial.time, 0.0)
        self.assertEqual(initial.interest, 0.0)
        self.assertEqual(initial.shares, expected_delta)
        self.assertEqual(initial.trade, expected_delta)
        self.assertEqual(initial.option_cash_flow, expected_premium)
        self.assertAlmostEqual(initial.cash, expected_premium - expected_delta * BASE_PATH[0])
        self.assertAlmostEqual(initial.portfolio_value, expected_premium)
        self.assertAlmostEqual(initial.net_value, 0.0)

    def test_uniform_times_and_pre_expiry_delta_targets(self):
        for path, maturity in ((BASE_PATH, 1.5), ((100, 110), 0.5)):
            parameters = {**BASE_PARAMETERS, "time_to_maturity": maturity}
            result = delta_hedge_path(path, **parameters)
            step_time = maturity / (len(path) - 1)
            with self.subTest(path=path, maturity=maturity):
                for index, record in enumerate(result.records):
                    self.assertAlmostEqual(record.time, index * step_time, delta=1e-14)
                    self.assertEqual(record.spot, path[index])
                for index, record in enumerate(result.records[:-1]):
                    expected_delta = black_scholes_greeks(
                        path[index], 100, maturity - index * step_time, 0.05, 0.2
                    ).delta
                    self.assertAlmostEqual(record.shares, expected_delta, delta=1e-13)
                self.assertEqual(result.records[-1].shares, 0.0)

    def test_terminal_record_liquidates_stock_and_pays_call(self):
        for terminal_spot in (80, 100, 120):
            result = delta_hedge_path((100, 105, terminal_spot), **BASE_PARAMETERS)
            previous, terminal = result.records[-2:]
            expected_payoff = max(terminal_spot - 100, 0)
            with self.subTest(terminal_spot=terminal_spot):
                self.assertEqual(result.payoff, expected_payoff)
                self.assertEqual(terminal.time, 1.0)
                self.assertEqual(terminal.shares, 0.0)
                self.assertEqual(terminal.trade, -previous.shares)
                self.assertEqual(terminal.option_cash_flow, -expected_payoff)
                self.assertEqual(terminal.option_value, 0.0)
                self.assertEqual(terminal.portfolio_value, terminal.cash)
                self.assertEqual(terminal.net_value, terminal.cash)
                self.assertEqual(result.terminal_pnl, terminal.cash)

    def test_cash_flows_are_self_financing_at_every_date(self):
        for rate in (-0.05, 0.0, 0.05):
            parameters = {**BASE_PARAMETERS, "rate": rate}
            result = delta_hedge_path(BASE_PATH, **parameters)
            step_time = 1 / (len(BASE_PATH) - 1)
            previous_cash = 0.0
            previous_shares = 0.0
            with self.subTest(rate=rate):
                for index, record in enumerate(result.records):
                    expected_interest = previous_cash * math.expm1(rate * step_time)
                    self.assertAlmostEqual(record.interest, expected_interest, delta=1e-12)
                    self.assertAlmostEqual(record.trade, record.shares - previous_shares)
                    expected_cash = (
                        previous_cash + expected_interest
                        - record.trade * record.spot + record.option_cash_flow
                    )
                    self.assertAlmostEqual(record.cash, expected_cash, delta=1e-11)
                    if 0 < index < len(result.records) - 1:
                        self.assertEqual(record.option_cash_flow, 0.0)
                    previous_cash = record.cash
                    previous_shares = record.shares

    def test_terminal_pnl_matches_independent_discounted_gains_identity(self):
        # Build discounted trading gains from prices and prescribed deltas,
        # without referring to the simulator's cash records or cash recurrence.
        for path, rate in product(
            (BASE_PATH, (100, 90, 75), (100, 110, 125, 140)), (-0.04, 0.06)
        ):
            maturity, strike, volatility = 1.5, 100, 0.3
            premium = black_scholes_price(path[0], strike, maturity, rate, volatility)
            step_time = maturity / (len(path) - 1)
            discounted_gains = []
            for index in range(len(path) - 1):
                time = index * step_time
                next_time = (index + 1) * step_time
                delta = black_scholes_greeks(
                    path[index], strike, maturity - time, rate, volatility
                ).delta
                discounted_gains.append(
                    delta * (
                        math.exp(-rate * next_time) * path[index + 1]
                        - math.exp(-rate * time) * path[index]
                    )
                )
            expected = (
                (premium + math.fsum(discounted_gains)) * math.exp(rate * maturity)
                - max(path[-1] - strike, 0)
            )
            with self.subTest(path=path, rate=rate):
                result = delta_hedge_path(path, strike, maturity, rate, volatility)
                self.assertAlmostEqual(result.terminal_pnl, expected, delta=1e-10)

    def test_one_step_matches_buy_and_hold_portfolio(self):
        for terminal_spot, rate in product((80, 100, 120), (-0.03, 0.05)):
            spot, strike, maturity, volatility = 100, 100, 1, 0.2
            premium = black_scholes_price(spot, strike, maturity, rate, volatility)
            delta = black_scholes_greeks(spot, strike, maturity, rate, volatility).delta
            expected = (
                delta * terminal_spot
                + (premium - delta * spot) * math.exp(rate * maturity)
                - max(terminal_spot - strike, 0)
            )
            with self.subTest(terminal_spot=terminal_spot, rate=rate):
                result = delta_hedge_path(
                    (spot, terminal_spot), strike, maturity, rate, volatility
                )
                self.assertEqual(len(result.records), 2)
                self.assertAlmostEqual(result.terminal_pnl, expected, delta=1e-11)

    def test_custom_premium_changes_pnl_by_its_financed_difference(self):
        for rate in (-0.04, 0.06):
            parameters = {**BASE_PARAMETERS, "rate": rate, "time_to_maturity": 1.5}
            baseline = delta_hedge_path(BASE_PATH, **parameters)
            for premium in (0.0, baseline.initial_premium + 7.5):
                result = delta_hedge_path(BASE_PATH, **parameters, initial_premium=premium)
                difference = premium - baseline.initial_premium
                with self.subTest(rate=rate, premium=premium):
                    self.assertEqual(result.initial_premium, premium)
                    self.assertAlmostEqual(
                        result.terminal_pnl - baseline.terminal_pnl,
                        difference * math.exp(rate * 1.5),
                        delta=1e-11,
                    )
                    self.assertAlmostEqual(result.records[0].net_value, difference, delta=1e-11)
                    self.assertEqual(result.records[0].option_value, baseline.initial_premium)
                    self.assertEqual(
                        tuple(record.shares for record in result.records),
                        tuple(record.shares for record in baseline.records),
                    )

    def test_portfolio_and_net_values_match_stock_cash_and_call_liability(self):
        result = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        for record in result.records:
            with self.subTest(time=record.time):
                self.assertAlmostEqual(
                    record.portfolio_value, record.shares * record.spot + record.cash
                )
                self.assertAlmostEqual(
                    record.net_value, record.portfolio_value - record.option_value
                )
        for record in result.records[:-1]:
            expected_liability = black_scholes_price(record.spot, 100, 1 - record.time, 0.05, 0.2)
            self.assertAlmostEqual(record.option_value, expected_liability, delta=1e-12)

    def test_accepts_sequences_without_mutating_input(self):
        path = list(BASE_PATH)
        original = path.copy()
        expected = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        self.assertEqual(delta_hedge_path(path, **BASE_PARAMETERS), expected)
        self.assertEqual(path, original)
        self.assertEqual(
            delta_hedge_path(range(100, 103), **BASE_PARAMETERS),
            delta_hedge_path((100, 101, 102), **BASE_PARAMETERS),
        )

    def test_accepts_real_financial_inputs(self):
        rational_result = delta_hedge_path(
            (Fraction(100), Fraction(110)), Fraction(100), Fraction(1),
            Fraction(1, 20), Fraction(1, 5), initial_premium=Fraction(10),
        )
        expected = delta_hedge_path((100, 110), **BASE_PARAMETERS, initial_premium=10)
        self.assertEqual(rational_result, expected)

    def test_rejects_invalid_path_containers_and_short_sequences(self):
        for path in (None, 100, "100,110", b"100", {100, 110}, {0: 100, 1: 110}, iter((100, 110))):
            with self.subTest(path=path), self.assertRaises(TypeError):
                delta_hedge_path(path, **BASE_PARAMETERS)
        for path in ([], (), [100], (100,)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                delta_hedge_path(path, **BASE_PARAMETERS)

    def test_validates_every_path_entry(self):
        for index, value in product(range(3), INVALID_NUMERIC_TYPES):
            path = [100, 105, 110]
            path[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(TypeError):
                delta_hedge_path(path, **BASE_PARAMETERS)
        for index, value in product(range(3), (0, -1, math.nan, math.inf, -math.inf)):
            path = [100, 105, 110]
            path[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                delta_hedge_path(path, **BASE_PARAMETERS)

    def test_rejects_nonreal_financial_inputs(self):
        for name, value in product(BASE_PARAMETERS, INVALID_NUMERIC_TYPES):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                delta_hedge_path(BASE_PATH, **parameters)

    def test_rejects_nonfinite_values_and_invalid_financial_domains(self):
        for name, value in product(BASE_PARAMETERS, (math.nan, math.inf, -math.inf)):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                delta_hedge_path(BASE_PATH, **parameters)
        for name, value in (
            ("strike", 0), ("strike", -1),
            ("time_to_maturity", 0), ("time_to_maturity", -1),
            ("volatility", 0), ("volatility", -0.1),
        ):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                delta_hedge_path(BASE_PATH, **parameters)

    def test_validates_custom_premium_and_accepts_none(self):
        self.assertEqual(
            delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, initial_premium=None),
            delta_hedge_path(BASE_PATH, **BASE_PARAMETERS),
        )
        for premium in (True, False, "10", 1j, [], {}, Decimal("10")):
            with self.subTest(premium=premium), self.assertRaises(TypeError):
                delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, initial_premium=premium)
        for premium in (-1, math.nan, math.inf, -math.inf):
            with self.subTest(premium=premium), self.assertRaises(ValueError):
                delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, initial_premium=premium)


class TransactionCostTests(unittest.TestCase):
    def test_zero_cost_rate_preserves_default_hedge(self):
        implicit = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        explicit = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=0.0)
        self.assertEqual(implicit, explicit)
        self.assertEqual(explicit.total_transaction_cost, 0.0)
        self.assertEqual(explicit.terminal_transaction_cost, 0.0)
        self.assertTrue(all(record.transaction_cost == 0.0 for record in explicit.records))

    def test_legacy_dataclass_constructors_default_to_zero_costs(self):
        record = HedgeRecord(0.0, 100.0, 0.5, 0.5, 0.0, 10.0, -40.0, 10.0, 10.0, 0.0)
        result = HedgeResult(10.0, 0.0, 0.0, (record,))
        self.assertEqual(record.transaction_cost, 0.0)
        self.assertEqual(result.total_transaction_cost, 0.0)
        self.assertEqual(result.terminal_transaction_cost, 0.0)

    def test_costs_apply_to_buys_sells_and_zero_trades(self):
        cost_rate = 0.003
        result = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=cost_rate)
        self.assertTrue(any(record.trade > 0 for record in result.records))
        self.assertTrue(any(record.trade < 0 for record in result.records))
        for record in result.records:
            with self.subTest(time=record.time, trade=record.trade):
                self.assertAlmostEqual(
                    record.transaction_cost, cost_rate * abs(record.trade) * record.spot,
                    delta=1e-12,
                )
                self.assertGreaterEqual(record.transaction_cost, 0.0)
        constant_delta = delta_hedge_path(
            (1000, 1000, 1000), 100, 0.01, 0, 0.1, transaction_cost_rate=cost_rate
        )
        self.assertEqual(constant_delta.records[1].trade, 0.0)
        self.assertEqual(constant_delta.records[1].transaction_cost, 0.0)

    def test_cash_ledger_debits_costs_and_finances_them(self):
        for rate, cost_rate in product((-0.05, 0.0, 0.05), (0.001, 0.01)):
            parameters = {**BASE_PARAMETERS, "rate": rate}
            result = delta_hedge_path(
                BASE_PATH, **parameters, transaction_cost_rate=cost_rate
            )
            previous_cash = 0.0
            previous_shares = 0.0
            step_time = 1 / (len(BASE_PATH) - 1)
            for record in result.records:
                with self.subTest(rate=rate, cost_rate=cost_rate, time=record.time):
                    trade = record.shares - previous_shares
                    interest = previous_cash * math.expm1(rate * step_time)
                    cost = cost_rate * abs(trade) * record.spot
                    expected_cash = (
                        previous_cash + interest - trade * record.spot
                        + record.option_cash_flow - cost
                    )
                    self.assertAlmostEqual(record.interest, interest, delta=1e-12)
                    self.assertAlmostEqual(record.cash, expected_cash, delta=1e-11)
                    self.assertAlmostEqual(
                        record.portfolio_value, record.shares * record.spot + record.cash
                    )
                    previous_cash = record.cash
                    previous_shares = record.shares
            self.assertEqual(result.terminal_pnl, result.records[-1].cash)
            self.assertAlmostEqual(
                result.records[0].net_value, -result.records[0].transaction_cost, delta=1e-11
            )

    def test_cost_rate_preserves_deltas_option_values_and_premium(self):
        baseline = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS)
        result = delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=0.01)
        self.assertEqual(result.initial_premium, baseline.initial_premium)
        self.assertEqual(result.payoff, baseline.payoff)
        for original, charged in zip(baseline.records, result.records):
            with self.subTest(time=original.time):
                self.assertEqual(charged.shares, original.shares)
                self.assertEqual(charged.trade, original.trade)
                self.assertEqual(charged.option_value, original.option_value)
                self.assertEqual(charged.option_cash_flow, original.option_cash_flow)

    def test_one_step_pays_opening_and_liquidation_costs(self):
        cost_rate = 0.005
        for terminal_spot, rate in product((80, 120), (-0.03, 0.05)):
            spot, strike, maturity, volatility = 100, 100, 1.5, 0.2
            delta = black_scholes_greeks(spot, strike, maturity, rate, volatility).delta
            premium = black_scholes_price(spot, strike, maturity, rate, volatility)
            opening_cost = cost_rate * delta * spot
            closing_cost = cost_rate * delta * terminal_spot
            expected_pnl = (
                (premium - delta * spot - opening_cost) * math.exp(rate * maturity)
                + delta * terminal_spot - max(terminal_spot - strike, 0) - closing_cost
            )
            with self.subTest(terminal_spot=terminal_spot, rate=rate):
                result = delta_hedge_path(
                    (spot, terminal_spot), strike, maturity, rate, volatility,
                    transaction_cost_rate=cost_rate,
                )
                self.assertAlmostEqual(result.records[0].transaction_cost, opening_cost)
                self.assertAlmostEqual(result.records[-1].transaction_cost, closing_cost)
                self.assertEqual(result.records[-1].shares, 0.0)
                self.assertEqual(result.records[-1].option_cash_flow, -result.payoff)
                self.assertAlmostEqual(result.total_transaction_cost, opening_cost + closing_cost)
                self.assertAlmostEqual(
                    result.terminal_transaction_cost,
                    opening_cost * math.exp(rate * maturity) + closing_cost,
                    delta=1e-12,
                )
                self.assertAlmostEqual(result.terminal_pnl, expected_pnl, delta=1e-11)

    def test_nominal_and_terminal_costs_explain_pnl_difference(self):
        for rate, cost_rate, premium in product((-0.05, 0.0, 0.05), (0.001, 0.02), (None, 0.0)):
            parameters = {**BASE_PARAMETERS, "rate": rate, "time_to_maturity": 1.5}
            baseline = delta_hedge_path(BASE_PATH, **parameters, initial_premium=premium)
            charged = delta_hedge_path(
                BASE_PATH, **parameters, initial_premium=premium, transaction_cost_rate=cost_rate
            )
            expected_nominal = math.fsum(record.transaction_cost for record in charged.records)
            expected_terminal = math.fsum(
                record.transaction_cost * math.exp(rate * (1.5 - record.time))
                for record in charged.records
            )
            with self.subTest(rate=rate, cost_rate=cost_rate, premium=premium):
                self.assertAlmostEqual(
                    charged.total_transaction_cost, expected_nominal, delta=1e-12
                )
                self.assertAlmostEqual(
                    charged.terminal_transaction_cost, expected_terminal, delta=1e-12
                )
                self.assertAlmostEqual(
                    baseline.terminal_pnl - charged.terminal_pnl,
                    charged.terminal_transaction_cost,
                    delta=1e-10,
                )

    def test_validates_cost_rate_and_accepts_real_scalars(self):
        self.assertEqual(
            delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=Fraction(1, 100)),
            delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=0.01),
        )
        for cost_rate in INVALID_NUMERIC_TYPES:
            with self.subTest(cost_rate=cost_rate), self.assertRaises(TypeError):
                delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=cost_rate)
        for cost_rate in (-0.01, math.nan, math.inf, -math.inf):
            with self.subTest(cost_rate=cost_rate), self.assertRaises(ValueError):
                delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=cost_rate)

    def test_reports_unrepresentable_transaction_cost(self):
        with self.assertRaises(OverflowError):
            delta_hedge_path(BASE_PATH, **BASE_PARAMETERS, transaction_cost_rate=1e308)


if __name__ == "__main__":
    unittest.main()
