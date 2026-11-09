"""Reference, finite-difference, and contract tests for Black–Scholes Greeks."""

import math
import unittest
from dataclasses import fields, is_dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

from options_lab import Greeks, black_scholes_greeks, black_scholes_price


BASE_PARAMETERS = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
OPTION_TYPES = ("call", "put")
INVALID_NUMERIC_TYPES = (True, False, None, "100", 1j, [], {}, Decimal("1"))
PARAMETER_GRID = tuple(
    product(
        (70.0, 100.0, 140.0),
        (80.0, 100.0, 120.0),
        (0.25, 1.0, 3.0),
        (-0.03, 0.0, 0.07),
        (0.15, 0.3, 0.6),
    )
)


class BlackScholesGreeksTests(unittest.TestCase):
    def test_public_api_returns_greeks_dataclass(self):
        result = black_scholes_greeks(**BASE_PARAMETERS)
        self.assertTrue(is_dataclass(Greeks))
        self.assertIsInstance(result, Greeks)
        self.assertEqual({field.name for field in fields(result)}, {"delta", "gamma", "vega"})
        for value in (result.delta, result.gamma, result.vega):
            self.assertTrue(math.isfinite(value))

    def test_reference_values_and_vega_units(self):
        # Vega is the derivative with respect to decimal volatility, so a unit
        # change represents 100 percentage points rather than one percentage point.
        for option_type, expected_delta in (
            ("call", 0.636830651175619),
            ("put", -0.363169348824381),
        ):
            with self.subTest(option_type=option_type):
                result = black_scholes_greeks(**BASE_PARAMETERS, option_type=option_type)
                self.assertAlmostEqual(result.delta, expected_delta, delta=1e-13)
                self.assertAlmostEqual(result.gamma, 0.0187620173458469, delta=1e-13)
                self.assertAlmostEqual(result.vega, 37.5240346916938, delta=1e-11)

    def test_nearby_float_prices_keep_their_moneyness_at_tiny_volatility(self):
        strike, volatility = 100.0, 1e-16
        for spot in (math.nextafter(strike, 0.0), math.nextafter(strike, math.inf)):
            # The independent reference computes the ratio and its logarithm
            # in Decimal; separate float logarithms lose this price movement.
            with localcontext() as context:
                context.prec = 80
                log_ratio = (Decimal.from_float(spot) / Decimal.from_float(strike)).ln()
                d1 = float(log_ratio / Decimal.from_float(volatility)) + volatility / 2
            density = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
            for option_type in OPTION_TYPES:
                with self.subTest(spot=spot, option_type=option_type):
                    expected_delta = (0.5 * math.erfc(-d1 / math.sqrt(2))
                                      if option_type == "call" else -0.5 * math.erfc(d1 / math.sqrt(2)))
                    actual = black_scholes_greeks(spot, strike, 1, 0, volatility, option_type)
                    self.assertAlmostEqual(actual.delta, expected_delta, delta=2e-15)
                    self.assertTrue(math.isclose(actual.gamma, density / (spot * volatility), rel_tol=1e-13))
                    self.assertTrue(math.isclose(actual.vega, spot * density, rel_tol=1e-13))

    def test_default_option_type_is_call(self):
        self.assertEqual(
            black_scholes_greeks(**BASE_PARAMETERS),
            black_scholes_greeks(**BASE_PARAMETERS, option_type="call"),
        )

    def test_accepts_real_scalars(self):
        integer_result = black_scholes_greeks(100, 100, 1, 0, 1)
        float_result = black_scholes_greeks(100.0, 100.0, 1.0, 0.0, 1.0)
        self.assertEqual(integer_result, float_result)
        rational_result = black_scholes_greeks(
            Fraction(100), Fraction(100), Fraction(1), Fraction(1, 20), Fraction(1, 5)
        )
        self.assertEqual(rational_result, black_scholes_greeks(**BASE_PARAMETERS))

    def test_delta_matches_central_price_difference(self):
        for parameters, option_type in product(PARAMETER_GRID, OPTION_TYPES):
            spot, strike, maturity, rate, volatility = parameters
            step = spot * 1e-4
            price_up = black_scholes_price(
                spot + step, strike, maturity, rate, volatility, option_type
            )
            price_down = black_scholes_price(
                spot - step, strike, maturity, rate, volatility, option_type
            )
            finite_difference = (price_up - price_down) / (2 * step)
            result = black_scholes_greeks(*parameters, option_type)
            with self.subTest(parameters=parameters, option_type=option_type):
                self.assertAlmostEqual(
                    result.delta,
                    finite_difference,
                    delta=max(3e-8, 3e-6 * abs(finite_difference)),
                )

    def test_gamma_matches_second_central_price_difference(self):
        # A relative spot step balances second-difference cancellation and
        # truncation error across the price scales in this grid.
        for parameters, option_type in product(PARAMETER_GRID, OPTION_TYPES):
            spot, strike, maturity, rate, volatility = parameters
            step = spot * 1e-4
            price_up = black_scholes_price(
                spot + step, strike, maturity, rate, volatility, option_type
            )
            price_at_spot = black_scholes_price(*parameters, option_type)
            price_down = black_scholes_price(
                spot - step, strike, maturity, rate, volatility, option_type
            )
            finite_difference = (price_up - 2 * price_at_spot + price_down) / step**2
            result = black_scholes_greeks(*parameters, option_type)
            with self.subTest(parameters=parameters, option_type=option_type):
                self.assertAlmostEqual(
                    result.gamma,
                    finite_difference,
                    delta=max(2e-8, 1e-5 * abs(finite_difference)),
                )

    def test_vega_matches_central_price_difference(self):
        for parameters, option_type in product(PARAMETER_GRID, OPTION_TYPES):
            spot, strike, maturity, rate, volatility = parameters
            step = volatility * 1e-4
            price_up = black_scholes_price(
                spot, strike, maturity, rate, volatility + step, option_type
            )
            price_down = black_scholes_price(
                spot, strike, maturity, rate, volatility - step, option_type
            )
            finite_difference = (price_up - price_down) / (2 * step)
            result = black_scholes_greeks(*parameters, option_type)
            with self.subTest(parameters=parameters, option_type=option_type):
                self.assertAlmostEqual(
                    result.vega,
                    finite_difference,
                    delta=max(2e-7, 3e-6 * abs(finite_difference)),
                )

    def test_call_put_relationships(self):
        for parameters in PARAMETER_GRID:
            with self.subTest(parameters=parameters):
                call = black_scholes_greeks(*parameters, "call")
                put = black_scholes_greeks(*parameters, "put")
                self.assertAlmostEqual(call.delta - put.delta, 1.0, delta=1e-14)
                self.assertAlmostEqual(call.gamma, put.gamma, delta=1e-14)
                self.assertAlmostEqual(call.vega, put.vega, delta=1e-12)

    def test_delta_bounds_and_positive_gamma_and_vega(self):
        for parameters in PARAMETER_GRID:
            for option_type, lower_delta, upper_delta in (
                ("call", 0.0, 1.0),
                ("put", -1.0, 0.0),
            ):
                with self.subTest(parameters=parameters, option_type=option_type):
                    result = black_scholes_greeks(*parameters, option_type)
                    self.assertGreaterEqual(result.delta, lower_delta)
                    self.assertLessEqual(result.delta, upper_delta)
                    self.assertGreater(result.gamma, 0.0)
                    self.assertGreater(result.vega, 0.0)

    def test_scaling_in_spot_and_strike(self):
        for parameters in (
            (100, 100, 1, 0.05, 0.2),
            (80, 120, 0.5, -0.03, 0.35),
            (140, 100, 2, -0.08, 0.15),
        ):
            spot, strike, maturity, rate, volatility = parameters
            for scale, option_type in product((0.1, 0.5, 2, 10), OPTION_TYPES):
                original = black_scholes_greeks(*parameters, option_type)
                scaled = black_scholes_greeks(
                    scale * spot, scale * strike, maturity, rate, volatility, option_type
                )
                with self.subTest(parameters=parameters, scale=scale, option_type=option_type):
                    self.assertAlmostEqual(scaled.delta, original.delta, delta=1e-13)
                    self.assertAlmostEqual(scaled.gamma, original.gamma / scale, delta=1e-12)
                    self.assertAlmostEqual(scaled.vega, original.vega * scale, delta=1e-10)

    def test_rejects_nonreal_numeric_inputs(self):
        for name, value in product(BASE_PARAMETERS, INVALID_NUMERIC_TYPES):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                black_scholes_greeks(**parameters)

    def test_rejects_nonfinite_inputs(self):
        for name, value in product(BASE_PARAMETERS, (math.nan, math.inf, -math.inf)):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                black_scholes_greeks(**parameters)

    def test_rejects_invalid_positive_domains(self):
        for name, value in (
            ("spot", 0),
            ("spot", -1),
            ("strike", 0),
            ("strike", -1),
            ("time_to_maturity", -0.01),
            ("volatility", -0.01),
        ):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                black_scholes_greeks(**parameters)

    def test_rejects_zero_time_to_maturity(self):
        for spot, option_type in product((80, 100, 120), OPTION_TYPES):
            parameters = {**BASE_PARAMETERS, "spot": spot, "time_to_maturity": 0}
            with self.subTest(spot=spot, option_type=option_type):
                with self.assertRaises(ValueError):
                    black_scholes_greeks(**parameters, option_type=option_type)

    def test_rejects_zero_volatility(self):
        for spot, option_type in product((80, 100, 120), OPTION_TYPES):
            parameters = {**BASE_PARAMETERS, "spot": spot, "volatility": 0}
            with self.subTest(spot=spot, option_type=option_type):
                with self.assertRaises(ValueError):
                    black_scholes_greeks(**parameters, option_type=option_type)

    def test_rejects_invalid_option_types(self):
        for option_type in ("", "CALL", "Put", " call", "put ", "straddle"):
            with self.subTest(option_type=option_type), self.assertRaises(ValueError):
                black_scholes_greeks(**BASE_PARAMETERS, option_type=option_type)
        for option_type in (None, True, 0, [], {}):
            with self.subTest(option_type=option_type), self.assertRaises(TypeError):
                black_scholes_greeks(**BASE_PARAMETERS, option_type=option_type)


if __name__ == "__main__":
    unittest.main()
