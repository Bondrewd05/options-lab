"""Contract, reference-value, and financial-property tests for pricing."""

import math
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product

from options_lab.pricing import black_scholes_price, european_payoff


BASE_PARAMETERS = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
INVALID_NUMERIC_TYPES = (True, False, None, "100", 1j, [], {}, Decimal("1"))
NONFINITE_VALUES = (math.nan, math.inf, -math.inf)
OPTION_TYPES = ("call", "put")
PARAMETER_GRID = tuple(
    product(
        (0.1, 50.0, 100.0, 250.0),
        (0.1, 50.0, 100.0, 250.0),
        (0.0, 0.0001, 0.1, 1.0, 10.0),
        (-0.1, 0.0, 0.05),
        (0.0, 0.01, 0.2, 1.5),
    )
)


class EuropeanPayoffTests(unittest.TestCase):
    def test_payoff_examples_and_default_call(self):
        for terminal_spot, expected_call, expected_put in (
            (0, 0, 100),
            (80, 0, 20),
            (100, 0, 0),
            (120, 20, 0),
        ):
            with self.subTest(terminal_spot=terminal_spot):
                self.assertEqual(european_payoff(terminal_spot, 100), expected_call)
                self.assertEqual(
                    european_payoff(terminal_spot, 100, "call"), expected_call
                )
                self.assertEqual(
                    european_payoff(terminal_spot, 100, "put"), expected_put
                )

    def test_accepts_real_scalars(self):
        for terminal_spot, strike in (
            (120, 100),
            (120.0, 100.0),
            (Fraction(120), Fraction(100)),
            (Fraction(241, 2), Fraction(201, 2)),
        ):
            with self.subTest(terminal_spot=terminal_spot, strike=strike):
                self.assertEqual(european_payoff(terminal_spot, strike), 20.0)

    def test_rejects_nonreal_numeric_inputs(self):
        for name, value in product(("terminal_spot", "strike"), INVALID_NUMERIC_TYPES):
            parameters = {"terminal_spot": 100.0, "strike": 100.0}
            parameters[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                european_payoff(**parameters)

    def test_rejects_nonfinite_inputs(self):
        for name, value in product(("terminal_spot", "strike"), NONFINITE_VALUES):
            parameters = {"terminal_spot": 100.0, "strike": 100.0}
            parameters[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                european_payoff(**parameters)

    def test_rejects_invalid_domains(self):
        for terminal_spot, strike in ((-1.0, 100.0), (100.0, 0.0), (100.0, -1.0)):
            with self.subTest(terminal_spot=terminal_spot, strike=strike):
                with self.assertRaises(ValueError):
                    european_payoff(terminal_spot, strike)

    def test_rejects_invalid_option_types(self):
        for option_type in ("", "CALL", "Put", " call", "put ", "straddle"):
            with self.subTest(option_type=option_type), self.assertRaises(ValueError):
                european_payoff(100, 100, option_type)
        for option_type in (None, True, 0, [], {}):
            with self.subTest(option_type=option_type), self.assertRaises(TypeError):
                european_payoff(100, 100, option_type)


class BlackScholesPriceTests(unittest.TestCase):
    def test_reference_prices(self):
        # The last three pairs were computed by independent numerical integration
        # of the discounted lognormal payoff with scipy.integrate.quad.
        reference_cases = (
            ((100, 100, 1, 0.05, 0.2), 10.450583572185565, 5.573526022256971),
            ((80, 100, 0.5, -0.03, 0.35), 1.9860868615027, 23.4973933230747),
            ((140, 100, 2, -0.08, 0.15), 25.7272478973012, 3.07833499648231),
            ((100, 120, 0.25, 0.01, 0.6), 5.50814494817265, 25.2085196358679),
        )
        for parameters, expected_call, expected_put in reference_cases:
            for option_type, expected in (("call", expected_call), ("put", expected_put)):
                with self.subTest(parameters=parameters, option_type=option_type):
                    self.assertAlmostEqual(
                        black_scholes_price(*parameters, option_type), expected, delta=1e-10
                    )

    def test_default_call_and_real_scalars(self):
        self.assertEqual(
            black_scholes_price(**BASE_PARAMETERS),
            black_scholes_price(**BASE_PARAMETERS, option_type="call"),
        )
        self.assertEqual(black_scholes_price(120, 100, 1, 0, 0), 20.0)
        rational_parameters = (
            Fraction(100), Fraction(100), Fraction(1), Fraction(1, 20), Fraction(1, 5)
        )
        self.assertAlmostEqual(
            black_scholes_price(*rational_parameters), 10.450583572185565, delta=1e-10
        )

    def test_expiry_equals_payoff(self):
        for spot, rate, volatility, option_type in product(
            (80, 100, 120), (-0.1, 0, 0.1), (0, 0.2, 1.0), OPTION_TYPES
        ):
            with self.subTest(spot=spot, rate=rate, volatility=volatility, option_type=option_type):
                self.assertEqual(
                    black_scholes_price(spot, 100, 0, rate, volatility, option_type),
                    european_payoff(spot, 100, option_type),
                )

    def test_zero_volatility_equals_discounted_deterministic_payoff(self):
        for spot, maturity, rate in product((60, 100, 160), (0.1, 1, 5), (-0.1, 0, 0.1)):
            discounted_strike = 100 * math.exp(-rate * maturity)
            for option_type, expected in (
                ("call", max(spot - discounted_strike, 0.0)),
                ("put", max(discounted_strike - spot, 0.0)),
            ):
                with self.subTest(spot=spot, maturity=maturity, rate=rate, option_type=option_type):
                    self.assertAlmostEqual(
                        black_scholes_price(spot, 100, maturity, rate, 0, option_type),
                        expected,
                        delta=1e-10,
                    )

    def test_discounted_strike_survives_extreme_discount_factors(self):
        # Decimal forms K*exp(-r) without an overflowing/underflowing float
        # factor. The middle case also covers a nonzero subnormal discount.
        for spot, strike, rate in (
            (1e-26, 1e300, 750.0),
            (1e-23, 1e300, 744.0),
            (1e25, 1e-300, -750.0),
        ):
            with localcontext() as context:
                context.prec = 80
                discounted_strike = Decimal.from_float(strike) * (-Decimal.from_float(rate)).exp()
                difference = Decimal.from_float(spot) - discounted_strike
                expected = {
                    "call": float(max(difference, Decimal(0))),
                    "put": float(max(-difference, Decimal(0))),
                }
            for option_type in OPTION_TYPES:
                with self.subTest(spot=spot, strike=strike, rate=rate, option_type=option_type):
                    actual = black_scholes_price(spot, strike, 1, rate, 0, option_type)
                    self.assertTrue(math.isclose(actual, expected[option_type], rel_tol=2e-12, abs_tol=0.0))

    def test_put_call_parity_on_parameter_grid(self):
        for parameters in PARAMETER_GRID:
            spot, strike, maturity, rate, _ = parameters
            discounted_strike = strike * math.exp(-rate * maturity)
            with self.subTest(parameters=parameters):
                call = black_scholes_price(*parameters, "call")
                put = black_scholes_price(*parameters, "put")
                self.assertAlmostEqual(
                    call - put,
                    spot - discounted_strike,
                    delta=1e-11 * (1 + spot + discounted_strike),
                )

    def test_no_arbitrage_bounds_on_parameter_grid(self):
        for parameters in PARAMETER_GRID:
            spot, strike, maturity, rate, _ = parameters
            discounted_strike = strike * math.exp(-rate * maturity)
            tolerance = 1e-11 * (1 + spot + discounted_strike)
            for option_type, lower_bound, upper_bound in (
                ("call", max(spot - discounted_strike, 0.0), spot),
                ("put", max(discounted_strike - spot, 0.0), discounted_strike),
            ):
                with self.subTest(parameters=parameters, option_type=option_type):
                    price = black_scholes_price(*parameters, option_type)
                    self.assertTrue(math.isfinite(price))
                    self.assertGreaterEqual(price, lower_bound - tolerance)
                    self.assertLessEqual(price, upper_bound + tolerance)

    def test_price_monotonicity_in_spot(self):
        for maturity, rate, volatility in product((0, 0.1, 1, 5), (-0.05, 0.05), (0, 0.2, 1)):
            calls = [
                black_scholes_price(spot, 100, maturity, rate, volatility, "call")
                for spot in (20, 80, 100, 120, 250)
            ]
            puts = [
                black_scholes_price(spot, 100, maturity, rate, volatility, "put")
                for spot in (20, 80, 100, 120, 250)
            ]
            with self.subTest(maturity=maturity, rate=rate, volatility=volatility):
                for lower, higher in zip(calls, calls[1:]):
                    self.assertLessEqual(lower, higher + 1e-10)
                for lower, higher in zip(puts, puts[1:]):
                    self.assertGreaterEqual(lower + 1e-10, higher)

    def test_price_monotonicity_in_strike(self):
        for maturity, rate, volatility in product((0, 0.1, 1, 5), (-0.05, 0.05), (0, 0.2, 1)):
            calls = [
                black_scholes_price(100, strike, maturity, rate, volatility, "call")
                for strike in (20, 80, 100, 120, 250)
            ]
            puts = [
                black_scholes_price(100, strike, maturity, rate, volatility, "put")
                for strike in (20, 80, 100, 120, 250)
            ]
            with self.subTest(maturity=maturity, rate=rate, volatility=volatility):
                for lower, higher in zip(calls, calls[1:]):
                    self.assertGreaterEqual(lower + 1e-10, higher)
                for lower, higher in zip(puts, puts[1:]):
                    self.assertLessEqual(lower, higher + 1e-10)

    def test_price_monotonicity_in_volatility(self):
        for spot, maturity, rate, option_type in product(
            (60, 100, 160), (0, 0.1, 1, 5), (-0.05, 0.05), OPTION_TYPES
        ):
            prices = [
                black_scholes_price(spot, 100, maturity, rate, volatility, option_type)
                for volatility in (0, 0.02, 0.1, 0.3, 0.7, 1.5)
            ]
            with self.subTest(spot=spot, maturity=maturity, rate=rate, option_type=option_type):
                for lower, higher in zip(prices, prices[1:]):
                    self.assertLessEqual(lower, higher + 1e-10)

    def test_positive_homogeneity_in_spot_and_strike(self):
        for spot, strike, maturity, rate, volatility in (
            (100, 100, 1, 0.05, 0.2),
            (80, 120, 0.5, -0.03, 0.35),
            (140, 100, 0, 0.05, 0.2),
            (140, 100, 2, -0.08, 0),
        ):
            for scale, option_type in product((0.01, 0.5, 2, 50), OPTION_TYPES):
                expected = scale * black_scholes_price(
                    spot, strike, maturity, rate, volatility, option_type
                )
                actual = black_scholes_price(
                    scale * spot, scale * strike, maturity, rate, volatility, option_type
                )
                with self.subTest(
                    spot=spot, strike=strike, maturity=maturity, rate=rate,
                    volatility=volatility, scale=scale, option_type=option_type
                ):
                    self.assertAlmostEqual(actual, expected, delta=1e-11 * (1 + abs(expected)))

    def test_rejects_nonreal_numeric_inputs_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            for name, value in product(BASE_PARAMETERS, INVALID_NUMERIC_TYPES):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(TypeError):
                        black_scholes_price(**parameters)

    def test_rejects_nonfinite_inputs_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            for name, value in product(BASE_PARAMETERS, NONFINITE_VALUES):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(ValueError):
                        black_scholes_price(**parameters)

    def test_rejects_invalid_domains_before_shortcuts(self):
        invalid_domains = (
            ("spot", 0),
            ("spot", -1),
            ("strike", 0),
            ("strike", -1),
            ("time_to_maturity", -0.01),
            ("volatility", -0.01),
        )
        for shortcut, (name, value) in product(
            ({}, {"time_to_maturity": 0}, {"volatility": 0}), invalid_domains
        ):
            parameters = {**BASE_PARAMETERS, **shortcut, name: value}
            with self.subTest(shortcut=shortcut, name=name, value=value):
                with self.assertRaises(ValueError):
                    black_scholes_price(**parameters)

    def test_rejects_invalid_option_types_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            parameters = {**BASE_PARAMETERS, **shortcut}
            for option_type in ("", "CALL", "Put", " call", "put ", "straddle"):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(ValueError):
                        black_scholes_price(**parameters, option_type=option_type)
            for option_type in (None, True, 0, [], {}):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(TypeError):
                        black_scholes_price(**parameters, option_type=option_type)


if __name__ == "__main__":
    unittest.main()
