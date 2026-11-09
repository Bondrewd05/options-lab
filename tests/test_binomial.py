"""Independent references and domain checks for European CRR prices."""

import math
import unittest
from decimal import Decimal
from fractions import Fraction
from itertools import product

from options_lab import binomial_crr_price, black_scholes_price, european_payoff


BASE_PARAMETERS = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
OPTION_TYPES = ("call", "put")
PARAMETER_GRID = tuple(
    product((80.0, 100.0, 120.0), (0.25, 2.0), (-0.05, 0.0, 0.07), (0.2, 0.5))
)


class BinomialCrrPriceTests(unittest.TestCase):
    def test_defaults_are_one_hundred_steps_and_call(self):
        self.assertEqual(
            binomial_crr_price(**BASE_PARAMETERS),
            binomial_crr_price(**BASE_PARAMETERS, steps=100, option_type="call"),
        )

    def test_one_step_matches_replicating_portfolio(self):
        # Solve the two terminal portfolio equations directly. This reference
        # does not use risk-neutral probabilities or backward induction.
        for spot, strike, maturity, rate, volatility in (
            (100, 100, 1, 0.05, 0.2),
            (90, 100, 0.5, -0.03, 0.4),
            (120, 100, 2, -0.05, 0.25),
        ):
            terminal_up = spot * math.exp(volatility * math.sqrt(maturity))
            terminal_down = spot * math.exp(-volatility * math.sqrt(maturity))
            for option_type in OPTION_TYPES:
                payoff_up = european_payoff(terminal_up, strike, option_type)
                payoff_down = european_payoff(terminal_down, strike, option_type)
                shares = (payoff_up - payoff_down) / (terminal_up - terminal_down)
                cash = math.exp(-rate * maturity) * (payoff_up - shares * terminal_up)
                expected = shares * spot + cash
                with self.subTest(spot=spot, rate=rate, option_type=option_type):
                    actual = binomial_crr_price(
                        spot, strike, maturity, rate, volatility,
                        steps=1, option_type=option_type,
                    )
                    self.assertAlmostEqual(actual, expected, delta=1e-11)

    def test_small_trees_match_direct_terminal_distribution(self):
        # Sum the binomial terminal distribution instead of recursing over nodes.
        for steps, rate, option_type in product((2, 3, 5, 8), (-0.04, 0.06), OPTION_TYPES):
            spot, strike, maturity, volatility = 95.0, 105.0, 1.5, 0.3
            step_time = maturity / steps
            up = math.exp(volatility * math.sqrt(step_time))
            down = 1 / up
            probability_up = (math.exp(rate * step_time) - down) / (up - down)
            expected_payoffs = []
            for up_moves in range(steps + 1):
                terminal_spot = spot * up**up_moves * down ** (steps - up_moves)
                probability = (
                    math.comb(steps, up_moves)
                    * probability_up**up_moves
                    * (1 - probability_up) ** (steps - up_moves)
                )
                expected_payoffs.append(
                    probability * european_payoff(terminal_spot, strike, option_type)
                )
            expected = math.exp(-rate * maturity) * math.fsum(expected_payoffs)
            with self.subTest(steps=steps, rate=rate, option_type=option_type):
                actual = binomial_crr_price(
                    spot, strike, maturity, rate, volatility,
                    steps=steps, option_type=option_type,
                )
                self.assertAlmostEqual(actual, expected, delta=1e-11)

    def test_expiry_equals_payoff(self):
        for spot, rate, volatility, steps, option_type in product(
            (80, 100, 120), (-1, 0, 1), (0, 0.2), (1, 3), OPTION_TYPES
        ):
            with self.subTest(
                spot=spot, rate=rate, volatility=volatility,
                steps=steps, option_type=option_type,
            ):
                self.assertEqual(
                    binomial_crr_price(spot, 100, 0, rate, volatility, steps, option_type),
                    european_payoff(spot, 100, option_type),
                )

    def test_zero_volatility_matches_discounted_deterministic_payoff(self):
        for spot, maturity, rate, steps in product(
            (60, 100, 160), (0.5, 3), (-0.25, 0, 0.25), (1, 7)
        ):
            discounted_strike = 100 * math.exp(-rate * maturity)
            for option_type, expected in (
                ("call", max(spot - discounted_strike, 0.0)),
                ("put", max(discounted_strike - spot, 0.0)),
            ):
                with self.subTest(
                    spot=spot, maturity=maturity, rate=rate,
                    steps=steps, option_type=option_type,
                ):
                    self.assertAlmostEqual(
                        binomial_crr_price(spot, 100, maturity, rate, 0, steps, option_type),
                        expected,
                        delta=1e-11,
                    )

    def test_put_call_parity(self):
        for spot, maturity, rate, volatility in PARAMETER_GRID:
            parameters = (spot, 100, maturity, rate, volatility)
            with self.subTest(parameters=parameters):
                call = binomial_crr_price(*parameters, steps=25, option_type="call")
                put = binomial_crr_price(*parameters, steps=25, option_type="put")
                self.assertAlmostEqual(
                    call - put, spot - 100 * math.exp(-rate * maturity), delta=1e-10
                )

    def test_no_arbitrage_price_bounds(self):
        for spot, maturity, rate, volatility in PARAMETER_GRID:
            parameters = (spot, 100, maturity, rate, volatility)
            discounted_strike = 100 * math.exp(-rate * maturity)
            for option_type, lower_bound, upper_bound in (
                ("call", max(spot - discounted_strike, 0.0), spot),
                ("put", max(discounted_strike - spot, 0.0), discounted_strike),
            ):
                with self.subTest(parameters=parameters, option_type=option_type):
                    price = binomial_crr_price(*parameters, steps=25, option_type=option_type)
                    self.assertTrue(math.isfinite(price))
                    self.assertGreaterEqual(price, lower_bound - 1e-10)
                    self.assertLessEqual(price, upper_bound + 1e-10)

    def test_even_and_odd_refinements_approach_black_scholes(self):
        # CRR prices may oscillate with the step count. Check both parities
        # against the continuous-model price without assuming monotone errors.
        for parameters in (
            (100, 100, 1, 0.05, 0.2),
            (80, 100, 0.5, -0.03, 0.35),
            (140, 100, 2, -0.08, 0.15),
            (100, 120, 0.25, 0.01, 0.6),
            (50, 50, 0.25, -0.02, 0.4),
        ):
            for steps, option_type in product((400, 401), OPTION_TYPES):
                with self.subTest(parameters=parameters, steps=steps, option_type=option_type):
                    actual = binomial_crr_price(*parameters, steps=steps, option_type=option_type)
                    expected = black_scholes_price(*parameters, option_type)
                    self.assertAlmostEqual(actual, expected, delta=0.02)

    def test_accepts_real_financial_inputs(self):
        self.assertEqual(
            binomial_crr_price(100, 100, 1, 0, 1, steps=5),
            binomial_crr_price(100.0, 100.0, 1.0, 0.0, 1.0, steps=5),
        )
        rational_result = binomial_crr_price(
            Fraction(100), Fraction(100), Fraction(1), Fraction(1, 20), Fraction(1, 5),
            steps=5,
        )
        self.assertEqual(rational_result, binomial_crr_price(**BASE_PARAMETERS, steps=5))

    def test_rejects_nonreal_financial_inputs_before_shortcuts(self):
        invalid_types = (True, False, None, "100", 1j, [], {}, Decimal("1"))
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            for name, value in product(BASE_PARAMETERS, invalid_types):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(TypeError):
                        binomial_crr_price(**parameters, steps=1)

    def test_rejects_nonfinite_financial_inputs_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            for name, value in product(BASE_PARAMETERS, (math.nan, math.inf, -math.inf)):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(ValueError):
                        binomial_crr_price(**parameters, steps=1)

    def test_rejects_invalid_financial_domains_before_shortcuts(self):
        invalid_domains = (
            ("spot", 0), ("spot", -1), ("strike", 0), ("strike", -1),
            ("time_to_maturity", -0.1), ("volatility", -0.1),
        )
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            for name, value in invalid_domains:
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(ValueError):
                        binomial_crr_price(**parameters, steps=1)

    def test_rejects_nonintegral_steps_before_shortcuts(self):
        invalid_steps = (
            True, False, None, "10", 10.0, 1.5, Fraction(10), Decimal("10"),
            1j, [], {}, math.nan, math.inf,
        )
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            parameters = {**BASE_PARAMETERS, **shortcut}
            for steps in invalid_steps:
                with self.subTest(shortcut=shortcut, steps=steps):
                    with self.assertRaises(TypeError):
                        binomial_crr_price(**parameters, steps=steps)

    def test_rejects_nonpositive_steps_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            parameters = {**BASE_PARAMETERS, **shortcut}
            for steps in (0, -1, -100):
                with self.subTest(shortcut=shortcut, steps=steps):
                    with self.assertRaises(ValueError):
                        binomial_crr_price(**parameters, steps=steps)

    def test_rejects_arbitrage_and_strict_probability_boundaries(self):
        # With one year and one step, the strict condition is -sigma < r < sigma.
        for rate, option_type in product((-0.3, -0.2, 0.2, 0.3), OPTION_TYPES):
            parameters = {**BASE_PARAMETERS, "rate": rate}
            with self.subTest(rate=rate, option_type=option_type):
                with self.assertRaisesRegex(ValueError, "steps"):
                    binomial_crr_price(**parameters, steps=1, option_type=option_type)

    def test_refinement_can_restore_no_arbitrage(self):
        for rate, option_type in product((-0.3, 0.3), OPTION_TYPES):
            parameters = {**BASE_PARAMETERS, "rate": rate}
            with self.subTest(rate=rate, option_type=option_type):
                with self.assertRaises(ValueError):
                    binomial_crr_price(**parameters, steps=1, option_type=option_type)
                price = binomial_crr_price(**parameters, steps=4, option_type=option_type)
                self.assertTrue(math.isfinite(price))
                self.assertGreaterEqual(price, 0.0)

    def test_rejects_invalid_option_types_before_shortcuts(self):
        for shortcut in ({}, {"time_to_maturity": 0}, {"volatility": 0}):
            parameters = {**BASE_PARAMETERS, **shortcut}
            for option_type in ("", "CALL", "Put", " call", "put ", "straddle"):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(ValueError):
                        binomial_crr_price(**parameters, steps=1, option_type=option_type)
            for option_type in (None, True, 0, [], {}):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(TypeError):
                        binomial_crr_price(**parameters, steps=1, option_type=option_type)

    def test_reports_unrepresentable_numeric_tree(self):
        for parameters in (
            {**BASE_PARAMETERS, "volatility": 1000.0},
            {
                "spot": 1.0,
                "strike": 1.0,
                "time_to_maturity": 1.0,
                "rate": -600.0,
                "volatility": 700.0,
            },
        ):
            with self.subTest(parameters=parameters), self.assertRaises(OverflowError):
                binomial_crr_price(**parameters, steps=1)


if __name__ == "__main__":
    unittest.main()
