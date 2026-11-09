"""Independent sample statistics and contract checks for Monte Carlo pricing."""

import math
import random
import statistics
import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import product

from options_lab import MonteCarloEstimate, black_scholes_price, monte_carlo_price


BASE_PARAMETERS = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
OPTION_TYPES = ("call", "put")
INVALID_NUMERIC_TYPES = (True, False, None, "100", 1j, [], {}, Decimal("1"))
BOUNDARY_PARAMETERS = ({}, {"time_to_maturity": 0}, {"volatility": 0})


class MonteCarloPriceTests(unittest.TestCase):
    def test_public_api_returns_frozen_estimate_dataclass(self):
        result = monte_carlo_price(**BASE_PARAMETERS, n_paths=8, seed=123)
        self.assertTrue(is_dataclass(MonteCarloEstimate))
        self.assertIsInstance(result, MonteCarloEstimate)
        self.assertEqual(
            {field.name for field in fields(result)},
            {
                "price", "standard_error", "confidence_interval",
                "n_paths", "seed", "confidence_level",
            },
        )
        self.assertIsInstance(result.price, float)
        self.assertIsInstance(result.standard_error, float)
        self.assertIsInstance(result.confidence_interval, tuple)
        self.assertEqual(len(result.confidence_interval), 2)
        self.assertEqual(result.n_paths, 8)
        self.assertEqual(result.seed, 123)
        self.assertEqual(result.confidence_level, 0.95)
        for value in (result.price, result.standard_error, *result.confidence_interval):
            self.assertTrue(math.isfinite(value))
        with self.assertRaises(FrozenInstanceError):
            result.price = 0.0

    def test_matches_independent_terminal_payoff_statistics(self):
        # Store terminal payoffs and use statistics.stdev, independently of the
        # online accumulator. n_paths=2 also detects the population-variance divisor.
        for parameters in (
            (100, 100, 1, 0.05, 0.2),
            (80, 100, 0.5, -0.03, 0.35),
            (140, 100, 2, -0.08, 0.15),
        ):
            spot, strike, maturity, rate, volatility = parameters
            for n_paths, seed, option_type in product((2, 7, 64), (0, -17), OPTION_TYPES):
                generator = random.Random(seed)
                discounted_payoffs = []
                for _ in range(n_paths):
                    terminal_spot = spot * math.exp(
                        (rate - volatility**2 / 2) * maturity
                        + volatility * math.sqrt(maturity) * generator.gauss(0, 1)
                    )
                    if option_type == "call":
                        payoff = max(terminal_spot - strike, 0.0)
                    else:
                        payoff = max(strike - terminal_spot, 0.0)
                    discounted_payoffs.append(math.exp(-rate * maturity) * payoff)
                expected_price = statistics.mean(discounted_payoffs)
                expected_error = statistics.stdev(discounted_payoffs) / math.sqrt(n_paths)
                with self.subTest(
                    parameters=parameters, n_paths=n_paths, seed=seed, option_type=option_type,
                ):
                    result = monte_carlo_price(
                        *parameters, n_paths=n_paths, option_type=option_type, seed=seed
                    )
                    self.assertAlmostEqual(result.price, expected_price, delta=1e-11)
                    self.assertAlmostEqual(result.standard_error, expected_error, delta=1e-11)

    def test_defaults_are_call_one_hundred_thousand_paths_and_no_seed(self):
        # Expiry makes the default path count inexpensive to check.
        result = monte_carlo_price(120, 100, 0, 0.05, 0.2)
        self.assertEqual(result.price, 20.0)
        self.assertEqual(result.n_paths, 100_000)
        self.assertIsNone(result.seed)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(
            monte_carlo_price(**BASE_PARAMETERS, n_paths=20, seed=7),
            monte_carlo_price(**BASE_PARAMETERS, n_paths=20, option_type="call", seed=7),
        )

    def test_fixed_seed_reproduces_the_estimate(self):
        for seed, option_type in product((0, 123, -17), OPTION_TYPES):
            with self.subTest(seed=seed, option_type=option_type):
                first = monte_carlo_price(
                    **BASE_PARAMETERS, n_paths=128, option_type=option_type, seed=seed
                )
                second = monte_carlo_price(
                    **BASE_PARAMETERS, n_paths=128, option_type=option_type, seed=seed
                )
                self.assertEqual(first, second)

    def test_simulation_preserves_global_random_state(self):
        for seed in (123, None):
            with self.subTest(seed=seed):
                state_before = random.getstate()
                try:
                    monte_carlo_price(**BASE_PARAMETERS, n_paths=32, seed=seed)
                    self.assertEqual(random.getstate(), state_before)
                finally:
                    random.setstate(state_before)

    def test_distinct_seeds_produce_distinct_samples(self):
        first = monte_carlo_price(**BASE_PARAMETERS, n_paths=64, seed=11)
        second = monte_carlo_price(**BASE_PARAMETERS, n_paths=64, seed=29)
        self.assertNotEqual(first.price, second.price)
        self.assertNotEqual(first.standard_error, second.standard_error)

    def test_confidence_interval_uses_normal_quantile_without_clipping(self):
        critical_value = statistics.NormalDist().inv_cdf(0.975)
        for option_type in OPTION_TYPES:
            with self.subTest(option_type=option_type):
                result = monte_carlo_price(
                    **BASE_PARAMETERS, n_paths=2, option_type=option_type, seed=0
                )
                lower, upper = result.confidence_interval
                self.assertAlmostEqual(
                    lower, result.price - critical_value * result.standard_error, delta=1e-12
                )
                self.assertAlmostEqual(
                    upper, result.price + critical_value * result.standard_error, delta=1e-12
                )
                self.assertLess(lower, 0.0)

    def test_price_and_uncertainty_scale_with_currency_units(self):
        for scale, option_type in product((0.1, 2, 10), OPTION_TYPES):
            original = monte_carlo_price(
                **BASE_PARAMETERS, n_paths=128, option_type=option_type, seed=42
            )
            scaled_parameters = {
                **BASE_PARAMETERS,
                "spot": scale * BASE_PARAMETERS["spot"],
                "strike": scale * BASE_PARAMETERS["strike"],
            }
            scaled = monte_carlo_price(
                **scaled_parameters, n_paths=128, option_type=option_type, seed=42
            )
            with self.subTest(scale=scale, option_type=option_type):
                for original_value, scaled_value in zip(
                    (original.price, original.standard_error, *original.confidence_interval),
                    (scaled.price, scaled.standard_error, *scaled.confidence_interval),
                ):
                    expected = scale * original_value
                    self.assertAlmostEqual(
                        scaled_value, expected, delta=1e-11 * max(1.0, abs(expected))
                    )

    def test_monte_carlo_estimates_agree_with_black_scholes_at_sampling_scale(self):
        # A fixed five-standard-error allowance checks pricing and uncertainty
        # without requiring every nominal 95% interval to contain the true price.
        for parameters in (
            (100, 100, 1, 0.05, 0.2),
            (80, 100, 0.5, -0.03, 0.35),
        ):
            for seed, option_type in product((42, 2026), OPTION_TYPES):
                with self.subTest(parameters=parameters, seed=seed, option_type=option_type):
                    result = monte_carlo_price(
                        *parameters, n_paths=20_000, option_type=option_type, seed=seed
                    )
                    expected = black_scholes_price(*parameters, option_type)
                    self.assertGreater(result.standard_error, 0.0)
                    self.assertLessEqual(abs(result.price - expected), 5 * result.standard_error)

    def test_expiry_has_exact_price_and_zero_sampling_error(self):
        for spot, volatility, option_type in product((80, 100, 120), (0, 0.2), OPTION_TYPES):
            expected = black_scholes_price(spot, 100, 0, -0.1, volatility, option_type)
            with self.subTest(spot=spot, volatility=volatility, option_type=option_type):
                result = monte_carlo_price(
                    spot, 100, 0, -0.1, volatility, n_paths=2, option_type=option_type, seed=3
                )
                self.assertEqual(result.price, expected)
                self.assertEqual(result.standard_error, 0.0)
                self.assertEqual(result.confidence_interval, (expected, expected))
                self.assertEqual(result.n_paths, 2)
                self.assertEqual(result.seed, 3)

    def test_zero_volatility_has_exact_price_and_zero_sampling_error(self):
        for spot, rate, option_type in product((60, 100, 160), (-0.1, 0, 0.1), OPTION_TYPES):
            expected = black_scholes_price(spot, 100, 2, rate, 0, option_type)
            with self.subTest(spot=spot, rate=rate, option_type=option_type):
                result = monte_carlo_price(
                    spot, 100, 2, rate, 0, n_paths=2, option_type=option_type, seed=None
                )
                self.assertEqual(result.price, expected)
                self.assertEqual(result.standard_error, 0.0)
                self.assertEqual(result.confidence_interval, (expected, expected))

    def test_all_zero_payoff_sample_reports_zero_empirical_error(self):
        # This small sample misses a positive-value option's rare exercise event.
        result = monte_carlo_price(100, 200, 1, 0.05, 0.2, n_paths=2, seed=0)
        self.assertEqual(result.price, 0.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual(result.confidence_interval, (0.0, 0.0))
        self.assertGreater(black_scholes_price(100, 200, 1, 0.05, 0.2), 0.0)

    def test_accepts_real_financial_inputs(self):
        self.assertEqual(
            monte_carlo_price(100, 100, 1, 0, 1, n_paths=8, seed=7),
            monte_carlo_price(100.0, 100.0, 1.0, 0.0, 1.0, n_paths=8, seed=7),
        )
        rational_result = monte_carlo_price(
            Fraction(100), Fraction(100), Fraction(1), Fraction(1, 20), Fraction(1, 5),
            n_paths=8, seed=7,
        )
        self.assertEqual(
            rational_result, monte_carlo_price(**BASE_PARAMETERS, n_paths=8, seed=7)
        )

    def test_rejects_nonreal_financial_inputs_before_shortcuts(self):
        for shortcut in BOUNDARY_PARAMETERS:
            for name, value in product(BASE_PARAMETERS, INVALID_NUMERIC_TYPES):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(TypeError):
                        monte_carlo_price(**parameters, n_paths=2, seed=0)

    def test_rejects_nonfinite_financial_inputs_before_shortcuts(self):
        for shortcut in BOUNDARY_PARAMETERS:
            for name, value in product(BASE_PARAMETERS, (math.nan, math.inf, -math.inf)):
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(ValueError):
                        monte_carlo_price(**parameters, n_paths=2, seed=0)

    def test_rejects_invalid_financial_domains_before_shortcuts(self):
        invalid_domains = (
            ("spot", 0), ("spot", -1), ("strike", 0), ("strike", -1),
            ("time_to_maturity", -0.1), ("volatility", -0.1),
        )
        for shortcut in BOUNDARY_PARAMETERS:
            for name, value in invalid_domains:
                parameters = {**BASE_PARAMETERS, **shortcut, name: value}
                with self.subTest(shortcut=shortcut, name=name, value=value):
                    with self.assertRaises(ValueError):
                        monte_carlo_price(**parameters, n_paths=2, seed=0)

    def test_rejects_invalid_path_counts_before_shortcuts(self):
        invalid_types = (
            True, False, None, "10", 10.0, Fraction(10), Decimal("10"),
            1j, [], {}, math.nan, math.inf,
        )
        for shortcut in BOUNDARY_PARAMETERS:
            parameters = {**BASE_PARAMETERS, **shortcut}
            for n_paths in invalid_types:
                with self.subTest(shortcut=shortcut, n_paths=n_paths):
                    with self.assertRaises(TypeError):
                        monte_carlo_price(**parameters, n_paths=n_paths, seed=0)
            for n_paths in (-10, 0, 1):
                with self.subTest(shortcut=shortcut, n_paths=n_paths):
                    with self.assertRaises(ValueError):
                        monte_carlo_price(**parameters, n_paths=n_paths, seed=0)

    def test_rejects_invalid_seeds_before_shortcuts(self):
        invalid_seeds = (
            True, False, "7", 7.0, Fraction(7), Decimal("7"), 1j, [], {},
            math.nan, math.inf,
        )
        for shortcut in BOUNDARY_PARAMETERS:
            parameters = {**BASE_PARAMETERS, **shortcut}
            for seed in invalid_seeds:
                with self.subTest(shortcut=shortcut, seed=seed):
                    with self.assertRaises(TypeError):
                        monte_carlo_price(**parameters, n_paths=2, seed=seed)

    def test_rejects_invalid_option_types_before_shortcuts(self):
        for shortcut in BOUNDARY_PARAMETERS:
            parameters = {**BASE_PARAMETERS, **shortcut}
            for option_type in ("", "CALL", "Put", " call", "put ", "straddle"):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(ValueError):
                        monte_carlo_price(**parameters, n_paths=2, option_type=option_type, seed=0)
            for option_type in (None, True, 0, [], {}):
                with self.subTest(shortcut=shortcut, option_type=option_type):
                    with self.assertRaises(TypeError):
                        monte_carlo_price(**parameters, n_paths=2, option_type=option_type, seed=0)


if __name__ == "__main__":
    unittest.main()
