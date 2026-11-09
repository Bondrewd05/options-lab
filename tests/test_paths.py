"""Exact-transition and input-contract tests for risk-neutral GBM paths."""

import math
import random
import unittest
from decimal import Decimal
from fractions import Fraction
from itertools import product

from options_lab import simulate_gbm_path


BASE_PARAMETERS = {
    "spot": 100.0,
    "time_to_maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
}
INVALID_NUMERIC_TYPES = (True, False, None, "100", 1j, [], {}, Decimal("1"))


class SimulateGbmPathTests(unittest.TestCase):
    def test_returns_positive_tuple_including_initial_spot_and_default_steps(self):
        path = simulate_gbm_path(**BASE_PARAMETERS, seed=7)
        self.assertIsInstance(path, tuple)
        self.assertEqual(len(path), 253)
        self.assertEqual(path[0], 100.0)
        for spot in path:
            self.assertTrue(math.isfinite(spot))
            self.assertGreater(spot, 0.0)
        unseeded = simulate_gbm_path(**BASE_PARAMETERS, steps=1)
        self.assertEqual(len(unseeded), 2)
        self.assertEqual(unseeded[0], 100.0)

    def test_matches_independent_cumulative_gaussian_solution(self):
        # Evaluate the solution from S0 and cumulative Gaussian increments,
        # independently of the implementation's successive price transitions.
        for steps, seed, rate in product((1, 5, 9), (0, -17), (-0.03, 0.05)):
            spot, maturity, volatility = 95.0, 1.5, 0.35
            generator = random.Random(seed)
            increments = [generator.gauss(0, 1) for _ in range(steps)]
            step_time = maturity / steps
            expected = [spot]
            for index in range(1, steps + 1):
                log_return = (
                    (rate - volatility**2 / 2) * index * step_time
                    + volatility * math.sqrt(step_time) * math.fsum(increments[:index])
                )
                expected.append(spot * math.exp(log_return))
            actual = simulate_gbm_path(spot, maturity, rate, volatility, steps, seed=seed)
            with self.subTest(steps=steps, seed=seed, rate=rate):
                for actual_spot, expected_spot in zip(actual, expected):
                    self.assertAlmostEqual(actual_spot, expected_spot, delta=1e-10)

    def test_seed_reproduces_path_and_different_seeds_change_it(self):
        for seed in (0, 42, -17):
            with self.subTest(seed=seed):
                self.assertEqual(
                    simulate_gbm_path(**BASE_PARAMETERS, steps=8, seed=seed),
                    simulate_gbm_path(**BASE_PARAMETERS, steps=8, seed=seed),
                )
        self.assertNotEqual(
            simulate_gbm_path(**BASE_PARAMETERS, steps=8, seed=11),
            simulate_gbm_path(**BASE_PARAMETERS, steps=8, seed=29),
        )

    def test_preserves_global_random_state(self):
        for seed in (7, None):
            with self.subTest(seed=seed):
                state_before = random.getstate()
                try:
                    simulate_gbm_path(**BASE_PARAMETERS, steps=8, seed=seed)
                    self.assertEqual(random.getstate(), state_before)
                finally:
                    random.setstate(state_before)

    def test_zero_volatility_gives_deterministic_growth(self):
        for rate, steps, seed in product((-0.1, 0, 0.1), (1, 8), (7, None)):
            path = simulate_gbm_path(100, 2, rate, 0, steps, seed=seed)
            with self.subTest(rate=rate, steps=steps, seed=seed):
                for index, spot in enumerate(path):
                    expected = 100 * math.exp(rate * 2 * index / steps)
                    self.assertAlmostEqual(spot, expected, delta=1e-10)

    def test_accepts_real_financial_inputs(self):
        rational_path = simulate_gbm_path(
            Fraction(100), Fraction(1), Fraction(1, 20), Fraction(1, 5), steps=4, seed=3
        )
        self.assertEqual(rational_path, simulate_gbm_path(**BASE_PARAMETERS, steps=4, seed=3))
        self.assertEqual(
            simulate_gbm_path(100, 1, 0, 1, steps=4, seed=3),
            simulate_gbm_path(100.0, 1.0, 0.0, 1.0, steps=4, seed=3),
        )

    def test_rejects_nonreal_financial_inputs(self):
        for name, value in product(BASE_PARAMETERS, INVALID_NUMERIC_TYPES):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                simulate_gbm_path(**parameters, steps=1, seed=0)

    def test_rejects_nonfinite_values_and_invalid_financial_domains(self):
        for name, value in product(BASE_PARAMETERS, (math.nan, math.inf, -math.inf)):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                simulate_gbm_path(**parameters, steps=1, seed=0)
        for name, value in (
            ("spot", 0), ("spot", -1),
            ("time_to_maturity", 0), ("time_to_maturity", -1),
            ("volatility", -0.1),
        ):
            parameters = {**BASE_PARAMETERS, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                simulate_gbm_path(**parameters, steps=1, seed=0)

    def test_validates_steps_even_for_zero_volatility(self):
        invalid_types = (
            True, False, None, "4", 4.0, Fraction(4), Decimal("4"),
            1j, [], {}, math.nan, math.inf,
        )
        for volatility in (0, 0.2):
            parameters = {**BASE_PARAMETERS, "volatility": volatility}
            for steps in invalid_types:
                with self.subTest(volatility=volatility, steps=steps):
                    with self.assertRaises(TypeError):
                        simulate_gbm_path(**parameters, steps=steps, seed=0)
            for steps in (0, -1):
                with self.subTest(volatility=volatility, steps=steps):
                    with self.assertRaises(ValueError):
                        simulate_gbm_path(**parameters, steps=steps, seed=0)

    def test_validates_seed_even_for_zero_volatility(self):
        invalid_seeds = (
            True, False, "4", 4.0, Fraction(4), Decimal("4"),
            1j, [], {}, math.nan, math.inf,
        )
        for volatility, seed in product((0, 0.2), invalid_seeds):
            parameters = {**BASE_PARAMETERS, "volatility": volatility}
            with self.subTest(volatility=volatility, seed=seed), self.assertRaises(TypeError):
                simulate_gbm_path(**parameters, steps=1, seed=seed)

    def test_reports_overflow_and_underflow_to_zero(self):
        for rate in (-1000.0, 1000.0):
            with self.subTest(rate=rate), self.assertRaises(OverflowError):
                simulate_gbm_path(1, 1, rate, 0, steps=1, seed=0)


if __name__ == "__main__":
    unittest.main()
