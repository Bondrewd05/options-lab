"""Pairing, accounting, and statistical contracts for hedge experiments."""

import json
import math
import random
import unittest
from unittest.mock import patch
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, product

from options_lab import (
    black_scholes_price,
    delta_hedge_path,
    run_hedging_experiments,
    simulate_gbm_path,
)


class HedgingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_hedging_experiments(3, seed_start=31)
        cls.scenarios = {item["id"]: item for item in cls.report["scenarios"]}
        cls.masters = [
            simulate_gbm_path(100, 1, 0.05, 0.2, steps=3276, seed=seed)
            for seed in range(31, 34)
        ]

    def test_scenarios_cover_the_requested_cartesian_products(self):
        self.assertEqual(len(self.report["scenarios"]), 18)
        self.assertEqual(len(self.scenarios), 18)
        frequency_cases = {
            (item["steps"], item["transaction_cost_rate"])
            for item in self.scenarios.values() if item["family"] == "frequency_costs"
        }
        volatility_cases = {
            (item["hedge_volatility"], item["premium_convention"])
            for item in self.scenarios.values() if item["family"] == "volatility"
        }
        self.assertEqual(frequency_cases, set(product((12, 52, 252), (0, .0005, .001, .002))))
        self.assertEqual(
            volatility_cases,
            set(product((.15, .2, .25), ("fixed_true_model", "hedge_model"))),
        )
        for item in self.scenarios.values():
            for field in ("terminal_pnls", "nominal_costs", "terminal_costs"):
                self.assertEqual(len(item[field]), 3)

    def test_subgrids_have_exact_endpoints_and_shared_observations(self):
        self.assertEqual(self.report["master_steps"], 3276)
        paths = self.report["example_paths"]
        timed_paths = {}
        for steps in (12, 52, 252):
            path = paths[str(steps)]
            self.assertEqual(path, list(self.masters[0][::3276 // steps]))
            self.assertEqual(len(path), steps + 1)
            self.assertEqual(path[0], 100)
            self.assertEqual(path[-1], self.report["terminal_spots"][0])
            timed_paths[steps] = {Fraction(i, steps): spot for i, spot in enumerate(path)}
        for left_steps, right_steps in combinations((12, 52, 252), 2):
            left, right = timed_paths[left_steps], timed_paths[right_steps]
            shared_times = left.keys() & right.keys()
            self.assertEqual(len(shared_times), math.gcd(left_steps, right_steps) + 1)
            for time in shared_times:
                self.assertEqual(left[time], right[time])

    def test_each_sample_matches_the_same_seeded_path_across_scenarios(self):
        self.assertEqual(self.report["terminal_spots"], [path[-1] for path in self.masters])
        for item in self.scenarios.values():
            for index, master in enumerate(self.masters):
                path = master[::3276 // item["steps"]]
                result = delta_hedge_path(
                    path, 100, 1, .05, item["hedge_volatility"],
                    initial_premium=item["initial_premium"],
                    transaction_cost_rate=item["transaction_cost_rate"],
                )
                with self.subTest(scenario=item["id"], sample=index):
                    self.assertEqual(item["terminal_pnls"][index], result.terminal_pnl)
                    self.assertEqual(item["nominal_costs"][index], result.total_transaction_cost)
                    self.assertEqual(item["terminal_costs"][index], result.terminal_transaction_cost)

    def test_premium_conventions_shift_every_sample_by_financed_premium(self):
        true_premium = black_scholes_price(100, 100, 1, .05, .2)
        self.assertEqual(self.report["true_model_premium"], true_premium)
        largest_error = 0.0
        for volatility in (.15, .2, .25):
            prefix = f"vol{round(volatility * 100)}_"
            fixed = self.scenarios[prefix + "fixed_true_model"]
            modeled = self.scenarios[prefix + "hedge_model"]
            model_premium = black_scholes_price(100, 100, 1, .05, volatility)
            shift = (model_premium - true_premium) * math.exp(.05)
            self.assertEqual(fixed["initial_premium"], true_premium)
            self.assertEqual(modeled["initial_premium"], model_premium)
            self.assertEqual(modeled["theoretical_mean_without_costs"], shift)
            self.assertEqual(fixed["theoretical_mean_without_costs"], 0)
            for baseline, changed in zip(fixed["terminal_pnls"], modeled["terminal_pnls"]):
                self.assertAlmostEqual(changed - baseline, shift, delta=1e-10)
                largest_error = max(largest_error, abs(changed - baseline - shift))
            self.assertAlmostEqual(
                fixed["summary"]["std_pnl"], modeled["summary"]["std_pnl"], delta=1e-10,
            )
        self.assertEqual(self.report["checks"]["max_premium_identity_error"], largest_error)

    def test_terminal_costs_explain_paired_pnl_losses(self):
        largest_error = 0.0
        for steps in (12, 52, 252):
            baseline = self.scenarios[f"n{steps}_cost0bp"]
            self.assertEqual(baseline["terminal_costs"], [0, 0, 0])
            self.assertEqual(baseline["nominal_costs"], [0, 0, 0])
            for basis_points in (5, 10, 20):
                charged = self.scenarios[f"n{steps}_cost{basis_points}bp"]
                for original, pnl, nominal, terminal in zip(
                    baseline["terminal_pnls"], charged["terminal_pnls"],
                    charged["nominal_costs"], charged["terminal_costs"],
                ):
                    self.assertGreater(nominal, 0)
                    self.assertGreaterEqual(terminal, nominal)
                    self.assertAlmostEqual(original - pnl, terminal, delta=1e-10)
                    largest_error = max(largest_error, abs(original - pnl - terminal))
        self.assertEqual(self.report["checks"]["max_cost_identity_error"], largest_error)

    def test_summaries_match_independent_sample_statistics(self):
        for item in self.scenarios.values():
            values = item["terminal_pnls"]
            mean = math.fsum(values) / 3
            variance = math.fsum((value - mean) ** 2 for value in values) / 2
            se = math.sqrt(variance / 3)
            low, middle, high = sorted(values)
            expected = {
                "mean_pnl": mean,
                "std_pnl": math.sqrt(variance),
                "rmse_pnl": math.sqrt(math.fsum(value ** 2 for value in values) / 3),
                "mean_standard_error": se,
                "percentile_5": .9 * low + .1 * middle,
                "median": middle,
                "percentile_95": .1 * middle + .9 * high,
                "negative_fraction": sum(value < 0 for value in values) / 3,
                "mean_nominal_cost": math.fsum(item["nominal_costs"]) / 3,
                "mean_terminal_cost": math.fsum(item["terminal_costs"]) / 3,
                "mean_pnl_plus_terminal_cost": math.fsum(
                    pnl + cost for pnl, cost in zip(values, item["terminal_costs"])
                ) / 3,
            }
            for field, value in expected.items():
                with self.subTest(scenario=item["id"], field=field):
                    self.assertAlmostEqual(item["summary"][field], value, delta=1e-11)
            lower, upper = item["summary"]["mean_confidence_interval_95_approx"]
            self.assertAlmostEqual(lower, mean - 1.959963984540054 * se, delta=1e-11)
            self.assertAlmostEqual(upper, mean + 1.959963984540054 * se, delta=1e-11)

    def test_paired_comparisons_use_the_variance_of_sample_differences(self):
        comparisons = self.report["paired_comparisons"]
        expected_pairs = {
            ("n252_cost0bp", "n12_cost0bp", "squared_pnl"),
            ("n252_cost10bp", "n12_cost10bp", "squared_pnl"),
            ("n52_cost10bp", "n52_cost0bp", "pnl"),
        }
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(
            {(item["left_id"], item["right_id"], item["quantity"]) for item in comparisons},
            expected_pairs,
        )
        for item in comparisons:
            left = self.scenarios[item["left_id"]]["terminal_pnls"]
            right = self.scenarios[item["right_id"]]["terminal_pnls"]
            power = 2 if item["quantity"] == "squared_pnl" else 1
            differences = [a ** power - b ** power for a, b in zip(left, right)]
            mean = math.fsum(differences) / 3
            se = math.sqrt(math.fsum((value - mean) ** 2 for value in differences) / 6)
            self.assertEqual(item["direction"], "left minus right")
            self.assertAlmostEqual(item["mean_difference"], mean, delta=1e-11)
            self.assertAlmostEqual(item["standard_error"], se, delta=1e-11)
            lower, upper = item["confidence_interval_95_approx"]
            self.assertAlmostEqual(lower, mean - 1.959963984540054 * se, delta=1e-11)
            self.assertAlmostEqual(upper, mean + 1.959963984540054 * se, delta=1e-11)

    def test_reproducibility_preserves_the_global_random_generator(self):
        state = random.getstate()
        try:
            self.assertEqual(run_hedging_experiments(3, seed_start=31), self.report)
            self.assertEqual(random.getstate(), state)
        finally:
            random.setstate(state)

    def test_two_paths_and_zero_seed_are_supported_with_stable_prefixes(self):
        small = run_hedging_experiments(2, seed_start=0)
        larger = run_hedging_experiments(3, seed_start=0)
        self.assertEqual(small["terminal_spots"], larger["terminal_spots"][:2])
        for left, right in zip(small["scenarios"], larger["scenarios"]):
            self.assertEqual(left["terminal_pnls"], right["terminal_pnls"][:2])
        self.assertNotEqual(small["terminal_spots"], self.report["terminal_spots"][:2])

    def test_negative_seed_start_is_rejected_before_generating_paths(self):
        with patch("options_lab.experiments.simulate_gbm_path") as simulate:
            for seed in (-1, -2026):
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    run_hedging_experiments(3, seed_start=seed)
            simulate.assert_not_called()

    def test_rejects_invalid_counts_and_seeds(self):
        invalid_types = (True, False, None, "3", 3.0, Fraction(3), Decimal(3), 1j, [], {}, math.nan, math.inf)
        for name, value in product(("n_paths", "seed_start"), invalid_types):
            parameters = {"n_paths": 2, "seed_start": 31, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                run_hedging_experiments(**parameters)
        for count in (-3, 0, 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                run_hedging_experiments(count)

    def test_report_round_trips_strict_json(self):
        encoded = json.dumps(self.report, allow_nan=False)
        self.assertEqual(json.loads(encoded), self.report)


if __name__ == "__main__":
    unittest.main()
