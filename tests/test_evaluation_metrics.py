"""Hand-checkable examples for the loss functions."""

from math import exp, log
import unittest

from evaluation_metrics import (
    LOG_LOSS_FLOOR,
    binary_metrics,
    observed_probability_loss,
    one_x_two_metrics,
    poisson_score_probability,
    realized_outcome,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_one_x_two_hand_calculation(self):
        probabilities = {"home": 0.7, "draw": 0.2, "away": 0.1}
        correct = one_x_two_metrics(probabilities, "home")
        self.assertAlmostEqual(correct["log_loss"], -log(0.7))
        self.assertAlmostEqual(correct["brier"], 0.14)
        self.assertTrue(correct["hit"])
        wrong = one_x_two_metrics(probabilities, "away")
        self.assertAlmostEqual(wrong["log_loss"], -log(0.1))
        self.assertAlmostEqual(wrong["brier"], 1.34)
        self.assertFalse(wrong["hit"])

    def test_uniform_and_perfect_predictions(self):
        uniform = one_x_two_metrics(dict.fromkeys(("home", "draw", "away"), 1 / 3), "draw")
        self.assertAlmostEqual(uniform["log_loss"], log(3))
        self.assertAlmostEqual(uniform["brier"], 2 / 3)
        perfect = one_x_two_metrics({"home": 1.0, "draw": 0.0, "away": 0.0}, "home")
        self.assertEqual(perfect["log_loss"], 0.0)
        self.assertEqual(perfect["brier"], 0.0)

    def test_binary_hand_calculation_and_complement(self):
        yes = binary_metrics(0.7, True)
        no = binary_metrics(0.7, False)
        self.assertAlmostEqual(yes["log_loss"], -log(0.7))
        self.assertAlmostEqual(yes["brier"], 0.09)
        self.assertAlmostEqual(no["log_loss"], -log(0.3))
        self.assertAlmostEqual(no["brier"], 0.49)
        complement = binary_metrics(0.3, False)
        self.assertAlmostEqual(yes["log_loss"], complement["log_loss"])
        self.assertAlmostEqual(yes["brier"], complement["brier"])

    def test_probability_floor_is_counted_without_changing_forecast(self):
        probabilities = {"home": 1.0, "draw": 0.0, "away": 0.0}
        result = one_x_two_metrics(probabilities, "away")
        self.assertEqual(result["log_loss"], -log(LOG_LOSS_FLOOR))
        self.assertEqual(result["brier"], 2.0)
        self.assertTrue(result["clipped"])
        self.assertEqual(probabilities["away"], 0.0)
        self.assertFalse(observed_probability_loss(LOG_LOSS_FLOOR)["clipped"])
        self.assertTrue(observed_probability_loss(LOG_LOSS_FLOOR / 2)["clipped"])

    def test_result_labels_and_invalid_goals(self):
        self.assertEqual(realized_outcome(2, 1), "home")
        self.assertEqual(realized_outcome(0, 0), "draw")
        self.assertEqual(realized_outcome(1, 3), "away")
        for home, away in [(-1, 0), (1.5, 0), (True, 0)]:
            with self.subTest(home=home, away=away), self.assertRaises(ValueError):
                realized_outcome(home, away)

    def test_exact_score_probability_and_zero_lambda(self):
        self.assertAlmostEqual(poisson_score_probability(1.2, 0.8, 2, 1), exp(-2) * 0.576)
        self.assertEqual(poisson_score_probability(0, 0, 0, 0), 1.0)
        self.assertEqual(poisson_score_probability(0, 1, 1, 0), 0.0)
        self.assertEqual(poisson_score_probability(0, 0, 1, 1), 0.0)
        # A 40–0 score lies outside the usual small-λ score table,
        # but has a positive probability that can be calculated directly from the distribution.
        probability = poisson_score_probability(2, 1, 40, 0)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1e-15)

    def test_invalid_probabilities_labels_and_events(self):
        for probability in [-0.1, 1.1, float("nan"), float("inf"), True]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                binary_metrics(probability, True)
        for observed in [-1, 2, 0.5, "yes"]:
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                binary_metrics(0.5, observed)
        for probabilities, outcome in [
            ({"home": 0.5, "draw": 0.3, "away": 0.3}, "home"),
            ({"home": 0.5, "draw": 0.5}, "home"),
            ({"home": 0.5, "draw": 0.3, "away": 0.2}, "other"),
        ]:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                one_x_two_metrics(probabilities, outcome)

    def test_invalid_expected_goals(self):
        for expected in [-1, float("nan"), float("inf"), True]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                poisson_score_probability(expected, 1, 0, 0)


if __name__ == "__main__":
    unittest.main()
