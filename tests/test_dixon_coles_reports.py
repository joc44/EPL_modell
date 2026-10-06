"""DC matchday diagnostics: coverage, constraints and numerical acceptance."""

import unittest

import pandas as pd

from dixon_coles_backtest import run_dixon_coles_backtest
from dixon_coles_reports import validate_dixon_coles_diagnostics
from test_dixon_coles_backtest import small_matches


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matches = small_matches()
        cls.fixtures = cls.matches.loc[cls.matches.Season == "2023/24"]
        cls.predictions, cls.diagnostics = run_dixon_coles_backtest(cls.matches, "2023/24")

    def validate(self, diagnostics: pd.DataFrame, predictions: pd.DataFrame | None = None) -> None:
        validate_dixon_coles_diagnostics(diagnostics, self.predictions if predictions is None else predictions,
                                        self.fixtures, "2023/24")

    def test_real_daily_diagnostics_pass(self) -> None:
        self.validate(self.diagnostics)

    def test_real_active_boundary_accepts_large_raw_gradient_with_small_kkt_residual(self) -> None:
        records = [
            {"Season": "2022/23", "Date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
             "HomeTeam": home, "AwayTeam": away, "FTHG": 1, "FTAG": 1}
            for day in range(4) for home, away in (("A", "B"), ("B", "A"))
        ]
        records.append({"Season": "2023/24", "Date": pd.Timestamp("2023-08-12"),
                        "HomeTeam": "A", "AwayTeam": "B", "FTHG": 1, "FTAG": 1})
        matches = pd.DataFrame(records)
        matches["match_id"] = "EPL|" + matches.Season + "|" + matches.Date.dt.strftime("%Y-%m-%d") + "|" + matches.HomeTeam + "|" + matches.AwayTeam
        predictions, diagnostics = run_dixon_coles_backtest(matches, "2023/24", half_life_days=None)
        self.assertGreater(diagnostics.active_tau_constraints.iloc[0], 0)
        self.assertGreater(diagnostics.gradient_max_abs.iloc[0], 1)
        validate_dixon_coles_diagnostics(diagnostics, predictions, matches.loc[matches.Season == "2023/24"], "2023/24")

    def test_missing_duplicate_and_wrong_count_days_are_rejected(self) -> None:
        wrong_count = self.diagnostics.copy()
        wrong_count.loc[0, "fixtures"] += 1
        for bad in (self.diagnostics.iloc[:1], pd.concat([self.diagnostics, self.diagnostics.iloc[:1]]), wrong_count):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                self.validate(bad)

    def test_failed_dc_or_poisson_initial_fit_is_rejected(self) -> None:
        for column in ("success", "poisson_start_success"):
            bad = self.diagnostics.copy()
            bad.loc[0, column] = False
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "unsuccessful"):
                self.validate(bad)

    def test_same_day_history_is_rejected(self) -> None:
        bad = self.diagnostics.copy()
        bad.loc[0, "last_history_date"] = bad.loc[0, "fit_date"]
        with self.assertRaisesRegex(ValueError, "date boundary"):
            self.validate(bad)

    def test_inaccurate_stationarity_and_invalid_tau_are_rejected(self) -> None:
        for column, value in (("stationarity_max_abs", 0.01), ("poisson_start_gradient_max_abs", 0.01),
                               ("minimum_tau_all_pairs", -0.01), ("rho_upper", -100.0)):
            bad = self.diagnostics.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.validate(bad)

    def test_changed_rho_in_forecast_is_rejected(self) -> None:
        predictions = self.predictions.copy()
        predictions.loc[0, "rho"] += 0.01
        with self.assertRaisesRegex(ValueError, "rho differ"):
            self.validate(self.diagnostics, predictions)

    def test_changed_history_count_is_rejected(self) -> None:
        predictions = self.predictions.copy()
        predictions.loc[0, "history_matches"] += 1
        with self.assertRaisesRegex(ValueError, "forecast history"):
            self.validate(self.diagnostics, predictions)

    def test_worse_training_objective_and_wrong_parameter_count_are_rejected(self) -> None:
        for column, value in (("final_penalized_nll", float(self.diagnostics.initial_penalized_nll.iloc[0]) + 1),
                               ("free_parameters", 999)):
            bad = self.diagnostics.copy()
            bad.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.validate(bad)


if __name__ == "__main__":
    unittest.main()
