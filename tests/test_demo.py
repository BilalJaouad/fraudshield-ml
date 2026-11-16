"""Lightweight, isolated checks that require no model artifact or API key."""
import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudshield_demo import generate_records, expected_decision_costs, observed_confusion_costs


class SyntheticDemoTests(unittest.TestCase):
    def test_generated_profiles_match_baf_schema(self):
        for profile in ("routine", "mixed", "watchlist"):
            records = generate_records(60, seed=93, profile=profile)
            self.assertEqual(len(records), 60)
            self.assertTrue(all(len(r) == 31 for r in records))
            for r in records:
                self.assertTrue(0 <= r["month"] <= 7)
                self.assertEqual(r["device_fraud_count"], 0)
                self.assertTrue(0.1 <= r["income"] <= 0.9)
                self.assertIn(r["payment_type"], ("AA", "AB", "AC", "AD", "AE"))
                self.assertIn(r["device_os"], ("windows", "linux", "macintosh", "other", "x11"))
                self.assertIn(r["prev_address_months_count"], [-1, *range(365)])
                self.assertTrue(np.isfinite(float(r["velocity_6h"])))

    def test_seed_reproducibility(self):
        self.assertEqual(generate_records(5, 731), generate_records(5, 731))
        self.assertNotEqual(generate_records(5, 731), generate_records(5, 732))

    def test_cost_proxy_is_not_confusion(self):
        costs = expected_decision_costs([0.01, 0.08, 0.30], 0.06, 100, 1000)
        self.assertEqual(costs["review_count"], 2)
        self.assertAlmostEqual(costs["expected_fp_cost"], 162.0)
        self.assertAlmostEqual(costs["expected_fn_cost"], 10.0)
        self.assertAlmostEqual(costs["expected_total_cost"], 172.0)

    def test_labeled_confusion_cost(self):
        # Truth: legit, fraud, fraud, legit. Predictions flagged: case 2 + case 4.
        values = observed_confusion_costs([0, 1, 1, 0], [0.1, 0.8, 0.2, 0.7], 0.5, 100, 1000)
        self.assertEqual((values["tp"], values["fp"], values["fn"], values["tn"]), (1, 1, 1, 1))
        self.assertEqual(values["observed_total_cost"], 1100)
        self.assertEqual(values["no_alert_baseline"], 2000)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            generate_records(0)
        with self.assertRaises(ValueError):
            generate_records(1, profile="fraud")
        with self.assertRaises(ValueError):
            expected_decision_costs([1.2], 0.06, 100, 1000)
        with self.assertRaises(ValueError):
            observed_confusion_costs([0, 2], [0.1, 0.8], 0.5, 100, 1000)


if __name__ == "__main__":
    unittest.main()
