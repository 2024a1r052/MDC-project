"""Quick self-checks:  python -m unittest -v"""
import unittest

import numpy as np

from anomaly_core import NUMERIC, Detector, evaluate, generate_flows, load_uploaded, split_data


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_flows(6000, 0.12, 1)
        cls.train, cls.test = split_data(cls.df, 0.3, 1)

    def test_generator_shape_and_ratio(self):
        self.assertEqual(len(self.df), 6000)
        self.assertAlmostEqual(self.df["label"].mean(), 0.12, places=2)
        self.assertFalse(self.df[NUMERIC].isna().any().any())

    def test_isolation_forest_beats_chance(self):
        det = Detector(Detector.ISO).fit(self.train, self.train["label"])
        s = det.score(self.test)
        m = evaluate(self.test["label"], s, (s > det.threshold(0.97)).astype(int))
        self.assertGreater(m["auc"], 0.9)
        self.assertLess(m["fpr"], 0.06)

    def test_random_forest_needs_labels(self):
        with self.assertRaises(ValueError):
            Detector(Detector.RF).fit(self.train.drop(columns=["label"]))

    def test_reasons_are_returned_per_row(self):
        det = Detector(Detector.ISO).fit(self.train, self.train["label"])
        self.assertEqual(len(det.reasons(self.test.head(5))), 5)

    def test_upload_validation(self):
        with self.assertRaises(ValueError):
            load_uploaded(self.df[["protocol", "duration"]])
        back = load_uploaded(self.df.copy())
        self.assertEqual(len(back), len(self.df))


if __name__ == "__main__":
    unittest.main()
