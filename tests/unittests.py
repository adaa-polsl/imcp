import unittest
import pandas as pd
import numpy as np
from mcp import mcp_curve, mcp_score, plot_curve, imcp_curve


class TestMCPCurve(unittest.TestCase):
    def setUp(self):
        with open("test_results.csv", "r") as file:
            input_set = pd.read_csv(file)

        self.y_true = input_set["y_true"]
        self.y_score = input_set.loc[:, "y_score_1":]

        with open("test_mcp_curve.csv", "r") as file:
            self.output_set = pd.read_csv(file)

        with open("test_imbalanced_class_probs.csv", "r") as file:
            self.imbalanced_probs = pd.read_csv(file)

    def test_mcp_range_values(self):
        x, y = mcp_curve(self.y_true, self.y_score)
        self.assertEqual(
            len(y),
            len(self.output_set),
            msg="calculated curve has different number of samples than ground truth curve",
        )

    def test_mcp_points(self):
        x, y = mcp_curve(self.y_true, self.y_score)
        self.assertAlmostEqual(
            all(y),
            all(self.output_set["y"]),
            msg="points on calculated curve differ from same points on ground truth curve",
        )

    def test_mcp_area(self):
        area = mcp_score(self.y_true, self.y_score)
        real_curve_area = np.trapz(self.output_set["y"], x=self.output_set["x"])
        self.assertAlmostEqual(area, real_curve_area, msg="areas not equal")

    def test_mcp_length_error(self):
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, self.y_score[:-2])

    def test_mcp_prob_error(self):
        y_score = self.y_score.copy()
        y_score.iloc[0, 0] = 2
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, y_score)

    def test_mcp_not_enough_class_probs_error(self):
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, self.y_score.iloc[:, 0:5])

    def test_mcp_abs_tolerance(self):
        y_score = np.array(
            [
                [0.33, 0.34, 0.32],
                [0.35, 0.32, 0.32],
                [0.29, 0.33, 0.37],
            ]
        )
        y_true = np.array([2, 1, 0])

        try:
            x, y = mcp_curve(y_true, y_score, abs_tolerance=0.1)
        except ValueError:
            self.fail("Failed with given tolerance threshold")

    def test_mcp_labels(self):
        y_score = np.array(
            [
                [0.1, 0.5, 0.4],
                [0.1, 0.7, 0.2],
                [0.1, 0.5, 0.4],
                [0.1, 0.3, 0.6],
                [0.1, 0.5, 0.4],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.3, 0.6],
            ]
        )

        y_true = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        # testing some user's inputs
        with self.assertRaises(ValueError):
            curve_x, curve_y = mcp_curve(y_true, y_score)

        with self.assertRaises(ValueError):
            curve_x, curve_y = mcp_curve(y_true, y_score, [1, 2])

        with self.assertRaises(ValueError):
            curve_x, curve_y = mcp_curve(y_true, y_score, [1, 2, 3, 4])

        with self.assertRaises(KeyError):
            curve_x, curve_y = mcp_curve(y_true, y_score, [1, 3, 4])

        y_true = np.array(["b", "b", "b", "b", "c", "c", "c", "c"])

        with self.assertRaises(KeyError):
            curve_x, curve_y = mcp_curve(y_true, y_score, ["a", "c", "d"])

        with self.assertRaises(TypeError):
            curve_x, curve_y = mcp_curve(y_true, y_score, [1, 2, "d"])

    def test_imcp_curve_lengths(self):
        curve_x, curve_y = imcp_curve(
            self.imbalanced_probs["y_true"],
            self.imbalanced_probs.drop("y_true", axis=1),
            abs_tolerance=0.000001,
        )

        self.assertEqual(
            len(curve_x),
            len(curve_y),
            msg="curve_x and curve_y are of different lengths!",
        )

        self.assertEqual(
            len(curve_x),
            len(self.imbalanced_probs["y_true"]) + 2,
            msg="curve_x should have 2 samples more than given y_true and y_score",
        )

    def test_plot(self):
        curve_1_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
        curve_2_x = curve_1_x
        curve_1_y = [0, 0.2, 0.4, 0.6]
        curve_2_y = [el / 2 for el in curve_1_y][:4]

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], curve_1_y)

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [curve_1_y])

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [curve_1_y, curve_2_y[:3]])

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [[curve_1_y, curve_2_y]])

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [curve_1_y, curve_2_y], label="test")

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [curve_1_y, curve_2_y], label=["test"])

        with self.assertRaises(ValueError):
            plot_curve(
                [curve_1_x, curve_2_x], [curve_1_y, curve_2_y], label=["test", 1]
            )

        with self.assertRaises(ValueError):
            plot_curve([curve_1_x, curve_2_x], [curve_1_y, curve_2_y], label=123)

        with self.assertRaises(ValueError):
            plot_curve(
                [curve_1_x, curve_2_x], [curve_1_y, curve_2_y], output_fig_path=123
            )


if __name__ == "__main__":
    unittest.main()
