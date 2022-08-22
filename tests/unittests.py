import unittest
import pandas as pd
import numpy as np
from mcp import mcp_curve, mcp_score

class TestMCPCurve(unittest.TestCase):
    def setUp(self):
        with open('test_results.csv', 'r') as file:
            input_set = pd.read_csv(file)
            
        self.y_true = input_set['y_true']
        self.y_score = input_set.loc[:, 'y_score_1':]
        
        with open('test_mcp_curve.csv', 'r') as file:
            self.output_set = pd.read_csv(file)
        
        
    def test_range_values(self):
        # test numbers of samples
        x, y = mcp_curve(self.y_true, self.y_score)
        self.assertEqual(len(y), len(self.output_set), msg='calculated curve has different number of samples than correct one')
        
        
    def test_points(self):
        # test if calculated points are the same
        x, y = mcp_curve(self.y_true, self.y_score)
        self.assertAlmostEqual(all(y), all(self.output_set['y']), msg='points on calculated curve differ from same points on correct curve')
        
        
    def test_area(self):
        # test whether areas are the same
        area = mcp_score(self.y_true, self.y_score)
        real_curve_area = np.trapz(self.output_set['y'], x=self.output_set['x'])
        self.assertAlmostEqual(area, real_curve_area, msg='areas not equal')
        
        
    def test_length_error(self):
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, self.y_score[:-2])
            
        
    def test_prob_error(self):
        y_score = self.y_score.copy()
        y_score.iloc[0, 0] = 2
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, y_score)
        
        
    def test_classes_error(self):
        with self.assertRaises(ValueError):
            mcp_curve(self.y_true, self.y_score.iloc[:, 0:5])
        
if __name__ == '__main__':
    unittest.main()