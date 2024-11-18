# import os
# import sys

# sys.path.append(os.path.abspath("/home/harryc/github/brainglobe_ccf_translator/"))
import unittest
import numpy as np
import json
import os
from brainglobe_ccf_translator import PointSet

class TestPointset(unittest.TestCase):
    def setUp(self):
        self.test_case_dir = os.path.join(os.path.dirname(__file__), 'PointSet_test_cases')

    def load_test_case(self, filename):
        with open(os.path.join(self.test_case_dir, filename), 'r') as file:
            return json.load(file)

    def run_test_case(self, test_case_filename):
        test_case = self.load_test_case(test_case_filename)
        points = np.array(test_case['points']) / test_case['scale']
        target = test_case['target']
        expected_values = np.array(test_case['expected_values'])

        pset = PointSet(points, 'allen_mouse', voxel_size_micron=25, age_PND=56)
        pset.transform(target_age=test_case['target_age'], target_space=target)

        np.testing.assert_array_almost_equal(pset.values * test_case['scale'], expected_values)

# List of test case filenames
test_case_files = [
    'perens_stpt_mouse.json',
    'demba_dev_mouse_56.json',
    'perens_lsfm_mouse.json',
    'perens_mri_mouse.json',
    'demba_dev_mouse_32.json',
]

# Dynamically create test methods for each test case file
for test_case_file in test_case_files:
    def test_method(self, test_case_file=test_case_file):
        self.run_test_case(test_case_file)
    setattr(TestPointset, f'test_{test_case_file.split(".")[0]}', test_method)

if __name__ == '__main__':
    unittest.main()