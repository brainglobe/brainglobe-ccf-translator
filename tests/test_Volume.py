# import os
# import sys
# sys.path.append(os.path.abspath("/home/harryc/github/brainglobe_ccf_translator/"))

import unittest
import numpy as np
import json
import os
from brainglobe_ccf_translator import Volume



class TestVolume(unittest.TestCase):
    def setUp(self):
        self.test_case_dir = os.path.join(os.path.dirname(__file__), 'Volume_test_cases')

    def load_test_case(self, filename):
        with open(os.path.join(self.test_case_dir, filename), 'r') as file:
            return json.load(file)

    def run_test_case(self, test_case_filename):
        test_case = self.load_test_case(test_case_filename)
        input_data_path = test_case['input']
        expected_output_path = test_case['expected_output']
        input_data = np.load(input_data_path)
        expected_output_data = np.load(expected_output_path)
        
        vol = Volume(  # Assuming volume is a class named Volume
            input_data['reference'],
            test_case['space'],
            test_case['voxel_size_micron'],
            test_case['age_PND'],
            test_case['segmentation_file']
        )

        vol.transform(
            test_case['target_age'],
            test_case['target_space'],
        )

        np.testing.assert_array_almost_equal(vol.values, expected_output_data['reference'])
        # self.assertEqual(vol.age_PND, )
        # self.assertEqual(vol.space, expected_output['space'])

# List of test case filenames
test_case_files = [
    'allen_to_perens_lsfm_mouse.json',
    'allen_to_demba_P5.json',
    'allen_to_demba_P56.json',
]

# Dynamically create test methods for each test case file
for test_case_file in test_case_files:
    def test_method(self, test_case_file=test_case_file):
        self.run_test_case(test_case_file)
    setattr(TestVolume, f'test_{test_case_file.split(".")[0]}', test_method)

if __name__ == '__main__':
    unittest.main()