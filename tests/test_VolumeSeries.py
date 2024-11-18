import os
import sys
sys.path.append(os.path.abspath("/home/harryc/github/brainglobe_ccf_translator/"))

import unittest
import nibabel as nib
import numpy as np
import json
from brainglobe_ccf_translator import VolumeSeries, Volume

class TestVolumeSeries(unittest.TestCase):
    def setUp(self):
        self.test_case_dir = os.path.join(os.path.dirname(__file__), 'VolumeSeries_test_cases')
        self.voxel_size_micron = 200
        self.space_name = "demba_dev_mouse"
        self.data_folder = "tests/test_data/volumes"

    def load_test_case(self, filename):
        with open(os.path.join(self.test_case_dir, filename), 'r') as file:
            return json.load(file)

    def run_test_case(self, test_case_filename):
        test_case = self.load_test_case(test_case_filename)
        key_ages = test_case['key_ages']
        expected_output_dir = test_case['expected_output_dir']

        volumes = []
        for age in key_ages:
            volume_path = os.path.join(self.data_folder, f"demba_P{age}_mouse_200um.npz")
            try:
                volume_data = np.load(volume_path)['reference']
            except FileNotFoundError:
                print(f"File not found: {volume_path}")
                continue
            volume = Volume(
                values=volume_data,
                space=self.space_name,
                voxel_size_micron=self.voxel_size_micron,
                segmentation_file=False,
                age_PND=age,
            )
            volumes.append(volume)

        volume_series = VolumeSeries(volumes)
        volume_series.interpolate_series()
        # Compare the expected outputs
        for v in volume_series.Volumes:
            expected_volume_path = os.path.join(expected_output_dir, f"demba_dev_mouse_P{v.age_PND}_interpolated.npz")
            expected_volume_data = np.load(expected_volume_path)['reference']
            np.testing.assert_array_almost_equal(v.values, expected_volume_data)
            

# List of test case filenames
test_case_files = [
    'demba_p4_interpolate_to_p8.json'
]

# Dynamically create test methods for each test case file
for test_case_file in test_case_files:
    def test_method(self, test_case_file=test_case_file):
        self.run_test_case(test_case_file)
    setattr(TestVolumeSeries, f'test_{test_case_file.split(".")[0]}', test_method)

if __name__ == '__main__':
    unittest.main()