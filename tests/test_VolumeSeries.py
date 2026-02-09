## Useful for debugger
# from pathlib import Path
# import sys
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
import json
import os
import unittest

import numpy as np

from brainglobe_ccf_translator import Volume, VolumeSeries


class TestVolumeSeries(unittest.TestCase):
    def setUp(self):
        self.test_case_dir = os.path.join(
            os.path.dirname(__file__), "VolumeSeries_test_cases"
        )
        self.voxel_size_micron = 200
        self.space_name = "demba_dev_mouse"
        self.data_folder = "tests/test_data/volumes"

    def load_test_case(self, filename):
        with open(os.path.join(self.test_case_dir, filename), "r") as file:
            return json.load(file)

    def run_test_case(self, test_case_filename):
        test_case = self.load_test_case(test_case_filename)
        key_ages = test_case["key_ages"]
        expected_output_dir = test_case["expected_output_dir"]

        volumes = []
        for age in key_ages:
            volume_path = os.path.join(
                self.data_folder, f"demba_P{age}_mouse_200um.npz"
            )
            try:
                volume_data = np.load(volume_path)["reference"]
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
            expected_volume_path = os.path.join(
                expected_output_dir,
                f"demba_dev_mouse_P{v.age_PND}_interpolated.npz",
            )
            expected_volume_data = np.load(expected_volume_path)["reference"]
            np.testing.assert_array_almost_equal(
                v.values, expected_volume_data
            )


# List of test case filenames
test_case_files = ["demba_p4_interpolate_to_p8.json"]


# Factory to avoid module-level pytest collection of the helper
def _make_test(test_case_file):
    def _test(self):
        self.run_test_case(test_case_file)

    return _test


# Dynamically create test methods for each test case file
for test_case_file in test_case_files:
    setattr(
        TestVolumeSeries,
        f"test_{test_case_file.split('.')[0]}",
        _make_test(test_case_file),
    )

if __name__ == "__main__":
    unittest.main()
