import os
import json
import numpy as np
from brainglobe_ccf_translator import PointSet
"""
If you want to validate that this output is correct run "manual_plot_all_pointset_outputs.py"
in the directory above.
"""

test_data_dir = os.path.dirname(__file__)
pointset_test_cases_dir = os.path.join(
    os.path.dirname(test_data_dir), "../PointSet_test_cases"
)

# Define test cases: input parameters (without expected_values)
test_cases = {
    "demba_dev_mouse_56.json": {
        "points": [[1195, 268, 575]],
        "scale": 2.5,
        "target": "demba_dev_mouse",
        "target_age": 56,
    },
    "perens_multimodal_lsfm.json": {
        "points": [[1195, 268, 575]],
        "scale": 2.5,
        "target": "perens_multimodal_lsfm",
        "target_age": 56,
    },
    "perens_mri_mouse.json": {
        "points": [[1195, 268, 575]],
        "scale": 2.5,
        "target": "perens_stereotaxic_mri_mouse",
        "target_age": 56,
    },
    "demba_dev_mouse_32.json": {
        "points": [[1195, 268, 575]],
        "scale": 2.5,
        "target": "demba_dev_mouse",
        "target_age": 32,
    },
}


def generate_expected_values(filename, test_case):
    os.makedirs(pointset_test_cases_dir, exist_ok=True)
    filepath = os.path.join(pointset_test_cases_dir, filename)

    points = np.array(test_case["points"]) / test_case["scale"]
    target = test_case["target"]
    target_age = test_case["target_age"]
    scale = test_case["scale"]

    pset = PointSet(points, "allen_mouse", voxel_size_micron=25, age_PND=56)
    pset.transform(target_age=target_age, target_space=target)

    # Add expected values (scaled back)
    test_case["expected_values"] = (pset.values * scale).tolist()

    with open(filepath, "w") as file:
        json.dump(test_case, file, indent=4)

    print(f"Generated: {filename}")


if __name__ == "__main__":
    for filename, test_case in test_cases.items():
        generate_expected_values(filename, test_case.copy())

    print("All PointSet test cases generated.")
