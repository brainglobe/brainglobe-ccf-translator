"""
Manual validation script for PointSet transformations.
Plots points in 3 planes (coronal, sagittal, horizontal) in both source and target volumes.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from brainglobe_atlasapi import BrainGlobeAtlas
import brainglobe_ccf_translator as ccft


# Mapping from space names to BrainGlobe atlas names
SPACE_TO_ATLAS = {
    "allen_mouse": "allen_mouse_25um",
    "demba_dev_mouse": {
        4: "demba_allen_seg_dev_mouse_p4_20um",
        7: "demba_allen_seg_dev_mouse_p7_20um",
        14: "demba_allen_seg_dev_mouse_p14_20um",
        28: "demba_allen_seg_dev_mouse_p28_20um",
        32: "demba_allen_seg_dev_mouse_p28_20um",  # Use closest available
        56: "demba_allen_seg_dev_mouse_p56_20um",
    },
    "perens_stpt_mouse": "perens_stpt_mouse_25um",
    "perens_multimodal_lsfm": "perens_multimodal_lsfm_25um",
    "perens_mri_mouse": "perens_stereotaxic_mri_mouse_25um",
    "perens_stereotaxic_mri_mouse": "perens_stereotaxic_mri_mouse_25um",
    "princeton_mouse": "princeton_mouse_20um",
}


def get_atlas_name(space, age):
    """Get BrainGlobe atlas name from space and age."""
    atlas_info = SPACE_TO_ATLAS.get(space)
    if isinstance(atlas_info, dict):
        return atlas_info.get(age, atlas_info.get(56))  # Default to P56 if age not found
    return atlas_info


def plot_point_in_volume(ax, volume, point, plane, title, color='red', marker_size=100):
    """
    Plot a single plane with a point marked.

    Args:
        ax: matplotlib axis
        volume: 3D numpy array (AP, DV, LR assumed)
        point: [AP, DV, LR] coordinates in voxels
        plane: 'coronal', 'sagittal', or 'horizontal'
        title: plot title
        color: marker color
        marker_size: size of the point marker
    """
    ap, dv, lr = [int(round(p)) for p in point]

    # Clamp to volume bounds
    ap = np.clip(ap, 0, volume.shape[0] - 1)
    dv = np.clip(dv, 0, volume.shape[1] - 1)
    lr = np.clip(lr, 0, volume.shape[2] - 1)

    if plane == 'coronal':
        slice_img = volume[ap, :, :]
        marker_x, marker_y = lr, dv
        xlabel, ylabel = 'LR', 'DV'
    elif plane == 'sagittal':
        slice_img = volume[:, :, lr]
        marker_x, marker_y = dv, ap
        xlabel, ylabel = 'DV', 'AP'
    elif plane == 'horizontal':
        slice_img = volume[:, dv, :]
        marker_x, marker_y = lr, ap
        xlabel, ylabel = 'LR', 'AP'
    else:
        raise ValueError(f"Unknown plane: {plane}")

    ax.imshow(slice_img, cmap='gray')
    ax.scatter([marker_x], [marker_y], c=color, s=marker_size, marker='x', linewidths=2)
    ax.set_title(f"{title}\n{plane.capitalize()} (idx={[ap, dv, lr][['coronal', 'sagittal', 'horizontal'].index(plane)]})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def validate_test_case(test_case_path):
    """Load and validate a single test case."""
    with open(test_case_path, 'r') as f:
        test_case = json.load(f)

    points = np.array(test_case["points"])
    scale = test_case["scale"]
    target_space = test_case["target"]
    target_age = test_case["target_age"]
    expected_values = np.array(test_case["expected_values"])

    # Source is always allen_mouse P56 based on test setup
    source_space = "allen_mouse"
    source_age = 56

    # Load atlases
    source_atlas_name = get_atlas_name(source_space, source_age)
    target_atlas_name = get_atlas_name(target_space, target_age)

    print(f"Loading source atlas: {source_atlas_name}")
    source_atlas = BrainGlobeAtlas(source_atlas_name)

    print(f"Loading target atlas: {target_atlas_name}")
    target_atlas = BrainGlobeAtlas(target_atlas_name)

    # Convert points to voxel coordinates for each atlas
    source_res = source_atlas.resolution[0]  # Assuming isotropic
    target_res = target_atlas.resolution[0]

    for i, (src_point, exp_point) in enumerate(zip(points, expected_values)):
        # Scale points to voxel coordinates
        # Input points are in 25um space (scale factor applied)
        src_voxel = (np.array(src_point) / scale) * (25 / source_res)
        exp_voxel = (np.array(exp_point) / scale) * (25 / target_res)

        # Create figure with 2 rows (source/target) x 3 columns (planes)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        planes = ['coronal', 'sagittal', 'horizontal']

        # Plot source
        for j, plane in enumerate(planes):
            plot_point_in_volume(
                axes[0, j],
                source_atlas.reference,
                src_voxel,
                plane,
                f"Source: {source_space} P{source_age}",
                color='lime'
            )

        # Plot target (expected)
        for j, plane in enumerate(planes):
            plot_point_in_volume(
                axes[1, j],
                target_atlas.reference,
                exp_voxel,
                plane,
                f"Target: {target_space} P{target_age}",
                color='red'
            )

        fig.suptitle(
            f"Point {i+1}: {src_point} -> {exp_point}\n"
            f"Test case: {os.path.basename(test_case_path)}",
            fontsize=12
        )
        plt.tight_layout()
        plt.show()


def main():
    test_cases_dir = os.path.join(
        os.path.dirname(__file__), "PointSet_test_cases"
    )

    test_case_files = [
        "perens_stpt_mouse.json",
        "demba_dev_mouse_56.json",
        "perens_multimodal_lsfm.json",
        "perens_mri_mouse.json",
        "demba_dev_mouse_32.json",
    ]

    for filename in test_case_files:
        filepath = os.path.join(test_cases_dir, filename)
        if os.path.exists(filepath):
            print(f"\n{'='*60}")
            print(f"Validating: {filename}")
            print('='*60)
            try:
                validate_test_case(filepath)
            except Exception as e:
                print(f"Error validating {filename}: {e}")
        else:
            print(f"Test case not found: {filepath}")


if __name__ == "__main__":
    main()
