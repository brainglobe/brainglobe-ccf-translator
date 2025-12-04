"""
To ensure everything is working, this script will plot all combinations of volumes
(For Demba it will just plot P56 and P4 so as to not create a huge number)

Outputs images to a directory that can be uploaded with a pull request.
"""

import matplotlib.pyplot as plt
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
import brainglobe_ccf_translator as ccft
from pathlib import Path


def plot_mid_slice(volume_data, title, ax):
    # Find middle indices
    mid_idx = volume_data.shape[0] // 2

    # Plot slice (assuming index 0 is AP, so this is coronal)
    ax.imshow(volume_data[mid_idx, :, :], cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_overlay(transformed_data, target_data, trans_res, target_res, title, ax):
    """Plot two images overlaid with different colors, aligned by corner origin"""
    mid_idx_trans = transformed_data.shape[0] // 2
    mid_idx_target = target_data.shape[0] // 2

    trans_slice = transformed_data[mid_idx_trans, :, :]
    target_slice = target_data[mid_idx_target, :, :]

    trans_norm = (trans_slice - trans_slice.min()) / (
        trans_slice.max() - trans_slice.min() + 1e-8
    )
    target_norm = (target_slice - target_slice.min()) / (
        target_slice.max() - target_slice.min() + 1e-8
    )

    # Extent: [left, right, bottom, top] in physical units (microns)
    trans_extent = [
        0,
        trans_slice.shape[1] * trans_res,
        trans_slice.shape[0] * trans_res,
        0,
    ]
    target_extent = [
        0,
        target_slice.shape[1] * target_res,
        target_slice.shape[0] * target_res,
        0,
    ]

    ax.imshow(target_norm, cmap="gray", extent=target_extent, alpha=1)
    ax.imshow(trans_norm, cmap="gray", extent=trans_extent, alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


def main():
    output_dir = Path(__file__).parent / "overlay_manual_test_output"
    output_dir.mkdir(exist_ok=True)

    configs = [
        {
            "name": "Demba P28",
            "bg_name": "demba_allen_seg_dev_mouse_p28_20um",
            "space": "demba_dev_mouse",
            "age": 28,
            "res": 20,
        },
        {
            "name": "Allen P56",
            "bg_name": "allen_mouse_25um",
            "space": "allen_mouse",
            "age": 56,
            "res": 25,
        },
        {
            "name": "Demba P56",
            "bg_name": "demba_allen_seg_dev_mouse_p56_20um",
            "space": "demba_dev_mouse",
            "age": 56,
            "res": 20,
        },
        {
            "name": "Perens MRI P56",
            "bg_name": "perens_stereotaxic_mri_mouse_25um",
            "space": "perens_stereotaxic_mri_mouse",
            "age": 56,
            "res": 25,
        },
        {
            "name": "Demba P4",
            "bg_name": "demba_allen_seg_dev_mouse_p4_20um",
            "space": "demba_dev_mouse",
            "age": 4,
            "res": 20,
        },
        {
            "name": "Perens LSFM P56",
            "bg_name": "perens_multimodal_lsfm_25um",
            "space": "perens_multimodal_lsfm",
            "age": 56,
            "res": 25,
        },
        {
            "name": "Princeton P56",
            "bg_name": "princeton_mouse_20um",
            "space": "princeton_mouse",
            "age": 56,
            "res": 20,
        },
    ]

    # Pre-load atlases to save time and check availability
    atlases = {}
    for config in configs:
        try:
            print(f"Loading atlas: {config['bg_name']}")
            atlases[config["name"]] = BrainGlobeAtlas(config["bg_name"])
        except Exception as e:
            print(f"Could not load atlas {config['bg_name']}: {e}")

    for source_cfg in configs:
        if source_cfg["name"] not in atlases:
            continue

        source_atlas = atlases[source_cfg["name"]]

        for target_cfg in configs:
            if source_cfg == target_cfg:
                continue

            if target_cfg["name"] not in atlases:
                continue

            print(f"Transforming {source_cfg['name']} -> {target_cfg['name']}")

            try:
                # Create CCFT volume
                ccft_vol = ccft.Volume(
                    values=source_atlas.reference,
                    space=source_cfg["space"],
                    voxel_size_micron=source_cfg["res"],
                    age_PND=source_cfg["age"],
                )

                # Transform
                ccft_vol.transform(
                    target_age=target_cfg["age"], target_space=target_cfg["space"]
                )

                # Load target for comparison
                target_atlas = atlases[target_cfg["name"]]

                # Plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes = axes.flatten()

                # Source
                plot_mid_slice(
                    source_atlas.reference, f"Source: {source_cfg['name']}", axes[0]
                )

                # Transformed
                plot_mid_slice(
                    ccft_vol.values, f"Transformed to {target_cfg['name']}", axes[1]
                )

                # Target (Ground Truth)
                plot_mid_slice(
                    target_atlas.reference,
                    f"Target Ground Truth: {target_cfg['name']}",
                    axes[2],
                )

                # Overlay
                plot_overlay(
                    ccft_vol.values,
                    target_atlas.reference,
                    ccft_vol.voxel_size_micron,
                    target_cfg["res"],
                    "Overlay",
                    axes[3],
                )

                plt.suptitle(f"{source_cfg['name']} to {target_cfg['name']}")

                # Save to file instead of showing
                filename = f"{source_cfg['name']}_to_{target_cfg['name']}.png".replace(
                    " ", "_"
                )
                output_path = output_dir / filename
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {output_path}")

            except Exception as e:
                print(
                    f"Failed transformation {source_cfg['name']} -> {target_cfg['name']}: {e}"
                )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
