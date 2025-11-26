import numpy as np
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from scipy.ndimage import zoom
from pathlib import Path

test_data_dir = Path(__file__).parent


def generate_test_data(atlas_name, zoom_factor, output_name, include_annotation=True):
    """Generate downsampled test data from a BrainGlobe atlas."""
    test_atlas = BrainGlobeAtlas(atlas_name)
    reference = zoom(test_atlas.reference, zoom_factor)

    if include_annotation:
        annotation = zoom(test_atlas.annotation, zoom_factor, order=0)
        np.savez_compressed(
            test_data_dir / "volumes" / output_name,
            reference=reference,
            annotation=annotation,
        )
    else:
        np.savez_compressed(
            test_data_dir / "volumes" / output_name,
            reference=reference,
        )


# List of (atlas_name, zoom_factor, output_name, include_annotation)
atlas_configs = [
    ("allen_mouse_100um", 0.5, "allen_mouse_200um", True),
    ("princeton_mouse_20um", 0.1, "princeton_mouse_200um", True),
    ("perens_stereotaxic_mri_mouse_25um", 0.125, "perens_stereotaxic_mri_mouse_200um", False),
    ("demba_allen_seg_dev_mouse_p5_20um", 0.1, "demba_P5_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p4_20um", 0.1, "demba_P4_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p7_20um", 0.1, "demba_P7_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p8_20um", 0.1, "demba_P8_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p56_20um", 0.1, "demba_P56_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p28_20um", 0.1, "demba_P28_mouse_200um", True),
    ("demba_allen_seg_dev_mouse_p14_20um", 0.1, "demba_P14_mouse_200um", True),
]

for atlas_name, zoom_factor, output_name, include_annotation in atlas_configs:
    generate_test_data(atlas_name, zoom_factor, output_name, include_annotation)



