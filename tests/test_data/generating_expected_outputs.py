import os
import sys

sys.path.append(os.path.abspath("/home/harryc/github/brainglobe_ccf_translator/"))

import os
import sys


import brainglobe_ccf_translator
import numpy as np


# test_data_dir = os.path.dirname(__file__)
test_data_dir = r"/home/harryc/github/brainglobe_ccf_translator/tests/test_data/"

princeton_atlas = np.load(
    os.path.join(test_data_dir, "volumes", "princeton_mouse_200um.npz")
)
reference = princeton_atlas["reference"]
annotation = princeton_atlas["annotation"]
volume = brainglobe_ccf_translator.Volume(
    values=reference, space="princeton_mouse", age_PND=56, voxel_size_micron=200
)
volume.transform(target_space="perens_lsfm_mouse", target_age=56)
reference_transformed = volume.values

volume = brainglobe_ccf_translator.Volume(
    values=annotation,
    space="princeton_mouse",
    age_PND=56,
    voxel_size_micron=200,
    segmentation_file=True,
)
volume.transform(target_space="perens_lsfm_mouse", target_age=56)

annotation_transformed = volume.values

np.savez_compressed(
    os.path.join(
        test_data_dir, "expected_outputs", "princeton_mouse_to_perens_lsfm_mouse"
    ),
    reference=reference_transformed,
    annotation=annotation_transformed,
)

allen_atlas = np.load(os.path.join(test_data_dir, "volumes", "allen_mouse_200um.npz"))
reference = allen_atlas["reference"]
annotation = allen_atlas["annotation"]
volume = brainglobe_ccf_translator.Volume(
    values=reference, space="allen_mouse", age_PND=56, voxel_size_micron=200
)
volume.transform(target_space="demba_dev_mouse", target_age=56)

reference_transformed = volume.values

volume = brainglobe_ccf_translator.Volume(
    values=annotation,
    space="allen_mouse",
    age_PND=56,
    voxel_size_micron=200,
    segmentation_file=True,
)
volume.transform(target_space="demba_dev_mouse", target_age=56)

annotation_transformed = volume.values


np.savez_compressed(
    os.path.join(
        test_data_dir, "expected_outputs", "allen_mouse_to_demba_dev_mouse_P56"
    ),
    reference=reference_transformed,
    annotation=annotation_transformed,
)

############################

allen_atlas = np.load(os.path.join(test_data_dir, "volumes", "allen_mouse_200um.npz"))
reference = allen_atlas["reference"]
annotation = allen_atlas["annotation"]
volume = brainglobe_ccf_translator.Volume(
    values=reference, space="allen_mouse", age_PND=56, voxel_size_micron=200
)
volume.transform(target_space="demba_dev_mouse", target_age=5)

reference_transformed = volume.values

volume = brainglobe_ccf_translator.Volume(
    values=annotation,
    space="allen_mouse",
    age_PND=56,
    voxel_size_micron=200,
    segmentation_file=True,
)
volume.transform(target_space="demba_dev_mouse", target_age=5)

annotation_transformed = volume.values


np.savez_compressed(
    os.path.join(
        test_data_dir, "expected_outputs", "allen_mouse_to_demba_dev_mouse_P5"
    ),
    reference=reference_transformed,
    annotation=annotation_transformed,
)


############################

demba_p4 = np.load(os.path.join(test_data_dir, "volumes", "demba_P4_mouse_200um.npz"))
demba_p4 = demba_p4["reference"]
volume_p4 = brainglobe_ccf_translator.Volume(
    values=demba_p4, space="demba_dev_mouse", age_PND=4, voxel_size_micron=200
)
demba_p7 = np.load(os.path.join(test_data_dir, "volumes", "demba_P7_mouse_200um.npz"))
demba_p7 = demba_p7["reference"]
volume_p7 = brainglobe_ccf_translator.Volume(
    values=demba_p7, space="demba_dev_mouse", age_PND=7, voxel_size_micron=200
)


demba_p8 = np.load(os.path.join(test_data_dir, "volumes", "demba_P8_mouse_200um.npz"))
demba_p8 = demba_p8["reference"]
volume_p8 = brainglobe_ccf_translator.Volume(
    values=demba_p8, space="demba_dev_mouse", age_PND=8, voxel_size_micron=200
)

series = brainglobe_ccf_translator.VolumeSeries(
    [
        volume_p7,
        volume_p4,
    ]
)
series.interpolate_series()

for volume in series.Volumes:
    np.savez_compressed(
        os.path.join(
            test_data_dir,
            "expected_outputs",
            f"demba_dev_mouse_P{volume.age_PND}_interpolated",
        ),
        reference=volume.values,
    )


import matplotlib.pyplot as plt

plt.imshow(reference_transformed[30])
####################################

# allen_atlas = np.load(os.path.join(test_data_dir, 'volumes', 'allen_mouse_200um.npz'))
# reference = allen_atlas['reference']
# annotation = allen_atlas['annotation']
# volume = brainglobe_ccf_translator.Volume(
#     values = reference,
#     space = 'allen_mouse',
#     age_PND = 56,
#     voxel_size_micron=200
# )
# volume.transform(
#     target_space='perens_lsfm_mouse',
#     target_age=56
# )

# reference_transformed = volume.values

# volume = brainglobe_ccf_translator.Volume(
#     values = annotation,
#     space = 'allen_mouse',
#     age_PND = 56,
#     voxel_size_micron=200,
#     segmentation_file=True
# )
# volume.transform(
#     target_space='perens_lsfm_mouse',
#     target_age=56
# )

# annotation_transformed = volume.values

# np.savez_compressed(os.path.join(test_data_dir,'expected_outputs', 'allen_mouse_to_perens_lsfm_mouse'), reference=reference_transformed, annotation=annotation_transformed)
