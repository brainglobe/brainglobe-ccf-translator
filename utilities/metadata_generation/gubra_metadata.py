import numpy as np
import pandas as pd

metadata_path = r"/home/harryc/github/ccf_translator/ccf_translator/metadata/translation_metadata.csv"

metadata_template = {
    "file_name": [],
    "source_space": [],
    "target_space": [],
    "affine": [],
    "dim_order": [],
    "key_age": [],
    "source_age_pnd": [],
    "target_age_pnd": [],
    "source_key_age": [],
    "target_key_age": [],
    "padding_micron": [],
    "transformation_resolution_micron": [],
    "X_physical_size_micron": [],
    "Y_physical_size_micron": [],
    "Z_physical_size_micron": [],
    "target_X_physical_size_micron": [],
    "target_Y_physical_size_micron": [],
    "target_Z_physical_size_micron": [],
    "dim_flip": [],
    "vector": [],
}
source_spaces = [
    "perens_mri_mouse",
    "perens_lsfm_mouse",
    "perens_stpt_mouse",
    "perens_stpt_mouse",
    "perens_stpt_mouse",
    "allen_mouse",
]
target_spaces = [
    "perens_stpt_mouse",
    "perens_stpt_mouse",
    "perens_mri_mouse",
    "perens_lsfm_mouse",
    "allen_mouse",
    "perens_stpt_mouse",
]
age = "P56"
for i in range(len(target_spaces)):
    source = source_spaces[i]
    target = target_spaces[i]
    metadata_template["file_name"].append(f"{source}_pull_{target}.nii.gz")
    metadata_template["source_space"].append(source)
    metadata_template["target_space"].append(target)
    metadata_template["affine"].append(np.eye(4).tolist())
    if source == "allen_mouse":
        metadata_template["dim_order"].append([1, 2, 0])
        metadata_template["dim_flip"].append([False, True, False])
    elif target == "allen_mouse":
        metadata_template["dim_order"].append([2, 0, 1])
        metadata_template["dim_flip"].append([False, False, True])
    else:
        metadata_template["dim_order"].append([0, 1, 2])
        metadata_template["dim_flip"].append([False, False, False])

    metadata_template["key_age"].append(True)
    metadata_template["padding_micron"].append([[0, 0], [0, 0], [0, 0]])
    metadata_template["vector"].append(1)
    metadata_template["source_age_pnd"].append(56)
    metadata_template["target_age_pnd"].append(56)
    metadata_template["source_key_age"].append(False)
    metadata_template["target_key_age"].append(False)
    metadata_template["transformation_resolution_micron"].append(25)
    if target == "perens_stpt_mouse":
        metadata_template["target_X_physical_size_micron"].append(11400)
        metadata_template["target_Y_physical_size_micron"].append(16700)
        metadata_template["target_Z_physical_size_micron"].append(8000)
    if target == "perens_lsfm_mouse":
        metadata_template["target_X_physical_size_micron"].append(9225)
        metadata_template["target_Y_physical_size_micron"].append(12800)
        metadata_template["target_Z_physical_size_micron"].append(6700)
    if target == "perens_mri_mouse":
        metadata_template["target_X_physical_size_micron"].append(11375)
        metadata_template["target_Y_physical_size_micron"].append(15375)
        metadata_template["target_Z_physical_size_micron"].append(7425)
    if target == "allen_mouse":
        metadata_template["target_X_physical_size_micron"].append(13200)
        metadata_template["target_Y_physical_size_micron"].append(8000)
        metadata_template["target_Z_physical_size_micron"].append(11400)

    if source == "perens_stpt_mouse":
        metadata_template["X_physical_size_micron"].append(11400)
        metadata_template["Y_physical_size_micron"].append(16700)
        metadata_template["Z_physical_size_micron"].append(8000)
    if source == "perens_lsfm_mouse":
        metadata_template["X_physical_size_micron"].append(9225)
        metadata_template["Y_physical_size_micron"].append(12800)
        metadata_template["Z_physical_size_micron"].append(6700)
    if source == "perens_mri_mouse":
        metadata_template["X_physical_size_micron"].append(11375)
        metadata_template["Y_physical_size_micron"].append(15375)
        metadata_template["Z_physical_size_micron"].append(7425)
    if source == "allen_mouse":
        metadata_template["X_physical_size_micron"].append(13200)
        metadata_template["Y_physical_size_micron"].append(8000)
        metadata_template["Z_physical_size_micron"].append(11400)


# {k:len(v) for k,v in metadata_template.items()}

metadata = pd.read_csv(metadata_path)
append_metadata = pd.DataFrame(metadata_template)
pd.concat([metadata, append_metadata]).to_csv(metadata_path, index=False)
