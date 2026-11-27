"""Append the swc_female_rat <-> whs_sd_rat rows to the translation metadata.

Fields are produced by ``ingesting_new_deformations/swc_female_rat.py``.
whs_sd_rat is P80, the age of the WHS specimen (Papp et al. 2014).
swc_female_rat (built from 7-15 week old animals) uses the same age as a
placeholder until CCF translator supports spaces without a single age.
"""

import numpy as np
import pandas as pd

VERSION = "1.0"
metadata_path = (
    r"../../brainglobe_ccf_translator/metadata/translation_metadata.csv"
)

AGE = 80
# Grid shape (ASR) and the voxel size each field is stored at.
spaces = {
    "swc_female_rat": {"shape": [1030, 500, 660], "resolution": 25},
    "whs_sd_rat": {"shape": [1024, 512, 512], "resolution": 39},
}
pairs = [("swc_female_rat", "whs_sd_rat"), ("whs_sd_rat", "swc_female_rat")]

rows = []
for source, target in pairs:
    source_size = (
        np.array(spaces[source]["shape"]) * (spaces[source]["resolution"])
    )
    target_size = (
        np.array(spaces[target]["shape"]) * (spaces[target]["resolution"])
    )
    rows.append(
        {
            "file_name": f"{source}_pull_{target}_v{VERSION}.nii.gz",
            "source_space": source,
            "target_space": target,
            "affine": np.eye(4).tolist(),
            "dim_order": [0, 1, 2],
            "key_age": True,
            "source_age_pnd": AGE,
            "target_age_pnd": AGE,
            "source_key_age": False,
            "target_key_age": False,
            "padding_micron": [[0, 0], [0, 0], [0, 0]],
            "transformation_resolution_micron": spaces[source]["resolution"],
            "X_physical_size_micron": source_size[0],
            "Y_physical_size_micron": source_size[1],
            "Z_physical_size_micron": source_size[2],
            "target_X_physical_size_micron": target_size[0],
            "target_Y_physical_size_micron": target_size[1],
            "target_Z_physical_size_micron": target_size[2],
            "dim_flip": [False, False, False],
            "vector": 1,
        }
    )

metadata = pd.read_csv(metadata_path)
pd.concat([metadata, pd.DataFrame(rows)]).to_csv(metadata_path, index=False)
