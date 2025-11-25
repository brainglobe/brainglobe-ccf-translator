import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd

VERSION = "1.1"
metadata_path = Path("/home/harryc/github/brainglobe-ccf-translator/brainglobe_ccf_translator/metadata/translation_metadata.csv")
out_path = Path("~/.brainglobe/").expanduser()
def_out_dir = out_path / "deformation_fields"

def create_metadata_dict(
    source_space,
    target_space,
    source_age_pnd,
    target_age_pnd,
    transformation_resolution_micron,
    affine="[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]]",
    dim_order="[0, 1, 2]",
    key_age=True,
    source_key_age=False,
    target_key_age=False,
    padding_micron="[[0, 0], [0, 0], [0, 0]]",
    dim_flip="[False, False, False]",
    vector=1,
    out_dir = "",
    VERSION = ""
):
    import nibabel as nib
    import numpy as np
    import pandas as pd

    filename = f"{target_space}_pull_{source_space}_v{VERSION}.nii.gz"
    img = nib.load(f"{out_dir}/{target_space}/{filename}")
    X_physical_size_micron,Y_physical_size_micron,Z_physical_size_micron,_ = np.array(img.shape) * transformation_resolution_micron

    filename = f"{source_space}_pull_{target_space}_v{VERSION}.nii.gz"
    img = nib.load(f"{out_dir}/{source_space}/{filename}")
    target_X_physical_size_micron,target_Y_physical_size_micron,target_Z_physical_size_micron,_ = np.array(img.shape) * transformation_resolution_micron
    return {
        "file_name": filename,
        "source_space": source_space,
        "target_space": target_space,
        "affine": affine,
        "dim_order": dim_order,
        "key_age": key_age,
        "source_age_pnd": source_age_pnd,
        "target_age_pnd": target_age_pnd,
        "source_key_age": source_key_age,
        "target_key_age": target_key_age,
        "padding_micron": padding_micron,
        "transformation_resolution_micron": transformation_resolution_micron,
        "X_physical_size_micron":X_physical_size_micron,
        "Y_physical_size_micron":Y_physical_size_micron,
        "Z_physical_size_micron":Z_physical_size_micron,
        "target_X_physical_size_micron":target_X_physical_size_micron,
        "target_Y_physical_size_micron":target_Y_physical_size_micron,
        "target_Z_physical_size_micron":target_Z_physical_size_micron,
        "dim_flip": dim_flip,
        "vector": vector,
    }

new_metadata = pd.DataFrame([
    create_metadata_dict(
        source_space = "perens_multimodal_lsfm",
        target_space = "allen_mouse",
        transformation_resolution_micron=25,
        source_age_pnd=56,
        target_age_pnd=56,
        out_dir=def_out_dir,
        VERSION = VERSION
        ),
    create_metadata_dict(
        source_space = "allen_mouse",
        target_space = "perens_multimodal_lsfm",
        transformation_resolution_micron=25,
        source_age_pnd=56,
        target_age_pnd=56,
        out_dir=def_out_dir,
        VERSION = VERSION

        ),
    create_metadata_dict(
        source_space = "perens_stereotaxic_mri_mouse",
        target_space = "allen_mouse",
        transformation_resolution_micron=25,
        source_age_pnd=56,
        target_age_pnd=56,
        out_dir=def_out_dir,
        VERSION = VERSION

        ),
    create_metadata_dict(
        source_space = "allen_mouse",
        target_space = "perens_stereotaxic_mri_mouse",
        transformation_resolution_micron=25,
        source_age_pnd=56,
        target_age_pnd=56,
        out_dir=def_out_dir,
        VERSION = VERSION
        ),
    ])

metadata = pd.read_csv(metadata_path)
pd.concat([metadata, new_metadata]).to_csv(metadata_path, index=False)
