import ast

import nibabel as nib
import numpy as np

from .Volume import Volume


def save_volume(ccft_vol, save_path):
    vol_metadata = {
        "space": ccft_vol.space,
        "age_PND": ccft_vol.age_PND,
        "segmentation_file": ccft_vol.segmentation_file,
    }
    affine = np.eye(4)
    affine[:3, :3] *= ccft_vol.voxel_size_micron
    image = nib.Nifti1Image(ccft_vol.values, affine=affine)
    image.header["descrip"] = vol_metadata
    image.header.set_xyzt_units(3)
    nib.save(image, save_path)


def read_volume(path):
    img = nib.load(path)
    try:
        string_representation = (
            bytes(img.header["descrip"]).decode("utf-8").rstrip("\x00")
        )
        # Convert the string to a dictionary
        dictionary = ast.literal_eval(string_representation)
        if not isinstance(dictionary, dict):
            raise TypeError("Volume metadata is not a dictionary.")
        data = np.asanyarray(img.dataobj)
        ccft_vol = Volume(
            values=data,
            space=dictionary["space"],
            voxel_size_micron=img.header.get_zooms()[0],
            age_PND=dictionary["age_PND"],
            segmentation_file=dictionary["segmentation_file"],
        )
    except (
        SyntaxError,
        TypeError,
        ValueError,
        KeyError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            "Failed to open volume. This function only works with volumes that were saved using ccft translator."
        ) from exc
    return ccft_vol
