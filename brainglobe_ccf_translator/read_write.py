import numpy as np
import nibabel as nib
import ast
import brainglobe_ccf_translator.Volume as Volume


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
    byte_string = img.header["descrip"]
    try:
        # Decode the byte string to a regular string
        string_representation = byte_string.decode("utf-8")
        # Convert the string to a dictionary
        dictionary = ast.literal_eval(string_representation)
        data = np.asanyarray(img.dataobj)
        ccft_vol = Volume(
            data=data,
            space=dictionary["space"],
            voxel_size_micron=img.affine[0],
            age_PND=dictionary["age_PND"],
            segmentation_file=dictionary["segmentation_file"],
        )
    except:
        raise (
            "Failed to open volume. This function only works with volumes that were saved using ccft translator."
        )
    return ccft_vol
