from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import brainglobe_ccf_translator
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
import numpy as np
import nibabel as nib

voxel_size_micron = 25
space_name = r"perens_stereotaxic_mri_mouse"
atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")

path = "/home/harryc/brainglobe_workingdir/perens_stereotaxic_mri_mouse/downloads/atlas_files/Multimodal_mouse_brain_atlas_files/MRI_space_oriented/mri_temp.nii.gz"
og_vol = nib.load(path).get_fdata()

og_vol = og_vol.transpose(1, 2, 0)
og_vol = og_vol[:, ::-1, :]


ccft_vol = brainglobe_ccf_translator.Volume(
    values=atlas.reference,
    space="perens_stereotaxic_mri_mouse",
    voxel_size_micron=voxel_size_micron,
    age_PND=56,
)


ccft_vol.transform(56, "perens_stpt_mouse")


import matplotlib.pyplot as plt


path = "/home/harryc/brainglobe_workingdir/perens_stereotaxic_mri_mouse/downloads/atlas_files/Multimodal_mouse_brain_atlas_files/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz"
ref = nib.load(path).get_fdata()

ref = ref.transpose(1, 2, 0)
ref = ref[:, ::-1, :]

plt.imshow(ref[:, :, 200] + ccft_vol.values[:, :, 200], cmap="gray")
plt.imshow(ref[:, 200], cmap="gray")
plt.imshow(ccft_vol.values[:, 200], cmap="gray")
current_orientation = "lai"

import nibabel as nib

in_path = r"/home/harryc/github/brainglobe-ccf-translator/brainglobe_ccf_translator/metadata/deformation_fields/perens_stpt_mouse/perens_stpt_mouse_pull_perens_mri_mouse.nii.gz"
out_path = r"/home/harryc/.brainglobe/deformation_fields/perens_stpt_mouse/perens_stpt_mouse_pull_perens_mri_mouse_v2.nii.gz"

# Load image and work on raw values (header/orientation may be unreliable)
img = nib.load(in_path)
data = img.get_fdata()
data = np.transpose(data, [1, 2, 0, 3])

data = data[:, :, :, [1, 2, 0]]

data = data[:, ::-1, :, :]
data[:, :, :, 1] *= -1

new_img = nib.Nifti1Image(data, np.eye(4))
nib.save(new_img, out_path)


import pandas as pd

df = pd.read_csv(
    "https://github.com/brainglobe/brainglobe-ccf-translator/raw/refs/heads/main/brainglobe_ccf_translator/metadata/translation_metadata.csv"
)
df[df["dim_order"] != "[0, 1, 2]"]

df["dim_order"][0]


#####Showing how it worked previously

import nibabel as nib
import brainglobe_ccf_translator
import matplotlib.pyplot as plt

voxel_size_micron = 25

path = "/home/harryc/brainglobe_workingdir/perens_stereotaxic_mri_mouse/downloads/atlas_files/Multimodal_mouse_brain_atlas_files/MRI_space_oriented/mri_temp.nii.gz"
og_vol = nib.load(path).get_fdata().transpose(1, 2, 0)[:, ::-1, :]
ccft_vol = brainglobe_ccf_translator.Volume(
    values=og_vol,
    space="perens_mri_mouse",
    voxel_size_micron=voxel_size_micron,
    age_PND=56,
)
ccft_vol.transform(56, "perens_stpt_mouse")


path = "/home/harryc/brainglobe_workingdir/perens_stereotaxic_mri_mouse/downloads/atlas_files/Multimodal_mouse_brain_atlas_files/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz"
ref = nib.load(path).get_fdata().transpose(1, 2, 0)[:, ::-1, :]
plt.imshow(ref[:, :, 200] + ccft_vol.values[:, :, 200], cmap="gray")
plt.imshow(ref[200], cmap="gray")
plt.imshow(ccft_vol.values[200], cmap="gray")
