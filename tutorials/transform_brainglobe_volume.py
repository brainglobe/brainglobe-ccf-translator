import os

# os.chdir("..")
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import ccf_translator


voxel_size_micron = 20
space_name = r"princeton_mouse"
atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")
source_age = 56
target_age = 56

CCFT_vol = ccf_translator.Volume(
    values=atlas.reference,
    space="princeton_mouse",
    voxel_size_micron=voxel_size_micron,
    segmentation_file=False,
    age_PND=source_age,
)
# CCFT_vol.transform(target_age, "allen_mouse")
# CCFT_vol.save(rf"demo_data/allen_mouse_from_princeton.nii.gz")

CCFT_vol.transform(target_age, "perens_lsfm_mouse")
CCFT_vol.save(rf"demo_data/perens_lsfm_from_princeton.nii.gz")
import os

# os.chdir("..")
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import ccf_translator


voxel_size_micron = 25
space_name = r"allen_mouse"
atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")
source_age = 56
target_age = 56

CCFT_vol = ccf_translator.Volume(
    values=atlas.reference,
    space=space_name,
    voxel_size_micron=voxel_size_micron,
    segmentation_file=True,
    age_PND=source_age,
)
# CCFT_vol.save(rf"demo_data/princeton_mouse_from_princeton.nii.gz")

CCFT_vol.transform(target_age, "princeton_mouse")
CCFT_vol.save(rf"demo_data/princeton_from_allen_mouse.nii.gz")
