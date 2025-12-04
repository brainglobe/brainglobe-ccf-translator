from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

import brainglobe_ccf_translator

voxel_size_micron = 20
space_name = r"princeton_mouse"
atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")
source_age = 56
target_age = 56

ccft_vol = brainglobe_ccf_translator.Volume(
    values=atlas.reference,
    space="princeton_mouse",
    voxel_size_micron=voxel_size_micron,
    segmentation_file=False,
    age_PND=source_age,
)

ccft_vol.transform(target_age, "perens_multimodal_lsfm")
ccft_vol.save(r"../demo_data/perens_lsfm_from_princeton.nii.gz")


"""
You can then run subsequent transformations like the following.
"""
ccft_vol.transform(target_age, "allen_mouse")
ccft_vol.save(r"../demo_data/allen_from_princeton.nii.gz")
