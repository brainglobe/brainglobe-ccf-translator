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
ccft_vol.save(rf"../demo_data/perens_lsfm_from_princeton.nii.gz")


### To transform it in the opposite direction simply do the following.

voxel_size_micron = 25
space_name = r"allen_mouse"
atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")

ccft_vol = brainglobe_ccf_translator.Volume(
    values=atlas.reference,
    space=space_name,
    voxel_size_micron=voxel_size_micron,
    segmentation_file=True,
    age_PND=source_age,
)

ccft_vol.transform(target_age, "princeton_mouse")
ccft_vol.save(rf"../demo_data/princeton_from_allen_mouse.nii.gz")
ccft_vol.transform(target_age, "allen_mouse")
ccft_vol.save(rf"../demo_data/princeton_from_allen_mouse_back_to_allen.nii.gz")
ccft_vol.transform(target_age, "perens_multimodal_lsfm")
ccft_vol.save(rf"../demo_data/now_to_perens_lsfm.nii.gz")
