import CCF_translator
import nibabel as nib
import numpy as np

image_path = r"demo_data/DeMBA_P56_brain.nii.gz"
img = nib.load(image_path) 
data = np.asanyarray(img.dataobj)


CCFT_vol = CCF_translator.volume(
    values = data,
    space = 'Demba',
    voxel_size_um=20,
    segmentation_file=False,
    age_PND = 56
)


CCFT_vol.transform_to_age(30)


"""TODO 
transform points
"""
"""
if you wanted to get the voxel size from the header you could do so like this
voxel_size = img.header['pixdim']
unit = img.header['xyzt_units'][1]
if unit==1:
    voxel_size *= 1e6
if unit==2:
    voxel_size *= 1e3
"""