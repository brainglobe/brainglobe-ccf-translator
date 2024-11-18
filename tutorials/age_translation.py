import os

# os.chdir("..")
import brainglobe_ccf_translator as ccft
import nibabel as nib
import numpy as np

# Load the image
image = nib.load("demo_data/DeMBA_P28.nii.gz")
volume = image.get_fdata()
# Create a ccft object
# this can be done for either volumes or points
# for volumes we would run the following
ccft_vol = ccft.Volume(
    values=volume, space="demba_dev_mouse", voxel_size_micron=20, age_PND=28
)
# we can then translate either the points or volumes into a new target age or space.
# ccft will try to find a path from the current space into the target one
ccft_vol.transform(target_age=40, target_space="demba_dev_mouse")
ccft_vol.save("demo_data/transform_to_40.nii.gz")


# alternatively for points we could do this
with open("my_points.txt") as f:
    points = np.array(f.readlines())
ccft_pts = ccft.PointSet(points, space="CCFv3", voxel_size_micron=20, age_PND=28)
# the API is the same for points
ccft_pts.transform(target_age=56)

# To check the current age of any ccft object run
print(ccft_vol.current_age)  # -> P56
# To check the original age of any object run
print(ccft_vol.original_age)  # -> P31
