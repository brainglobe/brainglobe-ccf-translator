import brainglobe_ccf_translator as ccft
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas


volume = BrainGlobeAtlas('demba_allen_seg_dev_mouse_p28_20um').reference
# Create a ccft object
# this can be done for either volumes or points
# for volumes we would run the following
ccft_vol = ccft.Volume(
    values=volume, space="demba_dev_mouse", voxel_size_micron=20, age_PND=28
)
# we can then translate either the points or volumes into a new target age or space.
# ccft will try to find a path from the current space into the target one
ccft_vol.transform(target_age=40, target_space="demba_dev_mouse")
ccft_vol.save("../demo_data/transform_to_40.nii.gz")


# alternatively for points we could do this
points = points = np.array([(286,250,267),
                            (414,247,452),
                            (100,200,100)])


ccft_pts = ccft.PointSet(points, space="CCFv3", voxel_size_micron=20, age_PND=28)
# the API is the same for points
ccft_pts.transform(target_age=56)

# To check the current age of any ccft object run
print(ccft_vol.current_age)  # -> P56
# To check the original age of any object run
print(ccft_vol.original_age)  # -> P31
