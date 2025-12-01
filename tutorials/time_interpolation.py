import brainglobe_ccf_translator as ccft
import nibabel as nib
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

# ccft is also able to interpolate time series data to create temporally continuous volumes.
my_ages = [
    4,
    6,
    9
    ]
ccft_vols = []
for age in my_ages:
    vol = BrainGlobeAtlas(f'demba_allen_seg_dev_mouse_p{age}_20um').reference
    ccft_vol = ccft.Volume(values=vol, space="demba_dev_mouse", voxel_size_micron=20, age_PND=age)
    ccft_vols.append(ccft_vol)
# Once you have a list of ccft volumes a time series can then be created
ccft_ts = ccft.VolumeSeries(ccft_vols)
# You can check which ages are present with the following
print([i.age_PND for i in ccft_ts.Volumes])  # -> 4, 6, 9
# We can then interpolate the missing data using the following line
ccft_ts.interpolate_series()
#you can check which ages are present after the interpolation
print([i.age_PND for i in ccft_ts.Volumes])  # -> 4, 5, 6, 7, 8, 9
#To save this to a single 4D file for viewing you can go
ccft_ts.save("../demo_data/ages_4_to_9")
