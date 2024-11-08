import ccf_translator as CCFT
import nibabel as nib
import numpy as np

# CCFT is also able to interpolate time series data to create temporally continuous volumes.
my_data_paths = {
    4: "P4_image_path.nii",
    7: "P7_image_path.nii",
    31: "P31_image_path.nii",
    56: "P56_image_path.nii",
}
CCFT_vols = []
for age, path in my_data_paths.items():
    data = nib.load(path)
    vol = data.get_fdata()
    CCFT_vol = CCFT.Volume(data=vol, space="CCFv3", voxel_size=20, age_PND=age)
    CCFT_vols.append(CCFT_vol)
# Once you have a list of CCFT volumes a time series can then be created
CCFT_TS = CCFT.VolumeSeries(CCFT_vols)
# You can check which ages are present with the following
print(CCFT_TS.data_ages)  # -> P4, P7, P31, P56
# We can then interpolate the missing data using the following line
CCFT_TS.interpolate_ages(ages_to_interpolate="all")
# If we wanted to only interpolate specific ages we could pass a list of these to ages_to_interpolate.
# We now have a 4D volume which includes every age.
# To retrieve this volume as an array we can do the following
CCFT_TS_arr = CCFT_TS.array()
# The shape of this array is T, X, Y, Z
print(CCFT_TS_arr.shape)
# We can use nibabel to save this volume for viewing in ITKsnap
out_image = nib.Nifti1Image(CCFT_TS_arr, np.eye(4))
nib.save(out_image, "4D_file.nii.gz")
# If we would like the arrays transformed to the same space for analysis we can do this
# Note there is no need to first run interpolate ages.
# This can be run as soon as a time series is created.
CCFT_translated = CCFT_TS.translate_series(target_age=56, ignore_interpolated=True)
# Above we ignored the interpolated volumes.
# This is the default behaviour but can be changed by switching the value to False
# This returns a list of translated volumes
# If we want to see these as a list of arrays we can do the following
CCF_translated_arrays = [i.array() for i in CCFT_translated]
