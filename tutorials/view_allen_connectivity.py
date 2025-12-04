"""
This script will show you how to take data from the allen connectivity API
and view it in a different space. In this case we will look at projection information
transformed down to a P9 brain.

To run this script you will need the allensdk which is installable via pip
"""

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import brainglobe_ccf_translator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# tell the cache class what resolution (in microns) of data you want to download
mcc = MouseConnectivityCache(resolution=25)
# download the projection density volume for one of the experiments
pd = mcc.get_projection_density(479982715)
pd_vals = pd[0]

voxel_size_micron = 25
space_name = r"allen_mouse"
adult_atlas = BrainGlobeAtlas(f"{space_name}_{voxel_size_micron}um")

source_age = 56
target_age = 9

ccft_vol = brainglobe_ccf_translator.Volume(
    values=pd_vals,
    space=space_name,
    voxel_size_micron=voxel_size_micron,
    segmentation_file=False,
    age_PND=source_age,
)
# transform to demba space
ccft_vol.transform(target_space="demba_dev_mouse", target_age=source_age)
P56_projection = ccft_vol.values
# transform to young space
ccft_vol.transform(target_space="demba_dev_mouse", target_age=target_age)
young_projection = ccft_vol.values

adult_vol = brainglobe_ccf_translator.Volume(
    values=adult_atlas.reference,
    space=space_name,
    voxel_size_micron=voxel_size_micron,
    segmentation_file=False,
    age_PND=source_age,
)
# transform to demba space
adult_vol.transform(target_space="demba_dev_mouse", target_age=source_age)
demba_adult_template = adult_vol.values
demba_young_template = BrainGlobeAtlas(
    f"demba_allen_seg_dev_mouse_p{target_age}_20um"
).reference

ccft_vol.save("../demo_data/p9_tracing.nii.gz")
# and that's it. to plot your data see below


def get_slice(volume, idx, axis):
    """Get a 2D slice from a 3D volume along the specified axis."""
    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


slice_axis = 2  # 0=coronal, 1=horizontal, 2=sagittal
slice_idx = 140
young_slice_idx = 150

# Calculate extents based on voxel sizes (in microns)
adult_voxel_size = 25
young_voxel_size = 20

# Get 2D slices along the chosen axis
adult_slice = np.rot90(get_slice(demba_adult_template, slice_idx, slice_axis), k=-1)
adult_proj_slice = np.rot90(get_slice(P56_projection, slice_idx, slice_axis), k=-1)
# scale slice index as the images are different resolutions
young_slice = np.rot90(
    get_slice(
        demba_young_template,
        int(young_slice_idx * (adult_voxel_size / young_voxel_size)),
        slice_axis,
    ),
    k=-1,
)
young_proj_slice = np.rot90(
    get_slice(young_projection, young_slice_idx, slice_axis), k=-1
)

# Calculate physical extents [left, right, bottom, top]
adult_extent = [
    0,
    adult_slice.shape[1] * adult_voxel_size,
    adult_slice.shape[0] * adult_voxel_size,
    0,
]
young_extent = [
    0,
    young_slice.shape[1] * young_voxel_size,
    young_slice.shape[0] * young_voxel_size,
    0,
]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.55)

# Plot the adult images
axes[0].imshow(adult_slice, cmap="gray", extent=adult_extent)
axes[0].imshow(adult_proj_slice, alpha=0.7, extent=adult_extent)
axes[0].set_title(f"Post natal day {source_age}")
axes[0].axis("off")

# Plot the young images
axes[1].imshow(young_slice, cmap="gray", extent=young_extent)
axes[1].imshow(young_proj_slice, alpha=0.7, extent=young_extent)
axes[1].set_title(f"Post natal day {target_age}")
axes[1].axis("off")

# Add text and an arrow between the plots
fig.text(0.52, 0.58, "brainglobe_ccf_translator", ha="center", fontsize=12)

# Add an arrow between the plots
arrow = patches.FancyArrowPatch(
    (0.46, 0.51),
    (0.58, 0.51),
    transform=fig.transFigure,
    arrowstyle="->",
    mutation_scale=20,
    color="black",
)
fig.add_artist(arrow)

# Save the figure as an image file
fig.savefig("../media/allen_connectivity_transform.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
