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
import nibabel as nib

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
demba_young_template = BrainGlobeAtlas(f'demba_allen_seg_dev_mouse_p{target_age}_20um').reference

#and that's it. to plot your data see below


slice = 140
young_slice = 150

# Calculate extents based on voxel sizes (in microns)
adult_voxel_size = 25
young_voxel_size = 20

# Get image dimensions for the slices
adult_shape = demba_adult_template[:,slice].shape
young_shape = demba_young_template[:,young_slice].shape

# Calculate physical extents [left, right, bottom, top]
adult_extent = [0, adult_shape[1] * adult_voxel_size, adult_shape[0] * adult_voxel_size, 0]
young_extent = [0, young_shape[1] * young_voxel_size, young_shape[0] * young_voxel_size, 0]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.5)  # Increase the width space between the subplots

# Plot the adult images
axes[0].imshow(demba_adult_template[slice], cmap="gray", extent=adult_extent)
axes[0].imshow(P56_projection[slice], alpha=0.7, extent=adult_extent)
axes[0].set_title(f"Post natal day {source_age}")
axes[0].axis("off")  # Remove axes and ticks

# Plot the young images
axes[1].imshow(demba_young_template[young_slice], cmap="gray", extent=young_extent)
axes[1].imshow(young_projection[young_slice], alpha=0.7, extent=young_extent)
axes[1].set_title(f"Post natal day {target_age}")
axes[1].axis("off")  # Remove axes and ticks

# Add text and an arrow between the plots
fig.text(0.5, 0.55, "brainglobe_ccf_translator", ha="center", fontsize=12)

# Add an arrow between the plots
arrow = patches.FancyArrowPatch(
    (0.45, 0.5),
    (0.55, 0.5),
    transform=fig.transFigure,
    arrowstyle="->",
    mutation_scale=20,
    color="black",
)
fig.add_artist(arrow)  # Add the arrow to the figure

# Save the figure as an image file
fig.savefig("../media/allen_connectivity_transform.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
