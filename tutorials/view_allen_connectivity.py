"""
This script will show you how to take data from the allen connectivity API 
and view it in a different space. In this case we will look at projection information
transformed down to a P9 brain. 
"""

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import brainglobe_ccf_translator
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
from scipy.ndimage import zoom

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

CCFT_vol = brainglobe_ccf_translator.Volume(
    values=pd_vals,
    space=space_name,
    voxel_size_micron=voxel_size_micron,
    segmentation_file=False,
    age_PND=source_age,
)
# transform to demba space
CCFT_vol.transform(target_space="demba_dev_mouse", target_age=source_age)
P56_projection = CCFT_vol.values
# transform to young space
CCFT_vol.transform(target_space="demba_dev_mouse", target_age=target_age)
young_projection = CCFT_vol.values

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
demba_young_template = nib.load(
    rf"/home/harryc/github/brainglobe_ccf_translator_local/demo_data/demba_vols/DeMBA_P{target_age}.nii.gz"
).get_fdata()
# rescale to 25µm
rescaled_demba_young_template = zoom(
    demba_young_template, 20 / 25, order=1
)  # order=1 for bilinear interpolation


slice = 140
young_slice = 150

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.5)  # Increase the width space between the subplots

# Plot the adult images
axes[0].imshow(demba_adult_template[slice], cmap="gray")
axes[0].imshow(P56_projection[slice], alpha=0.7)
axes[0].set_title(f"Post natal day {source_age}")
axes[0].axis("off")  # Remove axes and ticks

# Plot the young images
axes[1].imshow(rescaled_demba_young_template[young_slice], cmap="gray")
axes[1].imshow(young_projection[young_slice], alpha=0.7)
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
fig.savefig("media/allen_connectivity_transform.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
