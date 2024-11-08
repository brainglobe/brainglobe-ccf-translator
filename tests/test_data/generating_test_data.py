import numpy as np
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from scipy.ndimage import zoom
import os
import nibabel as nib
# test_data_dir = os.path.dirname(__file__)
test_data_dir = r"/home/harryc/github/ccf_translator/tests/test_data/"


test_atlas = BrainGlobeAtlas("allen_mouse_100um")
reference = zoom(test_atlas.reference, 0.5)
annotation = zoom(test_atlas.annotation, 0.5, order=0)
# Save the arrays using numpy with compression
np.savez_compressed(os.path.join(test_data_dir,'volumes', 'allen_mouse_200um'), reference=reference, annotation=annotation)


test_atlas = BrainGlobeAtlas("princeton_mouse_20um")
reference = zoom(test_atlas.reference, 0.1)
annotation = zoom(test_atlas.annotation, 0.1, order=0)
# Save the arrays using numpy with compression
np.savez_compressed(os.path.join(test_data_dir,'volumes', 'princeton_mouse_200um'), reference=reference, annotation=annotation)


#at present the latest gubra and demba atlases are not in brainglobe so we will load them manually
gubra_reference_mri = r"/home/harryc/github/gubra/Multimodal_mouse_brain_atlas_files/MRI_space_oriented/mri_new_header.nii.gz"
gubra_reference_mri_arr = nib.load(gubra_reference_mri).get_fdata()
reference = zoom(gubra_reference_mri_arr, 0.1)
np.savez_compressed(os.path.join(test_data_dir,'volumes', 'gubra_mri_mouse_200um'), reference=reference)

demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P5.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
demba_annotation = r"/home/harryc/github/ccf_translator_local/demo_data/demba_20um/P5_allen_2022_annotation_20um.nii.gz"
demba_annotation_arr = nib.load(demba_annotation).get_fdata()
reference = zoom(demba_reference_arr, 0.1)
annotation = zoom(demba_annotation_arr, 0.1, order=0)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P5_mouse_200um'), reference=reference, annotation=annotation)


demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P4.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P4_mouse_200um'), reference=reference)


demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P7.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P7_mouse_200um'), reference=reference)


demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P8.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P8_mouse_200um'), reference=reference)


demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P56.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P56_mouse_200um'), reference=reference)


demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P28.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P28_mouse_200um'), reference=reference)

demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P14.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P14_mouse_200um'), reference=reference)

demba_reference = r"/home/harryc/github/ccf_translator_local/demo_data/demba_vols/DeMBA_P4.nii.gz"
demba_reference_arr = nib.load(demba_reference).get_fdata()
reference = zoom(demba_reference_arr, 0.1)

np.savez_compressed(os.path.join(test_data_dir,'volumes', 'demba_P4_mouse_200um'), reference=reference)

