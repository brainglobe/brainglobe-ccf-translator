"""
Generating the full metadata for developmental ages is too much to do by hand. 
This will auto generate it assuming you only transform using deformation fields. 
"""

import numpy as np
import pandas as pd
import nrrd

space_name = "demba_dev_mouse"


import nibabel as nib

img = nib.load(r"/home/harryc/github/ccf_translator_local/demo_data/DeMBA_P56_double.nii.gz")
demba_dev_mouse_size_micron = np.array(img.shape) * 20
img, header = nrrd.read(r"/home/harryc/github/ccf_translator_local/demo_data//annotation_10.nrrd")
allen_size_micron = np.array(img.shape) * 10
key_ages = np.array([4, 7, 14, 21, 28, 56])
dim_flip = [False, False, False]
padding_micron = [[0, 0], [0, 0], [0, 0]]
file_name_template = "{}_pull_{}.nii.gz"

metadata_template = {
    "file_name": [],
    "source_space": [],
    "target_space": [],
    "affine": [],
    "dim_order": [],
    "key_age": [],
    "source_age_pnd": [],
    "target_age_pnd": [],
    "source_key_age": [],
    "target_key_age": [],
    "padding_micron": [],
    "transformation_resolution_micron": [],
    "X_physical_size_micron": [],
    "Y_physical_size_micron": [],
    "Z_physical_size_micron": [],
    "target_X_physical_size_micron": [],
    "target_Y_physical_size_micron": [],
    "target_Z_physical_size_micron": [],
    "dim_flip": [],
    "vector": [],
}


def update_metadata(
    file_name,
    source_space,
    target_space,
    affine,
    dim_order,
    key_age,
    source_age_pnd,
    target_age_pnd,
    source_key_age,
    target_key_age,
    vector,
    padding_micron,
    transformation_resolution_micron,
    X_physical_size_micron,
    Y_physical_size_micron,
    Z_physical_size_micron,
    target_X_physical_size_micron,
    target_Y_physical_size_micron,
    target_Z_physical_size_micron,
    dim_flip,
    metadata_template,
):
    metadata_template["file_name"].append(file_name)
    metadata_template["source_space"].append(source_space)
    metadata_template["target_space"].append(target_space)
    metadata_template["affine"].append(affine)
    metadata_template["dim_order"].append(dim_order)
    metadata_template["key_age"].append(key_age)
    metadata_template["source_age_pnd"].append(source_age_pnd)
    metadata_template["target_age_pnd"].append(target_age_pnd)
    metadata_template["source_key_age"].append(source_key_age)
    metadata_template["target_key_age"].append(target_key_age)
    metadata_template["padding_micron"].append(padding_micron)
    metadata_template["transformation_resolution_micron"].append(
        transformation_resolution_micron
    )
    metadata_template["X_physical_size_micron"].append(X_physical_size_micron)
    metadata_template["Y_physical_size_micron"].append(Y_physical_size_micron)
    metadata_template["Z_physical_size_micron"].append(Z_physical_size_micron)
    metadata_template["target_X_physical_size_micron"].append(
        target_X_physical_size_micron
    )
    metadata_template["target_Y_physical_size_micron"].append(
        target_Y_physical_size_micron
    )
    metadata_template["target_Z_physical_size_micron"].append(
        target_Z_physical_size_micron
    )
    metadata_template["dim_flip"].append(dim_flip)
    metadata_template["vector"].append(vector)
    return metadata_template


update_count = 0
for age in range(4, 57):
    key_age = age in key_ages
    if age in key_ages[1:]:
        target_key = np.max(key_ages[key_ages < age])
        file_name = file_name_template.format(age, age - 1)
        for target_age in range(target_key, age + 1):
            if target_age == age:
                continue
            vector = age - target_age
            metadata_template = update_metadata(
                file_name=file_name,
                source_space=space_name,
                target_space=space_name,
                affine=np.eye(4).tolist(),
                dim_order=[0, 1, 2],
                key_age=key_age,
                source_age_pnd=age,
                target_age_pnd=target_age,
                source_key_age=age,
                target_key_age=target_key,
                padding_micron=padding_micron,
                vector=vector,
                transformation_resolution_micron=20,
                dim_flip=dim_flip,
                X_physical_size_micron=demba_dev_mouse_size_micron[0],
                Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                target_X_physical_size_micron=demba_dev_mouse_size_micron[0],
                target_Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                target_Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                metadata_template=metadata_template,
            )
            update_count += 1
    if age in key_ages[:-1]:
        target_key = np.min(key_ages[key_ages > age])
        file_name = file_name_template.format(age, age + 1)

        for target_age in range(age, target_key + 1):
            if target_age == age:
                continue
            vector = target_age - age
            metadata_template = update_metadata(
                file_name=file_name,
                source_space=space_name,
                target_space=space_name,
                affine=np.eye(4).tolist(),
                dim_order=[0, 1, 2],
                key_age=key_age,
                source_age_pnd=age,
                target_age_pnd=target_age,
                source_key_age=age,
                target_key_age=target_key,
                padding_micron=padding_micron,
                vector=vector,
                transformation_resolution_micron=20,
                X_physical_size_micron=demba_dev_mouse_size_micron[0],
                dim_flip=dim_flip,
                Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                target_X_physical_size_micron=demba_dev_mouse_size_micron[0],
                target_Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                target_Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                metadata_template=metadata_template,
            )
            update_count += 1

    if not key_age:
        source_key = np.min(key_ages[key_ages > age])
        file_name = file_name_template.format(age, age - 1)
        target_key = np.max(key_ages[key_ages < age])
        for target_age in range(target_key, source_key + 1):
            if target_age == age:
                continue

            vector = age - target_age
            metadata_template = update_metadata(
                file_name=file_name,
                source_space=space_name,
                target_space=space_name,
                affine=np.eye(4).tolist(),
                dim_order=[0, 1, 2],
                key_age=key_age,
                source_age_pnd=age,
                target_age_pnd=target_age,
                source_key_age=source_key,
                target_key_age=target_key,
                padding_micron=padding_micron,
                vector=vector,
                transformation_resolution_micron=20,
                dim_flip=dim_flip,
                X_physical_size_micron=demba_dev_mouse_size_micron[0],
                Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                target_X_physical_size_micron=demba_dev_mouse_size_micron[0],
                target_Y_physical_size_micron=demba_dev_mouse_size_micron[1],
                target_Z_physical_size_micron=demba_dev_mouse_size_micron[2],
                metadata_template=metadata_template,
            )
            update_count += 1

"""
Add conversion to Allen Space
"""
file_name = False
source_space = "demba_dev_mouse"
target_space = "allen_mouse"
dim_order = [2, 1, 0]
dim_flip = [False, False, False]
age = 56
padding_micron = np.array([[0, 900], [0, 0], [0, 0]])


metadata_template = update_metadata(
    file_name=False,
    source_space=source_space,
    target_space=target_space,
    affine=np.eye(4).tolist(),
    dim_order=dim_order,
    key_age=True,
    source_age_pnd=age,
    target_age_pnd=age,
    source_key_age=False,
    target_key_age=False,
    vector=False,
    transformation_resolution_micron=False,
    padding_micron=padding_micron.tolist(),
    X_physical_size_micron=demba_dev_mouse_size_micron[0],
    Y_physical_size_micron=demba_dev_mouse_size_micron[1],
    Z_physical_size_micron=demba_dev_mouse_size_micron[2],
    target_X_physical_size_micron=allen_size_micron[0],
    target_Y_physical_size_micron=allen_size_micron[1],
    target_Z_physical_size_micron=allen_size_micron[2],
    dim_flip=dim_flip,
    metadata_template=metadata_template,
)


def invert_dim_order(order):
    inverse = [0] * len(order)
    for i, o in enumerate(order):
        inverse[o] = i
    return inverse


padding_micron = padding_micron[dim_order]
padding_micron *= -1
dim_flip = np.array(dim_flip)[dim_order].tolist()
dim_order = invert_dim_order(dim_order)
metadata_template = update_metadata(
    file_name=False,
    source_space=target_space,
    target_space=source_space,
    affine=np.eye(4).tolist(),
    dim_order=dim_order,
    key_age=True,
    source_age_pnd=age,
    target_age_pnd=age,
    source_key_age=False,
    target_key_age=False,
    vector=1,
    transformation_resolution_micron=False,
    padding_micron=padding_micron.tolist(),
    X_physical_size_micron=allen_size_micron[0],
    Y_physical_size_micron=allen_size_micron[1],
    Z_physical_size_micron=allen_size_micron[2],
    target_X_physical_size_micron=demba_dev_mouse_size_micron[0],
    target_Y_physical_size_micron=demba_dev_mouse_size_micron[1],
    target_Z_physical_size_micron=demba_dev_mouse_size_micron[2],
    dim_flip=dim_flip,
    metadata_template=metadata_template,
)


pd.DataFrame(metadata_template).to_csv(
    "/home/harryc/github/ccf_translator/ccf_translator/metadata/translation_metadata.csv",
    index=False,
)
