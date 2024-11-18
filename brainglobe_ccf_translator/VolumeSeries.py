from .deformation.route_calculation import find_hamiltonian_path, calculate_route,  find_path_through_nodes, create_G
import os
import pandas as pd
import nibabel as nib
import copy
import numpy as np
from .Volume import Volume
from pathlib import Path

base_path = os.path.dirname(__file__)

class VolumeSeries:
    def __init__(self, Volumes):
        self.Volumes = Volumes
        metadata_path = os.path.join(base_path, "metadata", "translation_metadata.csv")
        metadata = pd.read_csv(metadata_path)
        self.metadata = metadata

    def calculate_hamiltonian(self):
        terminals = {f"{i.space}_P{i.age_PND}" for i in self.Volumes}
        # metadata_vec_1 = self.metadata[self.metadata['vector'].astype(int).abs() == 1]
        # #Filter the graph to only include the nodes we are interested in
        # G = create_G(metadata_vec_1)
        # subgraph = list(find_minimal_subgraph(G, terminals))
        # space_and_ages_set = set([tuple(i.split('_P')) for i in subgraph])
        # source_mask = self.metadata.apply(lambda row: (row['source_space'], str(row['source_age_pnd'])) in space_and_ages_set, axis=1)
        # target_mask = self.metadata.apply(lambda row: (row['target_space'], str(row['target_age_pnd'])) in space_and_ages_set, axis=1)
        # filtered_metadata = self.metadata[source_mask & target_mask]
        G = create_G(self.metadata)
        #find route
        route = find_path_through_nodes(G, terminals)
        return route

    def split_volume_name(self, volume_name):
        parts = volume_name.split('_P')
        age = int(parts[-1])
        space = '_'.join(parts[:-1])
        return age, space

    def find_volume_by_age_and_space(self, age, space):
        return next((vol for vol in self.Volumes if vol.age_PND == age and vol.space == space), None)

    def filter_metadata(self):
        mask = np.abs(self.metadata['vector']) == 1
        return self.metadata[mask]

    def interpolate_series(self):
        route = self.calculate_hamiltonian()
        existing_route = [i for i in route if i in [f"{i.space}_P{i.age_PND}" for i in self.Volumes]]
        if not route:
            print("No valid route was found. Exiting.")
            return

        filtered_metadata = self.filter_metadata()

        for i in range(len(existing_route) - 1):
            start = existing_route[i]
            end = existing_route[i + 1]
            start_age, start_space = self.split_volume_name(start)
            end_age, end_space = self.split_volume_name(end)

            left_volume = self.find_volume_by_age_and_space(start_age, start_space)
            right_volume = self.find_volume_by_age_and_space(end_age, end_space)

            if left_volume is None or right_volume is None:
                print(f"Volume not found for start {start} or end {end}")
                continue
            G = create_G(filtered_metadata)
            sub_route = calculate_route(start, end, G)
            left_pos = sub_route.index(start)
            right_pos = sub_route.index(end)

            for target in sub_route[1:-1]:
                target_age, target_space = self.split_volume_name(target)
                left_volume_temp = copy.deepcopy(left_volume)
                right_volume_temp = copy.deepcopy(right_volume)

                left_volume_temp.transform(target_age, target_space)
                right_volume_temp.transform(target_age, target_space)

                target_pos = sub_route.index(target)
                right_factor = (left_pos - target_pos) / (left_pos - right_pos)
                left_factor = 1 - right_factor

                left_volume_temp.values *= left_factor
                right_volume_temp.values *= right_factor
                new_values = left_volume_temp.values + right_volume_temp.values

                target_volume = Volume(
                    values=new_values,
                    space=target_space,
                    voxel_size_micron=left_volume_temp.voxel_size_micron,
                    age_PND=target_age,
                    segmentation_file=left_volume_temp.segmentation_file
                )
                self.Volumes.append(target_volume)
                print(f"appended P{target_age}")
                print(len(self.Volumes))
    def save(self, output_dir):
        if not output_dir:
            raise ValueError("Output directory cannot be an empty string")
        output_path = Path(output_dir)
        for V in self.Volumes:
            filename = f"{V.space}_P{V.age_PND}_{'segmentation_' if V.segmentation_file else ''}{V.voxel_size_micron}micron.nii.gz"
            V.save(output_path / filename)