import numpy as np
import pandas as pd
import pytest

from brainglobe_ccf_translator import Volume
from brainglobe_ccf_translator.deformation import apply_deformation


@pytest.fixture
def constant_route(tmp_path, monkeypatch):
    # Constant displacements have a known physical sum across resolutions.
    fields = {}
    rows = []
    for age, resolution, size in [(2, 2, 20), (3, 3, 14)]:
        field = np.empty((3, size, size, size), dtype=np.float32)
        field[:] = (resolution * np.array([0.1, 0.2, 0.3]))[
            :, None, None, None
        ]
        name = f"{age}.nii.gz"
        path = tmp_path / "allen_mouse" / name
        path.parent.mkdir(exist_ok=True)
        path.touch()
        fields[str(path)] = field
        rows.append(
            dict(
                source_space="allen_mouse",
                source_age_pnd=age,
                target_space="allen_mouse",
                target_age_pnd=age - 1,
                file_name=name,
                transformation_resolution_micron=resolution,
                vector="1",
                padding_micron="[[0, 0], [0, 0], [0, 0]]",
                dim_order="[0, 1, 2]",
                dim_flip="[False, False, False]",
            )
        )
    monkeypatch.setattr(
        apply_deformation,
        "open_transformation",
        lambda path: fields[str(path)].copy(),
    )
    return tmp_path, pd.DataFrame(rows)


@pytest.mark.parametrize("output_resolution", [None, 1, 2, 4])
def test_compose_fields_in_consistent_voxel_units(
    constant_route, output_resolution
):
    path, metadata = constant_route
    result, _, _, _, resolution = apply_deformation.combine_route(
        ["allen_mouse_P1", "allen_mouse_P2", "allen_mouse_P3"],
        4,
        path,
        metadata,
        output_voxel_size=output_resolution,
    )

    expected_resolution = max(2, output_resolution or 2)
    assert resolution == expected_resolution
    assert result.shape == (3, *([int(14 * 3 / resolution)] * 3))
    # At the origin both fields contribute. Convert the physical sum into
    # voxels at the composition resolution, not each file's native resolution.
    expected = (2**2 + 3**2) * np.array([0.1, 0.2, 0.3]) / resolution
    np.testing.assert_allclose(result[:, 0, 0, 0], expected, rtol=1e-7)


@pytest.mark.parametrize("segmentation", [False, True])
def test_volume_uses_coarse_composition(constant_route, segmentation):
    path, metadata = constant_route
    indices = np.indices((10, 10, 10))
    values = indices.sum(axis=0)
    values = values.astype(np.uint16 if segmentation else np.float64)
    volume = Volume(values, "allen_mouse", 4, 1, segmentation)
    volume.metadata = metadata
    volume.deformation_dir = path

    volume.transform(3, "allen_mouse")

    assert volume.values.shape == (10, 10, 10)
    assert volume.values.dtype == values.dtype
    assert volume.voxel_size_micron == 4
    assert volume.age_PND == 3
    assert volume.space == "allen_mouse"
    expected_origin = 2 if segmentation else 1.95
    np.testing.assert_allclose(
        volume.values[0, 0, 0], expected_origin, rtol=1e-7
    )


def test_single_field_matches_resampling_after_loading(constant_route):
    path, metadata = constant_route
    route = ["allen_mouse_P1", "allen_mouse_P2"]
    native, *_, resolution = apply_deformation.combine_route(
        route, 4, path, metadata
    )
    expected = apply_deformation.resize_transform(
        native, (1, *([resolution / 4] * 3))
    )

    actual, *_, resolution = apply_deformation.combine_route(
        route, 4, path, metadata, output_voxel_size=4
    )

    np.testing.assert_array_equal(actual, expected)
    assert resolution == 4


def test_nonlinear_fields_are_downsampled_before_composition(
    constant_route, monkeypatch
):
    path, metadata = constant_route
    first = np.zeros((3, 20, 20, 20))
    first[0] = (np.arange(20) ** 2 * 0.01)[:, None, None]
    second = np.zeros((3, 14, 14, 14))
    second[0] = 0.3
    monkeypatch.setattr(
        apply_deformation,
        "open_transformation",
        lambda name: (
            first if str(name).endswith("2.nii.gz") else second
        ).copy(),
    )
    expected = apply_deformation.combine_deformations(
        apply_deformation.resize_transform(first, (1, 0.5, 0.5, 0.5)),
        apply_deformation.resize_transform(second, (1, 0.75, 0.75, 0.75)),
    )

    actual, *_, resolution = apply_deformation.combine_route(
        ["allen_mouse_P1", "allen_mouse_P2", "allen_mouse_P3"],
        4,
        path,
        metadata,
        output_voxel_size=4,
    )

    np.testing.assert_array_equal(actual, expected)
    assert resolution == 4

    # The first coarse-grid neighbor samples the quadratic at 19/9 native
    # voxels. Its scaled displacement is 0.022777..., sampled at a shift of
    # 0.225 coarse voxels, giving 0.225 + 0.005125 at the origin.
    np.testing.assert_allclose(actual[:, 0, 0, 0], [0.230125, 0, 0])


def test_long_route_composes_only_coarse_fields(constant_route, monkeypatch):
    path, metadata = constant_route
    last = metadata.iloc[-1].copy()
    last.source_age_pnd = 4
    last.target_age_pnd = 3
    metadata = pd.concat([metadata, last.to_frame().T], ignore_index=True)
    shapes = []
    combine = apply_deformation.combine_deformations

    def record_composition(first, second):
        shapes.append((first.shape, second.shape))
        return combine(first, second)

    monkeypatch.setattr(
        apply_deformation, "combine_deformations", record_composition
    )
    result, *_, resolution = apply_deformation.combine_route(
        [f"allen_mouse_P{age}" for age in range(1, 5)],
        4,
        path,
        metadata,
        output_voxel_size=4,
    )

    assert resolution == 4
    assert result.shape == (3, 10, 10, 10)
    assert shapes == [((3, 10, 10, 10), (3, 10, 10, 10))] * 2
