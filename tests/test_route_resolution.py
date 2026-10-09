import numpy as np
import pandas as pd
import pytest

from brainglobe_ccf_translator import Volume, VolumeSeries
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


@pytest.fixture
def nonlinear_route(constant_route, monkeypatch):
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
    return path, metadata, first, second


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


@pytest.mark.parametrize("downsample", [None, True, False])
@pytest.mark.parametrize("segmentation", [False, True])
def test_volume_native_composition_option(
    nonlinear_route, downsample, segmentation
):
    path, metadata, first, second = nonlinear_route
    values = np.indices((10, 10, 10)).sum(axis=0)
    values = values.astype(np.uint16 if segmentation else np.float64)
    volume = Volume(values, "allen_mouse", 4, 1, segmentation)
    volume.metadata = metadata
    volume.deformation_dir = path

    if downsample is False:
        # Original behavior: compose on the first field's native 2 µm grid,
        # then resample the composition onto the volume's 4 µm grid.
        deformation = apply_deformation.combine_deformations(
            first,
            apply_deformation.resize_transform(second, (1, 1.5, 1.5, 1.5)),
        )
        deformation = apply_deformation.resize_transform(
            deformation, (1, 0.5, 0.5, 0.5)
        )
    else:
        deformation = apply_deformation.combine_deformations(
            apply_deformation.resize_transform(first, (1, 0.5, 0.5, 0.5)),
            apply_deformation.resize_transform(second, (1, 0.75, 0.75, 0.75)),
        )
    expected = apply_deformation.apply_transform(
        values, deformation, order=0 if segmentation else 1
    )

    options = {} if downsample is None else {"downsample": downsample}
    volume.transform(3, "allen_mouse", **options)

    np.testing.assert_array_equal(volume.values, expected)
    assert volume.values.shape == values.shape
    assert volume.values.dtype == values.dtype
    assert volume.voxel_size_micron == 4
    assert volume.age_PND == 3
    assert volume.space == "allen_mouse"
    if not segmentation:
        # Downsampling first gives a coarse-grid neighbor of 0.022777...
        # sampled at a shift of 0.225, yielding 0.225 + 0.005125.
        expected_origin = 0.22725 if downsample is False else 0.230125
        np.testing.assert_allclose(volume.values[0, 0, 0], expected_origin)


@pytest.mark.parametrize("downsample", [None, True, False])
def test_series_forwards_composition_option(
    constant_route, monkeypatch, downsample
):
    _, metadata = constant_route
    volumes = [
        Volume(np.full((2, 2, 2), age - 1.0), "allen_mouse", 4, age)
        for age in (1, 3)
    ]
    series = VolumeSeries(volumes)
    series.metadata = metadata
    monkeypatch.setattr(
        series,
        "calculate_hamiltonian",
        lambda: [f"allen_mouse_P{age}" for age in (1, 2, 3)],
    )
    calls = []

    def transform(volume, target_age, target_space, *, downsample=True):
        calls.append(downsample)
        volume.values += 1 if downsample else 2
        volume.age_PND = target_age
        volume.space = target_space

    monkeypatch.setattr(Volume, "transform", transform)
    options = {} if downsample is None else {"downsample": downsample}
    series.interpolate_series(**options)

    selected = downsample is not False
    assert calls == [selected, selected]
    interpolated = series.find_volume_by_age_and_space(2, "allen_mouse")
    np.testing.assert_array_equal(interpolated.values, 2 if selected else 3)
    np.testing.assert_array_equal(volumes[0].values, 0)
    np.testing.assert_array_equal(volumes[1].values, 2)
    assert interpolated.voxel_size_micron == 4
