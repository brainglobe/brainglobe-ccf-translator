import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from brainglobe_ccf_translator import Volume
from brainglobe_ccf_translator.read_write import read_volume, save_volume


class TestReadWrite(unittest.TestCase):
    def test_read_volume_round_trip(self):
        data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
        volume = Volume(
            values=data,
            space="allen_mouse",
            voxel_size_micron=25,
            age_PND=56,
            segmentation_file=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "volume.nii.gz"
            save_volume(volume, save_path)
            loaded_volume = read_volume(save_path)

        np.testing.assert_array_equal(loaded_volume.values, data)
        self.assertEqual(loaded_volume.space, "allen_mouse")
        self.assertEqual(loaded_volume.age_PND, 56)
        self.assertEqual(loaded_volume.voxel_size_micron, 25)
        self.assertFalse(loaded_volume.segmentation_file)

    def test_read_volume_raises_for_invalid_metadata(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2)), affine=np.eye(4))
        image.header["descrip"] = "not-a-dictionary"

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "invalid_volume.nii.gz"
            nib.save(image, save_path)

            with self.assertRaises(ValueError) as ctx:
                read_volume(save_path)

        self.assertIn("Failed to open volume", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
