import os
import unittest
from pathlib import Path

import pandas as pd

from brainglobe_ccf_translator.space_utils import (
    normalise_space_name,
    validate_space_name,
)


class TestSpaceUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata_path = (
            Path(__file__).resolve().parents[1]
            / "brainglobe_ccf_translator"
            / "metadata"
            / "translation_metadata.csv"
        )
        cls.metadata = pd.read_csv(metadata_path)

    def test_aliases_resolve_to_allen(self):
        aliases = [
            "kim_mouse",
            "allen_mouse_bluebrain_barrels",
            "osten_mouse",
        ]
        for alias in aliases:
            self.assertEqual(normalise_space_name(alias), "allen_mouse")
            self.assertEqual(
                validate_space_name(alias, self.metadata), "allen_mouse"
            )

    def test_unknown_space_raises(self):
        with self.assertRaises(ValueError):
            validate_space_name("unknown_mouse", self.metadata)

    def test_near_miss_suggests_synonym(self):
        with self.assertRaises(ValueError) as ctx:
            validate_space_name("allen_mouse_fbluebrain_barrels", self.metadata)

        msg = str(ctx.exception)
        self.assertIn("allen_mouse_bluebrain_barrels", msg)
        self.assertIn("Did you mean", msg)


if __name__ == "__main__":
    unittest.main()
