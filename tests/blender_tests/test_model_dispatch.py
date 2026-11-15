import unittest

from SourceIO.blender_bindings.models import mdl36, mdl44
from SourceIO.blender_bindings.models.model_tags import choose_model_importer


class ModelDispatchTests(unittest.TestCase):
    def test_embedded_beta_models_use_legacy_importer(self):
        for version in (32, 35, 36, 39):
            with self.subTest(version=version):
                self.assertIs(
                    choose_model_importer(b"IDST", version),
                    mdl36.import_mdl36,
                )

    def test_vvd_backed_models_use_later_importer(self):
        for version in (37, 38, 40, 41, 42, 43, 44):
            with self.subTest(version=version):
                self.assertIs(
                    choose_model_importer(b"IDST", version),
                    mdl44.import_mdl44,
                )


if __name__ == "__main__":
    unittest.main()
