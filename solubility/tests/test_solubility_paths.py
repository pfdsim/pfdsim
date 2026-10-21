"""Exercise relocation and fingerprint integrity without running assessments."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'solubility_paths', Path(__file__).resolve().parents[1] / 'solubility_paths.py')
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class ManifestTests(unittest.TestCase):
    def test_whole_checkout_can_move_without_losing_inputs_or_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / 'original'
            source = original / 'study' / 'data' / 'observations.csv'
            source.parent.mkdir(parents=True)
            source.write_text('x\n0.01\n')
            output = original / 'study' / 'results' / 'run'
            output.mkdir(parents=True)
            # The helper is itself an input. Place a copy in the simulated
            # checkout so all manifest dependencies move together.
            helper = original / 'study' / 'solubility_paths.py'
            shutil.copy2(paths.__file__, helper)
            old_file = paths.__file__
            try:
                paths.__file__ = str(helper)
                paths.write_manifest(output / 'manifest.json',
                                     {'data_dir': source.parent, 'output_dir': output,
                                      'workers': 2}, [source], seed=123)
            finally:
                paths.__file__ = old_file
            relocated = Path(temporary) / 'relocated'
            original.rename(relocated)
            manifest_path = relocated / 'study' / 'results' / 'run' / 'manifest.json'
            manifest = paths.verify_manifest(manifest_path)
            self.assertEqual(manifest['seed'], 123)
            self.assertEqual(manifest['arguments']['workers'], 2)
            for value in manifest['sha256']:
                self.assertFalse(Path(value).is_absolute())
            data_dir = manifest_path.parent / manifest['arguments']['data_dir']
            self.assertEqual((data_dir / source.name).read_text(), 'x\n0.01\n')
            self.assertEqual((manifest_path.parent / manifest['arguments']['output_dir']).resolve(),
                             manifest_path.parent)
            (data_dir / source.name).write_text('x\n0.02\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                paths.verify_manifest(manifest_path)

    def test_custom_inputs_and_output_need_not_share_a_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / 'external.csv'
            source.write_text('source data')
            output = base / 'nested' / 'run'
            output.mkdir(parents=True)
            manifest = output / 'manifest.json'
            paths.write_manifest(manifest, {'data_dir': source.parent}, [source])
            paths.verify_manifest(manifest)
            self.assertEqual(json.loads(manifest.read_text())['arguments']['data_dir'], '../..')
            with self.assertRaises(FileExistsError):
                paths.write_manifest(manifest, {}, [source])

    def test_missing_historical_source_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / 'manifest.json'
            manifest.write_text(json.dumps({
                'path_base': 'manifest', 'sha256': {},
                'unavailable_historical_code_sha256': {'old_script.py': 'missing'},
            }))
            with self.assertRaisesRegex(ValueError, 'Historical source snapshots are unavailable'):
                paths.verify_manifest(manifest)

    def test_absolute_fingerprint_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / 'manifest.json'
            manifest.write_text(json.dumps({
                'path_base': 'manifest', 'sha256': {str(manifest): 'invalid'},
            }))
            with self.assertRaisesRegex(ValueError, 'Absolute fingerprint path'):
                paths.verify_manifest(manifest)
