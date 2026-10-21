"""Check exact raw downloads and preservation of existing dependency files."""

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
spec = importlib.util.spec_from_file_location('pull_pfdsim_dependencies', STUDY / 'pull_pfdsim_dependencies.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
from solubility_dependencies import SHARED_CODE_FILES, PARAMETER_FILES, SHARED_FILES
from solubility_paths import ROOT, file_sha256, relative_path, verify_manifest, write_manifest


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {'Content-Length': str(len(payload) if length is None else length)}


class DownloadTests(unittest.TestCase):
    def test_downloads_exactly_the_fifteen_declared_files_from_one_ref(self):
        requests = []

        def opener(request, timeout):
            requests.append(request.full_url)
            self.assertEqual(timeout, 12)
            return Response(request.full_url.encode())

        with tempfile.TemporaryDirectory() as temporary, patch.object(download, 'urlopen', opener):
            destination = Path(temporary)
            installed = download.download_dependencies(destination, 'owner/repo', 'feature/test', timeout=12)
            expected = [download.raw_url('owner/repo', 'feature/test', name) for name in SHARED_FILES]
            self.assertEqual(requests, expected)
            self.assertTrue(all('/feature%2Ftest/' in url for url in requests))
            self.assertEqual(set(installed), set(SHARED_FILES))
            actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
            self.assertEqual(actual, set(SHARED_FILES))
            for name, url in zip(SHARED_FILES, expected):
                self.assertEqual((destination / name).read_bytes(), url.encode())

    def test_existing_matching_files_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                download, 'urlopen', side_effect=lambda *args, **kwargs: Response(b'original')):
            destination = Path(temporary)
            existing = destination / SHARED_FILES[0]
            existing.write_bytes(b'original')
            before = existing.stat()
            installed = download.download_dependencies(destination)
            self.assertNotIn(SHARED_FILES[0], installed)
            self.assertEqual(len(installed), 14)
            self.assertEqual(existing.stat().st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(existing.stat().st_ino, before.st_ino)

    def test_conflicting_local_files_prevent_any_installation(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                download, 'urlopen', side_effect=lambda *args, **kwargs: Response(b'remote')):
            destination = Path(temporary)
            existing = destination / SHARED_FILES[-1]
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'local changes')
            with self.assertRaisesRegex(FileExistsError, 'Existing dependencies differ'):
                download.download_dependencies(destination)
            self.assertEqual(existing.read_bytes(), b'local changes')
            self.assertEqual([p for p in destination.rglob('*') if p.is_file()], [existing])

    def test_network_failure_leaves_no_dependency_files_installed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(download, 'urlopen') as opener:
            opener.side_effect = [Response(b'first file'), HTTPError('raw URL', 404, 'Not Found', {}, None)]
            with self.assertRaises(HTTPError):
                download.download_dependencies(temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_incomplete_response_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(download, 'urlopen') as opener:
            opener.return_value = Response(b'short', length=100)
            with self.assertRaisesRegex(OSError, 'Incomplete download'):
                download.download_dependencies(temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_repository_must_be_an_owner_and_name(self):
        with self.assertRaisesRegex(ValueError, 'OWNER/REPOSITORY'):
            download.raw_url('https://example.invalid/repo', 'main', 'unifac.py')


class FingerprintTests(unittest.TestCase):
    def test_base_benchmark_records_every_shared_dependency(self):
        import benchmark_solubility_transfer as benchmark

        class ManifestWritten(Exception):
            pass

        def write_and_stop(*args, **kwargs):
            write_manifest(*args, **kwargs)
            # Inspect the real manifest before launching numerical workers.
            raise ManifestWritten

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            for name in ('BigSolDBv2.0.csv', 'BigSolDBv2.0_densities.csv', 'curated-solubility-dataset.csv'):
                (data / name).write_text('fixture\n')
            output = root / 'results'
            with (patch.object(benchmark, 'self_check', return_value={}),
                  patch.object(benchmark, 'prepare', return_value=({}, {}, {}, {}, [])),
                  patch.object(benchmark, 'write_manifest', side_effect=write_and_stop),
                  patch.object(sys, 'argv', ['benchmark', '--data-dir', str(data), '--output-dir', str(output)]),
                  redirect_stdout(io.StringIO())):
                with self.assertRaises(ManifestWritten):
                    benchmark.main()
            manifest_path = output / 'manifest.json'
            saved = verify_manifest(manifest_path)
            for name in SHARED_FILES:
                self.assertEqual(saved['sha256'][relative_path(ROOT / name, manifest_path)],
                                 file_sha256(ROOT / name))

    def test_shared_inputs_include_the_transitive_support_files(self):
        self.assertEqual(len(SHARED_CODE_FILES), 6)
        self.assertEqual(len(PARAMETER_FILES), 9)
        self.assertEqual(len(set(SHARED_FILES)), 15)
        self.assertIn('compiled_cache.py', SHARED_CODE_FILES)
        self.assertIn('physical_constants.py', SHARED_CODE_FILES)
        self.assertTrue(all((ROOT / name).is_file() for name in SHARED_FILES))

    def test_support_file_changes_are_detected_after_manifest_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = []
            for name in ('compiled_cache.py', 'physical_constants.py'):
                source = root / name
                source.write_text('# original support file\n')
                files.append(source)
            manifest = root / 'manifest.json'
            write_manifest(manifest, {}, files)
            saved = verify_manifest(manifest)
            for source in files:
                self.assertEqual(saved['sha256'][relative_path(source, manifest)], file_sha256(source))
                source.write_text('# changed support file\n')
                with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                    verify_manifest(manifest)
                source.write_text('# original support file\n')
            verify_manifest(manifest)
