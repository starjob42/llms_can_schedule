"""SYNTHETIC temporary publication transitions; no processes or real artifacts."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import monitor_live as monitor
except ImportError:
    import monitor_live as monitor


def write_hashed(path, value):
    import hashlib
    payload = dict(value)
    payload['receipt_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


class JournalTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='SYNTHETIC_monitor_publication_')
        project = Path(self.temporary.name)
        self.root = project / 'experiments/phi3_revision'
        self.artifacts = self.root / 'auto_prompt/artifacts'
        self.paper = project / 'LLMs_can_Schedule_paper'
        self.paper.mkdir()
        self.protocol = {'protocol_sha256': 'SYNTHETIC_PROTOCOL', 'output_dir': str(self.artifacts / 'production')}
        scientific = write_hashed(self.artifacts / 'production/COMPLETE.json', {
            'purpose': 'SCIENTIFIC_AUTO_PROMPT', 'status': 'complete', 'protocol_sha256': 'SYNTHETIC_PROTOCOL'})
        for name in ('neurips_2024.tex', 'neurips_2024.pdf', 'rebuttal.tex', 'rebuttal.pdf'):
            (self.paper / name).write_text('SYNTHETIC ORIGINAL ' + name)
        self.publisher_path = self.artifacts / 'publishing/COMPLETE.json'
        evidence = self.artifacts / 'publishing/attempt_SYNTHETIC'
        evidence.mkdir(parents=True)
        self.original_build = self.paper / 'build/original_frozen_build'
        self.original_build.mkdir(parents=True)
        (self.original_build / 'rebuttal.pdf').write_bytes((self.paper / 'rebuttal.pdf').read_bytes())
        (evidence / 'LAUNCH.json').write_text(json.dumps({'build_dir': str(self.original_build)}))
        self.publisher = write_hashed(self.publisher_path, {
            'purpose': 'VERIFIED_AUTO_PROMPT_PUBLICATION', 'status': 'complete',
            'protocol_sha256': 'SYNTHETIC_PROTOCOL', 'scientific_receipt_sha256': scientific['receipt_sha256'],
            'paper_source_sha256': monitor.digest(self.paper / 'neurips_2024.tex'),
            'rebuttal_source_sha256': monitor.digest(self.paper / 'rebuttal.tex'),
            'evidence_dir': str(evidence),
            'pdfs': {name: {'pdf_sha256': monitor.digest(self.paper / (name + '.pdf'))}
                     for name in ('neurips_2024', 'rebuttal')}})

    def tearDown(self):
        self.temporary.cleanup()

    def bridge(self):
        backup = self.paper / 'build/original_publisher'
        backup.mkdir(parents=True)
        for name in ('rebuttal.tex', 'rebuttal.pdf'):
            (backup / name).write_bytes((self.paper / name).read_bytes())
        names = ('journal_abstract.tex', 'journal_intro.tex', 'journal_conclusion.tex',
                 'journal_results.tex', 'journal_status.tex', 'rebuttal.tex')
        for name in names:
            (self.paper / name).write_text('SYNTHETIC DERIVATIVE ' + name)
        self.bridge_path = self.paper / 'build/JOURNAL_RESULT_INTEGRATION.json'
        write_hashed(self.bridge_path, {
            'status': 'verified_journal_results_integrated_pdf_rebuild_required',
            'publisher_receipt_file_sha256': monitor.digest(self.publisher_path),
            'outputs_sha256': {name: monitor.digest(self.paper / name) for name in names},
            'backups': {name: str(backup / name) for name in ('rebuttal.tex', 'rebuttal.pdf')}})

    def final_build(self):
        (self.paper / 'manuscript.tex').write_text('SYNTHETIC JOURNAL WRAPPER')
        for name in ('rebuttal.pdf', 'manuscript.pdf'):
            (self.paper / name).write_text('SYNTHETIC FINAL ' + name)
        write_hashed(self.paper / 'JOURNAL_BUILD.json', {
            'status': 'compiled_and_checked', 'experimental_results_pending': False,
            'upstream_journal_integration_receipt_sha256': monitor.digest(self.bridge_path),
            'frozen_publisher_receipt_file_sha256': monitor.digest(self.publisher_path),
            'rebuttal_source_sha256': monitor.digest(self.paper / 'rebuttal.tex'),
            'source_sha256': {'manuscript.tex': monitor.digest(self.paper / 'manuscript.tex')},
            'pdf_checks': {name: {'sha256': monitor.digest(self.paper / (name + '.pdf'))}
                           for name in ('manuscript', 'rebuttal')}})

    def test_original_publication_checks_stay_strict(self):
        self.assertTrue(monitor.publication_done(self.root, self.artifacts, self.protocol))
        (self.paper / 'rebuttal.tex').write_text('SYNTHETIC UNBOUND CHANGE')
        with self.assertRaisesRegex(ValueError, 'source differs'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)

    def test_bridge_before_pdf_build_is_healthy_but_journal_pending(self):
        self.bridge()
        stage = {}
        self.assertTrue(monitor.publication_done(self.root, self.artifacts, self.protocol, stage))
        self.assertEqual(stage['status'], 'integrated_pdf_rebuild_pending')
        self.assertFalse(stage['final_root_review_complete'])
        (self.paper / 'rebuttal.pdf').write_text('SYNTHETIC UNBOUND PDF')
        with self.assertRaisesRegex(ValueError, 'lacks a verified final'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)

    def test_final_build_binds_current_sources_and_both_pdfs(self):
        self.bridge()
        self.final_build()
        stage = {}
        self.assertTrue(monitor.publication_done(self.root, self.artifacts, self.protocol, stage))
        self.assertEqual(stage['status'], 'final_compiled_root_review_required')
        self.assertFalse(stage['final_root_review_complete'])
        (self.paper / 'manuscript.tex').write_text('SYNTHETIC UNREBUILT AUTHOR CHANGE')
        with self.assertRaisesRegex(ValueError, 'source hashes changed'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)

    def test_tampered_backup_or_bridge_never_bypasses_original_receipt(self):
        self.bridge()
        backup = self.paper / 'build/original_publisher/rebuttal.tex'
        backup.write_text('SYNTHETIC WRONG ORIGINAL')
        with self.assertRaisesRegex(ValueError, 'original publisher-bound rebuttal source'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)

    def test_original_pdf_fallback_requires_exact_immutable_build(self):
        self.bridge()
        (self.paper / 'build/original_publisher/rebuttal.pdf').write_text('SYNTHETIC WRONG BACKUP')
        self.assertTrue(monitor.publication_done(self.root, self.artifacts, self.protocol))
        (self.original_build / 'rebuttal.pdf').write_text('SYNTHETIC WRONG BUILD PDF')
        with self.assertRaisesRegex(ValueError, 'original publisher-bound rebuttal PDF'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)

    def test_final_pdf_tampering_is_rejected(self):
        self.bridge()
        self.final_build()
        (self.paper / 'rebuttal.pdf').write_text('SYNTHETIC UNBOUND FINAL PDF')
        with self.assertRaisesRegex(ValueError, 'PDF differs from build receipt'):
            monitor.publication_done(self.root, self.artifacts, self.protocol)


if __name__ == '__main__':
    unittest.main()
