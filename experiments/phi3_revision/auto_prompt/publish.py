#!/usr/bin/env python3
"""Finish the separate automatic-prompt study and publish verified paper PDFs.

Default: a read-only readiness plan. Explicit --execute waits through the full
scientific optimizer, verifies/integrates measured results, compiles both actual
documents and inspects every PDF page before replacing either submission PDF.
An external nohup launch can keep this process alive across disconnections.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import unicodedata
import re

sys.dont_write_bytecode = True
from . import optimizer, report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_TECTONIC = Path('/home/lamsade/habgaryan/phd_project/clean_graphopt/.tools/tectonic/tectonic')
DEFAULT_PAPER = ROOT / 'LLMs_can_Schedule_paper'
DEFAULT_PIPELINE = HERE / 'artifacts' / 'publishing'
SYSTEM_PYTHON = '/usr/bin/python3'


def checked_path(path, label):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'{label} must remain inside the project')
    return path


def normalize(text):
    text = unicodedata.normalize('NFKC', text)
    for dash in '\u2010\u2011\u2012\u2013\u2014\u2212':
        text = text.replace(dash, '-')
    return ' '.join(text.split()).lower()


def text_key(text):
    # PDF extraction can split hyphenated words and TeX ligatures across lines.
    return re.sub('[^a-z0-9]', '', normalize(text))


def check_tex_log(log):
    """Benign inputenc/underfull warnings do not hide submission defects."""
    patterns = {
        'tex_error': r'(?m)^!|Emergency stop|Fatal error',
        'undefined_reference': r'(?:Reference|Citation)[^\n]*undefined|There were undefined references|undefined citations',
        'unstable_labels': r'Label\(s\) may have changed|multiply[- ]defined|multiply defined',
        'missing_character': r'Missing character:|Missing glyph|undefined control sequence|Unicode character[^\n]*not set up',
        'overfull_box': r'Overfull\s+\\[hv]box',
    }
    defects = [label for label, pattern in patterns.items() if re.search(pattern, log, re.I)]
    if defects:
        raise RuntimeError('Final TeX log failed checks: '+', '.join(defects))
    return {'no_tex_errors': True, 'references_resolved': True, 'labels_stable': True,
            'no_missing_characters': True, 'no_overfull_boxes': True}


@contextmanager
def pipeline_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'PUBLISH.lock').open('a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_copy(source, target):
    target = Path(target)
    temporary = target.with_name(target.name+'.auto_prompt.tmp')
    with Path(source).open('rb') as incoming, temporary.open('wb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    temporary.replace(target)


def plan(protocol_path, paper_dir, pipeline_dir, tectonic):
    protocol_path = checked_path(protocol_path, 'Protocol')
    paper_dir = checked_path(paper_dir, 'Paper directory')
    pipeline_dir = checked_path(pipeline_dir, 'Publishing state directory')
    tectonic = Path(tectonic).resolve()
    if tectonic != DEFAULT_TECTONIC.resolve():
        raise ValueError('Use the reviewed cached Tectonic executable')
    if not tectonic.is_file():
        raise ValueError('The cached Tectonic executable is missing')
    version = subprocess.check_output([str(tectonic), '--version'], text=True, timeout=15).strip()
    if version != 'Tectonic 0.16.9':
        raise ValueError('Publishing requires the reviewed cached Tectonic 0.16.9')
    if not all((paper_dir/name).is_file() for name in ('neurips_2024.tex', 'rebuttal.tex')):
        raise ValueError('Both actual document sources are required')
    data = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if data:
        scientific = checked_path(data['output_dir'], 'Scientific output')
        if pipeline_dir.is_relative_to(scientific) or scientific.is_relative_to(pipeline_dir):
            raise ValueError('Publishing state must be separate from scientific output')
    return dict(status='prepared_not_launched', protocol=str(protocol_path),
        protocol_exists=protocol_path.is_file(), protocol_sha256=data.get('protocol_sha256') if data else None,
        paper_dir=str(paper_dir), pipeline_dir=str(pipeline_dir),
        tectonic=str(tectonic), tectonic_version=version,
        driver_sha256=optimizer.file_hash(__file__), interpreter=str(Path(sys.executable).resolve()),
        primary_complete_exists=Path(data['parent_complete']).is_file() if data else False,
        training_complete_exists=Path(data['training_complete']).is_file() if data else False,
        scientific_complete_exists=(Path(data['output_dir'])/'COMPLETE.json').is_file() if data else False,
        stages=['wait_and_finish_optimizer', 'verify_and_integrate_measured_results',
                'compile_both_documents', 'inspect_all_pdf_text_and_final_logs',
                'atomic_submission_pdf_replacement'], final_root_audit_required=True)


def compile_document(tectonic, paper_dir, build, name, evidence):
    command = [str(tectonic), '-C', '--keep-logs', '--keep-intermediates',
               '-o', str(build), name+'.tex']
    try:
        result = subprocess.run(command, cwd=paper_dir, text=True, capture_output=True,
            timeout=600, env=dict(os.environ, XDG_CACHE_HOME='/home/lamsade/habgaryan/.cache'))
    except subprocess.TimeoutExpired as exc:
        fragments = [part.decode(errors='replace') if isinstance(part, bytes) else part or ''
                     for part in (exc.stdout, exc.stderr)]
        (evidence/(name+'.compile.log')).write_text(''.join(fragments))
        raise
    (evidence/(name+'.compile.log')).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'{name}.tex compilation failed with exit {result.returncode}')
    log_path = build/(name+'.log')
    if not log_path.is_file():
        raise RuntimeError(f'{name} final TeX log is missing')
    checks = check_tex_log(log_path.read_text(errors='replace'))
    pdf = build/(name+'.pdf')
    if not pdf.is_file() or pdf.stat().st_size < 1000:
        raise RuntimeError(f'{name} compiled PDF is missing or empty')
    return pdf, dict(checks, final_log_sha256=optimizer.file_hash(log_path),
                     compile_log_sha256=optimizer.file_hash(evidence/(name+'.compile.log')))


def inspect_pdf(pdf, name, integrated, evidence):
    # System Python provides the already installed pypdf package independently
    # of the training interpreter. Extract every page, including appendices.
    script = (
        'import json,sys; from pypdf import PdfReader; '
        'reader=PdfReader(sys.argv[1]); '
        'print(json.dumps({"pages":len(reader.pages),'
        '"text":"\\n".join(page.extract_text() or "" for page in reader.pages)}))'
    )
    result = subprocess.run([SYSTEM_PYTHON, '-B', '-c', script, str(pdf)],
                            check=True, capture_output=True, text=True, timeout=60)
    extracted = json.loads(result.stdout)
    raw = extracted['text']
    (evidence/(name+'.extracted.txt')).write_text(raw)
    text = normalize(raw)
    key = text_key(raw)
    compact = ''.join(text.split())
    if text_key('formatting instructions') in key or not raw.strip():
        raise RuntimeError(f'{name} PDF is empty or contains the formatting sample')
    selected = integrated['selected_instruction']['id']
    count = integrated['test_summary']['best_of_k']['10']['instances_with_feasible_candidate']
    if name == 'neurips_2024':
        title = 'llms can schedule'
        phrases = ['controlled phi-3 training budget and prompt experiments',
                   'prompt wording study', 'additional automated-refinement validation results',
                   'matched test comparison of the original instruction',
                   'validation-guided automatic instruction generation and refinement']
        if extracted['pages'] <= 12:
            raise RuntimeError('Manuscript page count is inconsistent with the measured revision')
    else:
        title = 'point-by-point response to reviews'
        phrases = ['additional automatic instruction generation and refinement',
                   '1,080 validation candidates', '2,160 test candidates']
        if extracted['pages'] < 2:
            raise RuntimeError('Rebuttal page count is inconsistent with the complete response')
    if text_key(title) not in text_key(raw[:2500]):
        raise RuntimeError(f'{name} PDF does not contain its actual title')
    missing = [phrase for phrase in phrases if text_key(phrase) not in key]
    if missing:
        raise RuntimeError(f'{name} PDF lacks expected measured sections/tables: {missing}')
    if f'{count}/216' not in compact:
        raise RuntimeError(f'{name} PDF lacks the verified measured feasibility headline')
    if name == 'neurips_2024' and text_key(selected) not in key:
        raise RuntimeError('Manuscript PDF lacks the exact selected instruction identifier')
    return dict(pages=extracted['pages'], pdf_sha256=optimizer.file_hash(pdf),
        extracted_text_sha256=optimizer.file_hash(evidence/(name+'.extracted.txt')),
        actual_title_present=True, formatting_sample_absent=True,
        measured_sections_and_tables_present=True, verified_headline_present=True,
        all_pages_inspected=True)


def publish(protocol_path, paper_dir=DEFAULT_PAPER, pipeline_dir=DEFAULT_PIPELINE,
            tectonic=DEFAULT_TECTONIC, *, execute=False, poll_seconds=60):
    planned = plan(protocol_path, paper_dir, pipeline_dir, tectonic)
    if not execute:
        return planned
    if Path(sys.executable).resolve() != Path(optimizer.DEFAULT_PYTHON).resolve():
        raise RuntimeError('Execute the publishing driver with the production training/evaluation Python')
    if not planned['protocol_exists']:
        raise ValueError('A frozen scientific protocol is required before execution')
    protocol_path, paper_dir, pipeline_dir, tectonic = map(Path,
        (planned['protocol'], planned['paper_dir'], planned['pipeline_dir'], planned['tectonic']))
    with pipeline_lock(pipeline_dir):
        # Prior successful receipts remain in their attempt folders; a failed
        # new attempt must not leave a misleading current completion marker.
        for name in ('COMPLETE.json', 'EXIT.json'):
            (pipeline_dir/name).unlink(missing_ok=True)
        attempt_name = 'attempt_'+optimizer.now().replace(':', '').replace('+', '_')+'_'+str(os.getpid())
        evidence = pipeline_dir/attempt_name
        evidence.mkdir()
        build = paper_dir/'build'/'auto_prompt_publication'/attempt_name
        build.mkdir(parents=True)
        optimizer.write_json(evidence/'LAUNCH.json', dict(planned, pid=os.getpid(),
            execute=True, started_at_utc=optimizer.now(), protocol_file_sha256=optimizer.file_hash(protocol_path),
            tectonic_sha256=optimizer.file_hash(tectonic), build_dir=str(build)))
        stage = 'wait_and_finish_optimizer'
        def status(value):
            optimizer.write_json(pipeline_dir/'STATUS.json', dict(status=value, stage=stage,
                evidence_dir=str(evidence), pid=os.getpid(), checked_at_utc=optimizer.now()))
        try:
            status('running')
            scientific = optimizer.optimize(protocol_path, execute=True, poll_seconds=poll_seconds)
            if scientific.get('status') != 'complete' or scientific.get('purpose') != 'SCIENTIFIC_AUTO_PROMPT':
                raise RuntimeError('Optimizer did not return complete scientific evidence')
            if optimizer.file_hash(__file__) != planned['driver_sha256']:
                raise RuntimeError('Publishing driver changed after this durable launch')
            stage = 'verify_and_integrate_measured_results'
            status('running')
            integrated = report.integrate(protocol_path, paper_dir)
            optimizer.write_json(evidence/'INTEGRATION.json', integrated)
            if integrated['protocol_sha256'] != scientific['protocol_sha256']:
                raise RuntimeError('Scientific and publication protocols differ')
            stage = 'compile_both_documents'
            status('running')
            compiled, logs = {}, {}
            for name in ('neurips_2024', 'rebuttal'):
                compiled[name], logs[name] = compile_document(tectonic, paper_dir, build, name, evidence)
            stage = 'inspect_all_pdf_text_and_final_logs'
            status('running')
            inspections = {name: dict(inspect_pdf(pdf, name, integrated, evidence), **logs[name])
                           for name, pdf in compiled.items()}
            optimizer.write_json(evidence/'PDF_INSPECTION.json', inspections)
            # Both documents have passed before the first root PDF is replaced.
            if optimizer.file_hash(__file__) != planned['driver_sha256']:
                raise RuntimeError('Publishing driver changed before PDF publication')
            stage = 'atomic_submission_pdf_replacement'
            status('running')
            for name, pdf in compiled.items():
                target = paper_dir/(name+'.pdf')
                if target.exists():
                    shutil.copy2(target, evidence/(name+'.before_publication.pdf'))
                atomic_copy(pdf, target)
                if optimizer.file_hash(target) != inspections[name]['pdf_sha256']:
                    raise RuntimeError(f'{name} published PDF differs from its inspected build')
            receipt = dict(status='complete', purpose='VERIFIED_AUTO_PROMPT_PUBLICATION',
                protocol_sha256=scientific['protocol_sha256'],
                scientific_receipt_sha256=scientific['receipt_sha256'],
                integrated_receipt_sha256=integrated['receipt_sha256'],
                driver_sha256=optimizer.file_hash(__file__), pdfs=inspections,
                paper_source_sha256=optimizer.file_hash(paper_dir/'neurips_2024.tex'),
                rebuttal_source_sha256=optimizer.file_hash(paper_dir/'rebuttal.tex'),
                completed_at_utc=optimizer.now(), evidence_dir=str(evidence),
                final_root_audit_required=True)
            receipt = optimizer.with_hash(receipt, 'receipt_sha256')
            optimizer.write_json(evidence/'COMPLETE.json', receipt)
            optimizer.write_json(pipeline_dir/'COMPLETE.json', receipt)
            optimizer.write_json(pipeline_dir/'EXIT.json', {'exit_code': 0, 'at_utc': optimizer.now(), 'evidence_dir': str(evidence)})
            status('complete')
            return receipt
        except BaseException as exc:
            failure = dict(status='failed', stage=stage, error_type=type(exc).__name__,
                error_message=str(exc), at_utc=optimizer.now(), evidence_dir=str(evidence),
                traceback=traceback.format_exc(), final_root_audit_required=True)
            optimizer.write_json(evidence/'FAILURE.json', failure)
            optimizer.write_json(pipeline_dir/'EXIT.json', {'exit_code': 1, 'at_utc': optimizer.now(), 'evidence_dir': str(evidence)})
            status('failed')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=optimizer.DEFAULT_PROTOCOL)
    parser.add_argument('--paper-dir', type=Path, default=DEFAULT_PAPER)
    parser.add_argument('--pipeline-dir', type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument('--tectonic', type=Path, default=DEFAULT_TECTONIC)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish(args.protocol, args.paper_dir, args.pipeline_dir,
                             args.tectonic, execute=args.execute, poll_seconds=args.poll_seconds), indent=2))


if __name__ == '__main__':
    main()
