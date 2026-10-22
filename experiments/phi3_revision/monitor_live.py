#!/usr/bin/env python3
"""One read-only snapshot of the live study and its publication chain.

Run on the manager host with authoritative process visibility. In Codex, invoke
exec_command with sandbox_permissions='require_escalated'; a sandbox PID
namespace can otherwise turn a live host PID into a misleading 'not found'.
Example: python3 -B experiments/phi3_revision/monitor_live.py --once

Only reads shared artifacts and invokes ps/SSH ps. No signals, job launches,
restarts, GPU access, imports of experiment code, or file writes. Exit codes:
0 healthy/completed, 1 confirmed failed/dead job, 2 observation unknown
(including transient SSH failure or an unreadable/invalid receipt). This
snapshot is not the final raw audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def json_read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def self_hashed(value, key):
    payload = {k: v for k, v in value.items() if k != key}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if value.get(key) != actual:
        raise ValueError(f'invalid {key}')
    return value


def tail(path, length=65536):
    path = Path(path)
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - length))
        return stream.read(length).decode(errors='replace')


def process_probe(host, pid, expected):
    """ps rejects zombies; SSH transport failures are unknown, never dead."""
    result = {'host': host, 'pid': pid}
    if type(pid) is not int or pid <= 0:
        return dict(result, state='unknown', reason='missing or malformed PID')
    local = {'localhost', '127.0.0.1', socket.gethostname(), socket.gethostname().split('.')[0]}
    command = ['ps', '-o', 'stat=', '-o', 'args=', '-p', str(pid)]
    remote = host not in local
    if remote:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host):
            return dict(result, state='unknown', reason='invalid host name')
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
                   '-o', 'ControlMaster=no', '-o', 'ControlPath=none', host,
                   'ps -o stat= -o args= -p ' + str(pid)]
    try:
        observed = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(result, state='unknown', reason=f'{type(exc).__name__}: {str(exc)[:160]}')
    if remote and observed.returncode == 255:
        return dict(result, state='unknown', reason='SSH transport: ' + observed.stderr.strip()[-200:])
    if observed.returncode == 1 and not observed.stdout.strip() and not observed.stderr.strip():
        return dict(result, state='dead')
    if observed.returncode != 0 or not observed.stdout.strip():
        return dict(result, state='unknown', reason=f'ps exit {observed.returncode}: {observed.stderr.strip()[-160:]}')
    fields = observed.stdout.strip().split(maxsplit=1)
    state, args = fields[0], fields[1] if len(fields) == 2 else ''
    if state.startswith(('Z', 'X')):
        return dict(result, state='dead', ps_state=state, reason='zombie or exited process')
    if expected and not any(token in args for token in expected):
        return dict(result, state='unexpected_process', ps_state=state, reason='PID command identity mismatch')
    return dict(result, state='alive', ps_state=state)


def raw_counts(folder):
    path = Path(folder) / 'candidates.jsonl'
    result = {'complete_rows': 0, 'unique_candidate_slots': 0, 'partial_last_line': False}
    identities = set()
    if not path.is_file():
        return result
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                result['partial_last_line'] = True
                break  # A concurrent append can be observed before its flush.
            if not line.strip():
                continue
            row = json.loads(line)
            identities.add((row['instance_id'], row['prompt_id'], row['candidate_id']))
            result['complete_rows'] += 1
    result['unique_candidate_slots'] = len(identities)
    return result


def evaluation_done(folder):
    folder = Path(folder)
    marker = json_read(folder / 'COMPLETE')
    if marker is None:
        return False
    manifest = self_hashed(json_read(folder / 'evaluation_manifest.json'), 'configuration_sha256')
    summary = json_read(folder / 'summary.json')
    expected = manifest['expected_candidate_slots']
    if (manifest['run_kind'] != 'FULL' or marker.get('status') != 'complete'
            or summary.get('status') != 'complete'
            or marker.get('configuration_sha256') != manifest['configuration_sha256']
            or summary.get('configuration_sha256') != manifest['configuration_sha256']
            or marker.get('candidate_slots') != expected or summary.get('terminal_candidate_slots') != expected
            or summary.get('not_generated_slots') != 0):
        raise ValueError('inconsistent evaluation completion receipt')
    return True


def training_done(runs):
    marker = json_read(runs / 'main/training_complete.json')
    manifest = json_read(runs / 'main/run_manifest.json', {})
    return bool(marker and marker.get('global_step') == 1844
        and not marker.get('pilot') and manifest.get('status') == 'complete'
        and not manifest.get('config', {}).get('pilot')
        and json_read(runs / 'main/checkpoint-1844/experiment_checkpoint.json', {}).get('global_step') == 1844)


def journal_publication_state(paper, publisher_path, publisher):
    """Observe a legitimate derivative without weakening its frozen evidence."""
    paper = Path(paper)
    path = paper / 'build/JOURNAL_RESULT_INTEGRATION.json'
    waiting = {'status': 'not_integrated', 'final_root_review_complete': False}
    if not path.is_file():
        return waiting
    bridge = self_hashed(json_read(path), 'receipt_sha256')
    if (bridge.get('status') != 'verified_journal_results_integrated_pdf_rebuild_required'
            or bridge.get('publisher_receipt_file_sha256') != digest(publisher_path)):
        raise ValueError('journal bridge does not bind the frozen publisher receipt')
    required = {'journal_abstract.tex', 'journal_intro.tex', 'journal_conclusion.tex',
                'journal_results.tex', 'journal_status.tex', 'rebuttal.tex'}
    if set(bridge.get('outputs_sha256', {})) != required:
        raise ValueError('journal bridge has incomplete or unexpected output identities')
    for name, expected in bridge['outputs_sha256'].items():
        if digest(paper / name) != expected:
            raise ValueError('journal bridge output changed: ' + name)
    old_source = Path(bridge.get('backups', {}).get('rebuttal.tex', ''))
    if not old_source.is_file() or digest(old_source) != publisher['rebuttal_source_sha256']:
        raise ValueError('original publisher-bound rebuttal source is not preserved')
    old_pdf = Path(bridge.get('backups', {}).get('rebuttal.pdf', ''))
    if not old_pdf.is_file() or digest(old_pdf) != publisher['pdfs']['rebuttal']['pdf_sha256']:
        # The frozen publisher's compiled build is another immutable copy of
        # the exact inspected original PDF; never accept an arbitrary new PDF.
        launch = json_read(Path(publisher['evidence_dir']) / 'LAUNCH.json', {})
        old_pdf = Path(launch.get('build_dir', '')) / 'rebuttal.pdf'
        if not old_pdf.is_file() or digest(old_pdf) != publisher['pdfs']['rebuttal']['pdf_sha256']:
            raise ValueError('original publisher-bound rebuttal PDF is not preserved')
    result = {'status': 'integrated_pdf_rebuild_pending', 'final_root_review_complete': False,
              'bridge_receipt_sha256': bridge['receipt_sha256'],
              'original_publisher_source_and_pdf_preserved': True}
    build = json_read(paper / 'JOURNAL_BUILD.json')
    # An earlier pending-draft build is not evidence of the new final PDF.
    if build and build.get('experimental_results_pending') is False:
        self_hashed(build, 'receipt_sha256')
        if (build.get('status') != 'compiled_and_checked'
                or build.get('upstream_journal_integration_receipt_sha256') != digest(path)
                or build.get('frozen_publisher_receipt_file_sha256') != digest(publisher_path)
                or build.get('rebuttal_source_sha256') != digest(paper / 'rebuttal.tex')):
            raise ValueError('final journal build does not bind its upstream/source receipts')
        sources = build.get('source_sha256', {})
        if 'manuscript.tex' not in sources or any(digest(paper / name) != expected for name, expected in sources.items()):
            raise ValueError('final journal build source hashes changed')
        for name in ('manuscript', 'rebuttal'):
            if digest(paper / (name + '.pdf')) != build['pdf_checks'][name]['sha256']:
                raise ValueError('final journal ' + name + ' PDF differs from build receipt')
        result.update(status='final_compiled_root_review_required',
                      journal_build_receipt_sha256=build['receipt_sha256'])
    elif digest(paper / 'rebuttal.pdf') != publisher['pdfs']['rebuttal']['pdf_sha256']:
        raise ValueError('rewritten rebuttal PDF lacks a verified final JOURNAL_BUILD receipt')
    return result


def publication_done(root, artifacts, protocol, journal_status=None):
    if journal_status is not None:
        journal_status.update(status='waiting_for_frozen_publication', final_root_review_complete=False)
    marker = json_read(artifacts / 'publishing/COMPLETE.json')
    if marker is None:
        return False
    self_hashed(marker, 'receipt_sha256')
    scientific = self_hashed(json_read(Path(protocol['output_dir']) / 'COMPLETE.json'), 'receipt_sha256')
    if (marker.get('status') != 'complete' or marker.get('purpose') != 'VERIFIED_AUTO_PROMPT_PUBLICATION'
            or marker.get('protocol_sha256') != protocol['protocol_sha256']
            or scientific.get('status') != 'complete' or scientific.get('purpose') != 'SCIENTIFIC_AUTO_PROMPT'
            or scientific.get('protocol_sha256') != protocol['protocol_sha256']
            or marker.get('scientific_receipt_sha256') != scientific['receipt_sha256']):
        raise ValueError('invalid publication/scientific completion binding')
    paper = root.parents[1] / 'LLMs_can_Schedule_paper'
    # The legacy carrier remains unchanged by the derivative journal bridge.
    if digest(paper / 'neurips_2024.pdf') != marker['pdfs']['neurips_2024']['pdf_sha256']:
        raise ValueError('published neurips_2024 PDF hash differs from receipt')
    if digest(paper / 'neurips_2024.tex') != marker['paper_source_sha256']:
        raise ValueError('published neurips_2024.tex source differs from receipt')
    stage = journal_publication_state(paper, artifacts / 'publishing/COMPLETE.json', marker)
    if stage['status'] == 'not_integrated':
        if digest(paper / 'rebuttal.pdf') != marker['pdfs']['rebuttal']['pdf_sha256']:
            raise ValueError('published rebuttal PDF hash differs from receipt')
        if digest(paper / 'rebuttal.tex') != marker['rebuttal_source_sha256']:
            raise ValueError('published rebuttal.tex source differs from receipt')
    if journal_status is not None:
        journal_status.update(stage)
    return True


def snapshot(root, manager_host='localhost'):
    root = Path(root).resolve()
    runs, artifacts = root / 'runs', root / 'auto_prompt/artifacts'
    warnings, errors, unknown, probes, completion_checks = [], [], [], {}, {}
    state = json_read(runs / 'manager/status.json')
    if not state:
        raise ValueError('manager/status.json missing or empty')
    study = self_hashed(json_read(root / 'study/STUDY.json'), 'protocol_sha256')
    primary_marker = json_read(runs / 'manager/COMPLETE')
    primary_done = bool(primary_marker and primary_marker.get('status') == 'complete'
                        and primary_marker.get('study_protocol_sha256') == study['protocol_sha256'])
    if primary_marker and not primary_done:
        errors.append('invalid primary manager COMPLETE')
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'observer_host': socket.gethostname(), 'read_only': True,
        'manager': {'status': state.get('status'), 'updated_utc': state.get('updated_utc'),
                    'primary_complete': primary_done, 'report': state.get('report'), 'pdf': state.get('pdf')},
        'warnings': warnings, 'errors': errors, 'unknown_observations': unknown}
    if not primary_done:
        probes['manager'] = (manager_host, int((runs / 'manager.pid').read_text().strip()), ('run_study',))
        completion_checks['manager'] = lambda: (json_read(runs / 'manager/COMPLETE', {}).get('status') == 'complete'
            and json_read(runs / 'manager/COMPLETE', {}).get('study_protocol_sha256') == study['protocol_sha256'])
        if state.get('status') in ('failed_incomplete', 'interrupted_incomplete', 'timeout_incomplete'):
            errors.append('manager ' + state['status'] + ': ' + state.get('error', ''))
    heartbeat = json_read(runs / 'main/training_heartbeat.json', {})
    training_manifest = json_read(runs / 'main/run_manifest.json', {})
    training_marker = json_read(runs / 'main/training_complete.json')
    trained = training_done(runs)
    training = state.get('training', {})
    log_tail = tail(training.get('log', runs / 'manager/training.log'))
    matches = list(re.finditer(r'(\d+)\s*/\s*1000\s*\[([^\]\r\n]*)\]', log_tail))
    report['training'] = {k: heartbeat.get(k) for k in ('global_step', 'session_microbatches_finished', 'utc', 'session_seconds')}
    report['training'].update(status='complete' if trained else training.get('status'),
        last_validation_tqdm={'batch': int(matches[-1][1]), 'total': 1000, 'timing': matches[-1][2]} if matches else None)
    logged = training_manifest.get('training_logs', [])
    if isinstance(logged, list):
        for name, metric in (('latest_logged_training', 'loss'),
                             ('latest_logged_validation', 'eval_loss'),
                             ('final_logged_train_loss', 'train_loss')):
            latest = next((event for event in reversed(logged) if isinstance(event, dict) and metric in event), None)
            if latest is not None:
                report['training'][name] = dict(latest)
    if training_marker and not trained:
        errors.append('invalid training completion receipt')
    if not trained:
        if training.get('status') in ('running', 'launching'):
            probes['training'] = (training.get('host', ''), training.get('pid'), ('phi3_revision.train', 'train.py'))
            completion_checks['training'] = lambda: training_done(runs)
        if training.get('status') == 'failed' or training_manifest.get('status') in ('failed', 'interrupted'):
            errors.append('training failed: ' + training_manifest.get('error', 'see training.log'))
    report['evaluations'] = {}
    for name, job in state.get('jobs', {}).items():
        folder = runs / 'evaluations' / name
        done = evaluation_done(folder)
        value = {'status': 'complete' if done else job.get('status')}
        if not done and job.get('status') in ('running', 'launching'):
            probes['evaluation:' + name] = (job.get('host', ''), job.get('pid'), ('phi3_revision.evaluate', 'evaluate.py'))
            completion_checks['evaluation:' + name] = lambda directory=folder: evaluation_done(directory)
            value['raw'] = raw_counts(folder)
            progress = json_read(folder / 'summary.json', {})
            value.update(expected_candidate_slots=progress.get('expected_candidate_slots'),
                         summary_candidate_slots=progress.get('terminal_candidate_slots'))
        elif not done and job.get('status') in ('failed', 'complete'):
            errors.append(f'{name}: {job.get("status")} without valid complete receipt')
        report['evaluations'][name] = value
    protocol = self_hashed(json_read(artifacts / 'PROTOCOL.json'), 'protocol_sha256')
    report['journal'] = {}
    published = publication_done(root, artifacts, protocol, report['journal'])
    launch = json_read(artifacts / 'PUBLISH_LAUNCH.json', {})
    pub_state = json_read(artifacts / 'publishing/STATUS.json', {})
    auto = Path(protocol['output_dir'])
    auto_state = json_read(auto / 'STATUS.json', {})
    report['automatic_prompt'] = {'publication_complete_verified': published,
        'publication_status': pub_state.get('status'), 'publication_stage': pub_state.get('stage'),
        'scientific_status': auto_state.get('status'), 'run_bound': (auto / 'RUN.json').is_file(),
        'selected_instruction_id': json_read(auto / 'SELECTED.json', {}).get('selected_instruction_id'),
        'validation_completed': [], 'active_evaluations': {}}
    if not published:
        pid_path = artifacts / 'publish.pid'
        pid = int(pid_path.read_text().strip()) if pid_path.is_file() else launch.get('pid')
        probes['publisher'] = (launch.get('host', 'seacove'), pid, ('auto_prompt.publish', 'publish.py'))
        completion_checks['publisher'] = lambda: publication_done(root, artifacts, protocol, report['journal'])
        if pub_state.get('status') == 'failed' or json_read(artifacts / 'publishing/EXIT.json', {}).get('exit_code', 0) != 0:
            errors.append('publication chain failed; inspect publishing/STATUS.json and attempt FAILURE.json')
    for split in ('validation', 'test'):
        for folder in sorted((auto / 'evaluations' / split).glob('*')):
            if not folder.is_dir():
                continue
            if evaluation_done(folder):
                if split == 'validation':
                    report['automatic_prompt']['validation_completed'].append(folder.name)
            elif (folder / 'evaluation_manifest.json').is_file() and not published:
                report['automatic_prompt']['active_evaluations'][split + '/' + folder.name] = raw_counts(folder)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(process_probe, *arguments) for name, arguments in probes.items()}
        report['processes'] = {name: future.result() for name, future in futures.items()}
    for name, observed in report['processes'].items():
        # Completion can occur during an SSH probe. Re-read its receipt before
        # declaring an exited PID dead; completed jobs legitimately disappear.
        if observed['state'] != 'alive' and completion_checks.get(name, lambda: False)():
            observed.update(state='completed', completion_receipt_rechecked=True)
            if name == 'manager':
                report['manager'].update(status='complete', primary_complete=True)
            elif name == 'training':
                report['training']['status'] = 'complete'
            elif name == 'publisher':
                published = True
                report['automatic_prompt']['publication_complete_verified'] = True
            elif name.startswith('evaluation:'):
                report['evaluations'][name.split(':', 1)[1]]['status'] = 'complete'
        if observed['state'] in ('dead', 'unexpected_process'):
            errors.append(f'{name}: {observed["state"]} on {observed["host"]} PID{observed["pid"]}')
        elif observed['state'] == 'unknown':
            unknown.append(f'{name}: {observed.get("reason")}')
    if state.get('last_transport_error'):
        warnings.append({'manager_last_transport_error': state['last_transport_error']})
    if 'Traceback (most recent call last)' in log_tail:
        warnings.append('training log tail contains traceback; inspect original log')
    if 'Traceback (most recent call last)' in tail(artifacts / 'publish.log'):
        warnings.append('publication log tail contains traceback; inspect original log')
    # Completion of the frozen publisher is separate from final journal
    # rendering, author review, and the user's whole task.
    report['status'] = 'failed' if errors else 'unknown' if unknown else 'healthy'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--once', action='store_true', help='one snapshot (also the default)')
    parser.add_argument('--root-dir', type=Path, default=HERE)
    parser.add_argument('--manager-host', default='localhost', help='host of runs/manager.pid; default is this authoritative host')
    args = parser.parse_args()
    try:
        result = snapshot(args.root_dir, args.manager_host)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {'status': 'unknown', 'read_only': True, 'errors': [],
                  'unknown_observations': [f'{type(exc).__name__}: {exc}']}
    print(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 1 if result['status'] == 'failed' else 2 if result['status'] == 'unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
