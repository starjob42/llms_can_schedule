"""Excluded native-proposer format pilot; no validation/test schedules are run."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from . import optimizer as o
from experiments.phi3_revision.evaluate import artifact_digest
from experiments.phi3_revision.run_study import parse_gpu_inventory, idle_a6000


def idle_uuid(uuid):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    proc = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    matches = [d for d in parse_gpu_inventory(gpu, proc) if d['uuid'] == uuid]
    if len(matches) != 1 or not idle_a6000(matches[0]):
        raise RuntimeError('Excluded engineering pilot requires a freshly verified idle A6000')
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-path', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(o.HERE/'artifacts'):
        raise ValueError('Engineering output must remain under auto_prompt/artifacts')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh engineering output directory; do not overwrite evidence')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.gpu_uuid:
        raise ValueError('Pilot CUDA assignment differs from the verified device UUID')
    base = artifact_digest(args.base_path)
    device = idle_uuid(args.gpu_uuid)  # Immediately before any CUDA model allocation.
    output.mkdir(parents=True, exist_ok=True)
    pilot = {'purpose': 'ENGINEERING_ONLY', 'pilot_excluded': True,
             'base_artifacts': base, 'proposer': o.PROPOSER_CONFIG,
             'synthetic_feedback_only': True, 'scientific_schedule_candidates': 0,
             'host': socket.gethostname(), 'pid': os.getpid(), 'device_before_load': device,
             'source_sha256': {str(p): o.file_hash(p) for p in (Path(__file__), Path(o.__file__))}}
    protocol = {'protocol_sha256': o.json_hash(pilot), 'output_dir': str(output)}
    pilot['engineering_protocol_sha256'] = protocol['protocol_sha256']
    o.write_json(output/'PILOT.json', pilot)
    proposer = o.NativeProposer(args.base_path, base['sha256'])
    initial = o._initial_instruction(proposer.tokenizer)
    accepted = [initial]
    o.immutable_json(output/'instructions'/f"{initial['id']}.json", initial)
    for round_index in (1, 2):
        # Synthetic incumbent choice exercises chaining, never performance tuning.
        incumbent = initial if round_index == 1 else accepted[1]
        feedback = dict(purpose='ENGINEERING_ONLY', synthetic_aggregates=True,
            instruction_id=incumbent['id'], text_sha256=incumbent['text_sha256'],
            n_instances=2, candidate_count=3, n_candidates=6, feasible_candidates=3,
            conditional_mean_gap_percent=25.0, error_counts={'missing_operation': 3})
        for slot in (1, 2):
            instruction = o.propose_instruction(protocol, incumbent, feedback,
                            accepted, round_index, slot, proposer)
            accepted.append(instruction)
            print(json.dumps({'purpose': 'ENGINEERING_ONLY', 'instruction': instruction}), flush=True)
    attempts = sorted(output.glob('proposals/*/attempt_*.json'))
    records = [json.loads(p.read_text()) for p in attempts]
    for record in records:
        o.check_hash(record, 'attempt_sha256')
    checks = {
        'native_pinned_unadapted_base': all(r['generator_metadata']['unadapted'] and
            r['generator_metadata']['base_artifacts_sha256'] == base['sha256'] for r in records),
        'all_four_generated_instructions_accepted_within_four_attempts': len(accepted) == 5 and
            all(1 <= s['origin']['attempt'] <= 4 for s in accepted[1:]),
        'accepted_text_printable_ascii_and_official_tokens_bounded': all(
            all(32 <= ord(ch) <= 126 for ch in s['text']) and s['token_count'] <= 128 for s in accepted),
        'all_five_instruction_texts_unique': len({o.normalization_key(s['text']) for s in accepted}) == 5,
        'every_attempt_saved_with_messages_and_seed': all(r['messages'] and
            r['seed'] == o.proposal_seed(r['round'], r['slot'], r['attempt']) for r in records),
        'synthetic_aggregate_feedback_only': all(r['validation_feedback']['synthetic_aggregates'] for r in records),
        'no_scientific_validation_or_test_schedule_candidates': True,
    }
    if not all(checks.values()):
        raise RuntimeError('Excluded native proposer engineering checks failed')
    certificate = dict(status='verified', purpose='ENGINEERING_ONLY', checks=checks,
        created_at_utc=o.now(), pid=os.getpid(), gpu_uuid=args.gpu_uuid,
        generated_instructions=4, proposal_attempts=len(records), scientific_schedule_candidates=0,
        artifact_sha256={str(p.relative_to(output)): o.file_hash(p) for p in
                        [output/'PILOT.json', *attempts, *sorted((output/'instructions').glob('*.json'))]})
    o.write_json(output/'ENGINEERING_CERTIFICATE.json', o.with_hash(certificate, 'certificate_sha256'))


if __name__ == '__main__':
    main()
