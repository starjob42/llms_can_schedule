#!/usr/bin/env python3
"""Validation-guided global instruction refinement, separate from frozen studies.

The default CLI is observational. Scientific execution waits for the complete
primary study, then leases the explicitly verified idle seacove A6000. Proposal
workers exit before schedule evaluation workers start, releasing model memory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import unicodedata

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PROTOCOL = HERE / 'artifacts' / 'PROTOCOL.json'
DEFAULT_PYTHON = '/home/lamsade/habgaryan/miniconda3/envs/unsloth_env/bin/python'
SEACOVE_UUID = 'GPU-025e10e6-263f-d814-6dd5-added86fc8af'
INITIAL_ID = 'auto_0'
INSTRUCTION_IDS = ('auto_0', 'auto_r1_s1', 'auto_r1_s2', 'auto_r2_s1', 'auto_r2_s2')
PROPOSER_SYSTEM = (
    'You write a single global instruction for a job shop scheduling model. '
    'Improve the instruction using only the supplied validation aggregates. '
    'Return exactly one JSON object with one key, "instruction", whose value '
    'is a single-line string in English using printable ASCII characters only. '
    'Do not include Markdown, explanations, or any other keys. '
    'Object format example: {"instruction":"Schedule every operation and report the makespan."}'
)
PROPOSER_TASK = (
    'Refine the incumbent instruction for job shop scheduling. Preserve the task: '
    'schedule every operation exactly once, in job order, without overlapping '
    'operations on a machine, and minimize makespan. The schedule output must '
    'identify each job, operation and machine, provide start time, duration and '
    'end time, and report makespan. Produce a different concise instruction; '
    'do not solve a particular instance. Validation ranks instructions first by '
    'the number of feasible candidates, then by their conditional mean gap '
    'against reference makespans. Error counts are aggregate validator codes.'
)
PROPOSER_CONFIG = {'temperature': 0.8, 'top_k': 50, 'top_p': 0.95,
    'max_new_tokens': 256, 'max_attempts_per_instruction': 4,
    'min_chars': 20, 'max_chars': 600, 'max_instruction_tokens': 128,
    'seed': 42, 'precision': 'float16', 'adapter': None}
FEEDBACK_AGGREGATES = ('n_instances', 'candidate_count', 'n_candidates',
                       'feasible_candidates', 'conditional_mean_gap_percent', 'error_counts')


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Immutable experiment artifact differs: {path}')
    else:
        write_json(path, value)


def with_hash(value, key):
    value = dict(value)
    value[key] = json_hash(value)
    return value


def check_hash(value, key):
    if value.get(key) != json_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f'Artifact self-hash mismatch: {key}')


def proposal_seed(round_index, slot_index, attempt_index):
    if round_index not in (1, 2) or slot_index not in (1, 2) or attempt_index not in (1, 2, 3, 4):
        raise ValueError('Proposal coordinates outside the fixed two-round budget')
    material = f'phi3_auto_prompt:42:{round_index}:{slot_index}:{attempt_index}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') % (2**31-1)


def feedback_projection(feedback):
    """Explicit allowlist prevents problem data or test provenance in messages."""
    selected = {key: feedback[key] for key in FEEDBACK_AGGREGATES}
    for key in ('n_instances', 'candidate_count', 'n_candidates', 'feasible_candidates'):
        if type(selected[key]) is not int or selected[key] < 0:
            raise ValueError(f'Invalid aggregate feedback count: {key}')
    if not selected['n_instances'] or not selected['candidate_count'] or selected['n_candidates'] != selected['n_instances']*selected['candidate_count']:
        raise ValueError('Feedback candidate denominator is inconsistent')
    if selected['feasible_candidates'] > selected['n_candidates']:
        raise ValueError('Feasible count exceeds denominator')
    gap = selected['conditional_mean_gap_percent']
    if selected['feasible_candidates']:
        if type(gap) not in (int, float) or not math.isfinite(gap):
            raise ValueError('Feasible validation feedback requires a finite conditional gap')
    elif gap is not None:
        raise ValueError('No feasible candidates requires a null conditional gap')
    errors = selected['error_counts']
    if not isinstance(errors, dict) or any(not isinstance(k, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', k)
            or type(v) is not int or not 0 <= v <= selected['n_candidates'] for k, v in errors.items()):
        raise ValueError('Feedback errors must be aggregate validator codes and counts')
    selected['error_counts'] = dict(sorted(errors.items()))
    return selected


def build_proposer_messages(incumbent, feedback, round_index, slot_index,
                            attempt_index, previous_rejection=None):
    proposal_seed(round_index, slot_index, attempt_index)
    if not isinstance(incumbent, dict) or not isinstance(incumbent.get('text'), str):
        raise ValueError('Incumbent instruction text is required')
    if previous_rejection is not None and previous_rejection not in ProposalRejected.CODES:
        raise ValueError('Unknown prior format rejection')
    payload = {'task': PROPOSER_TASK, 'incumbent_instruction': incumbent['text'],
        'validation_feedback': feedback_projection(feedback),
        'limits': {'single_line': True, 'minimum_characters': 20,
                   'maximum_characters': 600, 'maximum_instruction_tokens': 128,
                   'english_printable_ascii_only': True,
                   'no_leading_or_trailing_whitespace': True,
                   'unique_against_previously_accepted_instructions': True,
                   'no_role_or_special_token_delimiters': True},
        'response_schema': {'type': 'object', 'required': ['instruction'],
            'additionalProperties': False, 'properties': {'instruction': {
                'type': 'string', 'minLength': 20, 'maxLength': 600}}},
        'proposal': {'round': round_index, 'slot': slot_index, 'attempt': attempt_index},
        'previous_format_rejection': previous_rejection}
    suffix = ('\n\nReturn only the required JSON object with the "instruction" key. '
              'Begin your answer with { and finish with }. Put your revised '
              'instruction inside its string value. Format example, replacing '
              'the value with your own revised instruction: '
              '{"instruction":"Schedule every operation and report the makespan."}')
    return [{'role': 'system', 'content': PROPOSER_SYSTEM},
            {'role': 'user', 'content': json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + suffix}]


class ProposalRejected(ValueError):
    CODES = frozenset(('invalid_json', 'wrong_schema', 'not_single_line',
        'invalid_character', 'non_ascii', 'surrounding_whitespace', 'character_limit', 'token_limit', 'role_delimiter', 'duplicate'))

    def __init__(self, code):
        if code not in self.CODES:
            raise ValueError('Unknown proposal rejection code')
        self.code = code
        super().__init__(code)


def normalization_key(text):
    return ' '.join(text.split()).casefold()


def parse_proposal(raw_output, tokenizer, accepted_texts):
    if not isinstance(raw_output, str):
        raise ProposalRejected('invalid_json')
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Repeated JSON key')
            obj[key] = value
        return obj
    try:
        value = json.loads(raw_output, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        raise ProposalRejected('invalid_json')
    if not isinstance(value, dict) or set(value) != {'instruction'} or not isinstance(value['instruction'], str):
        raise ProposalRejected('wrong_schema')
    text = value['instruction']  # Accepted text is preserved exactly; no rewriting.
    if any(ch in text for ch in ('\n', '\r', '\u2028', '\u2029')):
        raise ProposalRejected('not_single_line')
    if text != text.strip():
        raise ProposalRejected('surrounding_whitespace')
    if any(unicodedata.category(ch).startswith('C') for ch in text):
        raise ProposalRejected('invalid_character')
    if any(ord(ch) > 126 for ch in text):
        raise ProposalRejected('non_ascii')
    if not 20 <= len(text) <= 600:
        raise ProposalRejected('character_limit')
    if re.search(r'<\||\|>|</?s>|(?:^|\s)(?:system|assistant|user)\s*:', text, re.I):
        raise ProposalRejected('role_delimiter')
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count > 128:
        raise ProposalRejected('token_limit')
    key = normalization_key(text)
    if key in {normalization_key(previous) for previous in accepted_texts}:
        raise ProposalRejected('duplicate')
    return {'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'token_count': token_count, 'normalization_key': key}


def make_instruction(instruction_id, parsed, round_index, slot_index, attempt_index):
    order = 0 if round_index == 0 else 2*(round_index-1)+slot_index
    if INSTRUCTION_IDS[order] != instruction_id:
        raise ValueError('Instruction identity disagrees with creation order')
    return dict(id=instruction_id, text=parsed['text'], text_sha256=parsed['text_sha256'],
                origin={'round': round_index, 'slot': slot_index, 'attempt': attempt_index},
                token_count=parsed['token_count'], creation_order=order)


def selection_key(instruction, feedback):
    projection = feedback_projection(feedback)
    gap = projection['conditional_mean_gap_percent']
    return (-projection['feasible_candidates'], float('inf') if gap is None else gap,
            instruction['creation_order'])


def choose_incumbent(instructions, scores):
    if not instructions or len({s['id'] for s in instructions}) != len(instructions):
        raise ValueError('Require nonempty unique instruction identities')
    if set(scores) != {s['id'] for s in instructions}:
        raise ValueError('Every eligible instruction needs verified validation feedback')
    populations = set()
    for spec in instructions:
        feedback = scores[spec['id']]
        if feedback.get('split') != 'validation' or feedback.get('instruction_id') != spec['id'] or feedback.get('text_sha256', feedback.get('instruction_sha256')) != spec['text_sha256']:
            raise ValueError('Selection requires matching validation-only instruction feedback')
        populations.add((feedback.get('instance_ids_sha256'), feedback['n_instances'], feedback['candidate_count'], feedback['n_candidates']))
    if len(populations) != 1 or next(iter(populations))[0] is None:
        raise ValueError('Instructions were evaluated on different validation populations')
    return min(instructions, key=lambda spec: selection_key(spec, scores[spec['id']]))


class NativeProposer:
    """Unadapted official Phi-3 with its native chat and fixed proposal decoding."""
    def __init__(self, base_path, base_artifacts_sha256):
        from experiments.phi3_revision.evaluate import EvaluationConfig, NativeGenerator
        config = EvaluationConfig(base_path=str(base_path), data='', output_dir='',
                    adapter=None, temperature=.8, top_k=50, top_p=.95)
        self.engine = NativeGenerator(config)
        if hasattr(self.engine.model, 'peft_config'):
            raise RuntimeError('The proposer must use the unadapted base')
        self.tokenizer = self.engine.tokenizer
        self.base_artifacts_sha256 = base_artifacts_sha256

    def propose(self, messages, seed):
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_tensors='pt').to('cuda')
        if encoded.shape[-1]+256 > self.engine.max_context:
            raise RuntimeError('Proposal context exceeds the unadapted model context')
        sample = self.engine.generate(encoded, 1, seed, 256)[0]
        metadata = {k: v for k, v in sample.items() if k != 'output'}
        metadata.update(base_artifacts_sha256=self.base_artifacts_sha256,
                        unadapted=True, temperature=.8, top_k=50, top_p=.95,
                        max_new_tokens=256, seed=seed, runtime=self.engine.runtime_info())
        return sample['output'], metadata


def _attempt_path(output, round_index, slot_index, attempt_index):
    return Path(output)/'proposals'/f'round_{round_index}_slot_{slot_index}'/f'attempt_{attempt_index}.json'


def propose_instruction(protocol, incumbent, feedback, accepted, round_index,
                        slot_index, proposer, *, output_dir=None):
    """Persist each raw attempt before exposing an accepted instruction."""
    output = Path(output_dir or protocol['output_dir'])
    instruction_id = f'auto_r{round_index}_s{slot_index}'
    previous_rejection = None
    for attempt_index in range(1, 5):
        path = _attempt_path(output, round_index, slot_index, attempt_index)
        failure_path = path.with_suffix('.failure.json')
        if failure_path.exists():
            raise RuntimeError('A prior proposal generation failure is preserved; refusing automatic retry')
        messages = build_proposer_messages(incumbent, feedback, round_index,
                         slot_index, attempt_index, previous_rejection)
        seed = proposal_seed(round_index, slot_index, attempt_index)
        header = dict(schema_version=1, protocol_sha256=protocol['protocol_sha256'],
            round=round_index, slot=slot_index, attempt=attempt_index, seed=seed,
            messages=messages, incumbent_instruction_id=incumbent['id'],
            incumbent_text_sha256=incumbent['text_sha256'], validation_feedback=feedback,
            previous_rejection=previous_rejection,
            accepted_instruction_ids_before_attempt=[s['id'] for s in accepted])
        if path.exists():
            record = json.loads(path.read_text())
            check_hash(record, 'attempt_sha256')
            if any(record.get(k) != v for k, v in header.items()):
                raise ValueError('Saved proposal attempt context changed on resume')
            raw, metadata = record['raw_output'], record['generator_metadata']
        else:
            try:
                raw, metadata = proposer.propose(messages, seed)
            except Exception as exc:
                failure = with_hash(dict(header, status='generation_failed',
                    error_type=type(exc).__name__, error_message=str(exc)), 'failure_sha256')
                immutable_json(failure_path, failure)
                raise
            record = dict(header, raw_output=raw, generator_metadata=metadata,
                          created_at_utc=now())
        try:
            parsed = parse_proposal(raw, proposer.tokenizer, [s['text'] for s in accepted])
            spec = make_instruction(instruction_id, parsed, round_index, slot_index, attempt_index)
            decision = dict(accepted=True, rejection_code=None, instruction=spec)
        except ProposalRejected as exc:
            decision = dict(accepted=False, rejection_code=exc.code, instruction=None)
        finalized = with_hash(dict(record, **decision), 'attempt_sha256') if not path.exists() else record
        if path.exists() and any(record.get(k) != v for k, v in decision.items()):
            raise ValueError('Saved proposal format/uniqueness decision changed on resume')
        immutable_json(path, finalized)
        if decision['accepted']:
            immutable_json(output/'instructions'/f'{instruction_id}.json', spec)
            return spec
        previous_rejection = decision['rejection_code']
    raise RuntimeError(f'No format-valid unique instruction after four fixed attempts: {instruction_id}')


def primary_ready(protocol):
    """Read completion/identity metadata only, never primary test outcomes."""
    runs = Path(protocol.get('runs_dir', Path(protocol['adapter_path']).parents[1]))
    complete = runs/'manager'/'COMPLETE'
    training = runs/'main'/'training_complete.json'
    if not complete.is_file() or not training.is_file():
        return False
    parent = json.loads(complete.read_text())
    train = json.loads(training.read_text())
    if parent.get('status') != 'complete' or parent.get('study_protocol_sha256') != protocol['parent_protocol_sha256']:
        raise ValueError('Primary completion does not match the frozen parent study')
    if train.get('pilot') or train.get('global_step') != 1844:
        raise ValueError('Primary training is not the complete 1844-step production run')
    manifest_path = runs/'main'/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or manifest.get('config', {}).get('pilot'):
        raise ValueError('Primary training manifest is incomplete or a pilot')
    from experiments.phi3_revision.run_study import checkpoint_ready
    if not checkpoint_ready(runs, 1844):
        raise ValueError('Final adapter checkpoint lacks its complete 1844-step marker')
    return True


def verified_idle_device(uuid=SEACOVE_UUID):
    from experiments.phi3_revision.run_study import parse_gpu_inventory, idle_a6000
    if socket.gethostname().split('.')[0] != 'seacove':
        raise RuntimeError('Automatic prompt production must run locally on seacove')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=30)
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True, timeout=30)
    matches = [d for d in parse_gpu_inventory(gpu.stdout, proc.stdout) if d['uuid'] == uuid]
    if len(matches) != 1:
        raise RuntimeError('The required seacove A6000 UUID is not present')
    return matches[0] if idle_a6000(matches[0]) else None


@contextmanager
def device_lease(output_dir):
    path = HERE/'artifacts'/'SEACOVE_GPU.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        f.write(json.dumps({'pid': os.getpid(), 'hostname': socket.gethostname(),
                            'output_dir': str(output_dir), 'gpu_uuid': SEACOVE_UUID, 'at_utc': now()}))
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _receipt_fields(verified):
    receipt = verified['receipt']
    return {key: receipt[key] for key in ('evaluation_manifest_sha256', 'candidates_sha256',
                                         'complete_sha256', 'instruction_manifest_sha256')}


def _verify_feedback(protocol, spec):
    from .evaluate import verify_instruction
    verified = verify_instruction(spec, protocol['validation_file'], protocol['adapter_path'],
        Path(protocol['output_dir'])/'evaluations'/'validation'/spec['id'], protocol['base_path'],
        split='validation', study_dir=Path(protocol['parent_study']).parent,
        protocol_path=protocol.get('_protocol_path', DEFAULT_PROTOCOL))
    feedback = dict(verified['feedback'], **_receipt_fields(verified),
                    instruction_id=spec['id'], text_sha256=spec['text_sha256'],
                    protocol_sha256=protocol['protocol_sha256'])
    feedback = with_hash(feedback, 'feedback_sha256')
    path = Path(protocol['output_dir'])/'feedback'/f"{spec['id']}.json"
    immutable_json(path, feedback)
    return feedback, _receipt_fields(verified)


def wait_idle(protocol, *, phase, poll_seconds=60):
    while True:
        if not primary_ready(protocol):
            raise RuntimeError('Primary completion disappeared before a GPU child launch')
        device = verified_idle_device()
        if device is not None:
            return device
        write_json(Path(protocol['output_dir'])/'STATUS.json', dict(
            status='waiting_for_idle_seacove_a6000', phase=phase, checked_at_utc=now()))
        time.sleep(poll_seconds)


def _evaluate(protocol, spec, *, split, poll_seconds=60):
    from .evaluate import run_instruction
    data = protocol['validation_file'] if split == 'validation' else protocol['test_file']
    wait_idle(protocol, phase=f'evaluation_{split}_{spec["id"]}', poll_seconds=poll_seconds)
    run_instruction(spec, data, protocol['adapter_path'],
        Path(protocol['output_dir'])/'evaluations'/split/spec['id'], protocol['base_path'],
        split=split, study_dir=Path(protocol['parent_study']).parent,
        protocol_path=protocol.get('_protocol_path', DEFAULT_PROTOCOL),
        python=sys.executable, device_uuid=SEACOVE_UUID)


def _read_instructions(output, ids):
    return [json.loads((Path(output)/'instructions'/f'{identity}.json').read_text()) for identity in ids]


def proposal_round_worker(protocol_path, round_index):
    """Short-lived GPU process: generate two instructions, then release all VRAM."""
    from .protocol import load_protocol
    protocol = load_protocol(protocol_path)
    protocol['_protocol_path'] = str(Path(protocol_path).resolve())
    output = Path(protocol['output_dir'])
    if not primary_ready(protocol) or not (output/'RUN.json').is_file():
        raise RuntimeError('Scientific proposals require completed primary training and a bound RUN.json')
    run_binding = json.loads((output/'RUN.json').read_text())
    check_hash(run_binding, 'run_sha256')
    if run_binding['protocol_sha256'] != protocol['protocol_sha256'] or os.environ.get('CUDA_VISIBLE_DEVICES') != SEACOVE_UUID:
        raise RuntimeError('Proposal worker lacks its bound scientific run/device')
    ids = list(INSTRUCTION_IDS[:1+2*(round_index-1)])
    accepted = _read_instructions(output, ids)
    scores = {s['id']: _verify_feedback(protocol, s)[0] for s in accepted}
    incumbent = choose_incumbent(accepted, scores)
    proposer = NativeProposer(protocol['base_path'], protocol['base_artifacts']['sha256'])
    for slot in (1, 2):
        instruction_id = f'auto_r{round_index}_s{slot}'
        spec = propose_instruction(protocol, incumbent, scores[incumbent['id']],
                    accepted, round_index, slot, proposer)
        accepted.append(spec)


def _initial_instruction(tokenizer):
    from experiments.phi3_revision.core import PROMPTS
    text = next(p.text for p in PROMPTS if p.id == 'seen_1')
    parsed = parse_proposal(json.dumps({'instruction': text}), tokenizer, [])
    return make_instruction(INITIAL_ID, parsed, 0, 0, 0)


def bind_run(protocol, device):
    from experiments.phi3_revision.evaluate import artifact_digest
    runs = Path(protocol.get('runs_dir', Path(protocol['adapter_path']).parents[1]))
    value = dict(schema_version=1, purpose='SCIENTIFIC_AUTO_PROMPT',
        protocol_sha256=protocol['protocol_sha256'], parent_protocol_sha256=protocol['parent_protocol_sha256'],
        adapter_artifacts=artifact_digest(Path(protocol['adapter_path']), adapter=True),
        primary_manager_complete_sha256=file_hash(runs/'manager'/'COMPLETE'),
        training_complete_sha256=file_hash(runs/'main'/'training_complete.json'),
        training_manifest_sha256=file_hash(runs/'main'/'run_manifest.json'),
        device_uuid=SEACOVE_UUID, device_host='seacove')
    path = Path(protocol['output_dir'])/'RUN.json'
    if path.exists():
        previous = json.loads(path.read_text())
        check_hash(previous, 'run_sha256')
        value['started_at_utc'] = previous['started_at_utc']
    else:
        value['started_at_utc'] = now()
    value = with_hash(value, 'run_sha256')
    immutable_json(path, value)
    return value


def freeze_selection(protocol, instructions, scores, receipts):
    output = Path(protocol['output_dir'])
    if (output/'evaluations'/'test').exists() and not (output/'SELECTED.json').exists():
        raise ValueError('Test artifacts appeared before automatic instruction selection')
    if [s['id'] for s in instructions] != list(INSTRUCTION_IDS):
        raise ValueError('Final selection requires all five fixed instructions')
    if set(scores) != set(INSTRUCTION_IDS) or sum(s['n_candidates'] for s in scores.values()) != 1080:
        raise ValueError('Final selection requires exactly 1080 fresh validation schedules')
    if any((s['n_instances'], s['candidate_count'], s['n_candidates']) != (72, 3, 216)
           or s.get('pilot_excluded', False) for s in scores.values()):
        raise ValueError('Final selection requires the full held-out validation budget, never pilots')
    if set(receipts) != set(INSTRUCTION_IDS) or any(set(receipt) != {
            'evaluation_manifest_sha256', 'candidates_sha256', 'complete_sha256', 'instruction_manifest_sha256'}
            or any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in receipt.values())
            for receipt in receipts.values()):
        raise ValueError('Final selection requires all five complete evaluation receipts')
    selected = choose_incumbent(instructions, scores)
    existing = output/'SELECTED.json'
    value = dict(schema_version=1, status='frozen_before_auto_test',
        protocol_sha256=protocol['protocol_sha256'], selected_instruction=selected,
        selected_instruction_id=selected['id'], text_sha256=selected['text_sha256'],
        scores=scores, validation_receipts=receipts)
    if existing.exists():
        prior = json.loads(existing.read_text())
        check_hash(prior, 'selection_sha256')
        if any(prior.get(k) != v for k, v in value.items()):
            raise ValueError('Frozen automatic prompt selection changed on resume')
        return prior
    value['selected_at_utc'] = now()
    value = with_hash(value, 'selection_sha256')
    immutable_json(existing, value)
    return value


def optimize(protocol_path, *, execute=False, poll_seconds=60):
    from .protocol import load_protocol
    protocol = load_protocol(protocol_path)
    protocol['_protocol_path'] = str(Path(protocol_path).resolve())
    if not 0 < poll_seconds <= 60:
        raise ValueError('Polling must be within (0,60] seconds')
    if not execute:
        return dict(status='prepared_not_launched', protocol_sha256=protocol['protocol_sha256'],
                    parent_ready=primary_ready(protocol), instructions=list(INSTRUCTION_IDS),
                    validation_schedules=1080, test_schedules=2160)
    output = Path(protocol['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    while not primary_ready(protocol):
        write_json(output/'STATUS.json', dict(status='waiting_for_primary_complete', checked_at_utc=now()))
        print('Waiting for complete primary manager and 1844-step training', flush=True)
        time.sleep(poll_seconds)
    with device_lease(output):
        device = verified_idle_device()
        while device is None:
            write_json(output/'STATUS.json', dict(status='waiting_for_idle_seacove_a6000', checked_at_utc=now()))
            time.sleep(poll_seconds)
            device = verified_idle_device()
        bind_run(protocol, device)
        # Tokenizer loading is CPU-only; no proposer remains resident during eval.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(protocol['base_path'], local_files_only=True,
                                                  trust_remote_code=False)
        initial = _initial_instruction(tokenizer)
        immutable_json(output/'instructions'/f'{INITIAL_ID}.json', initial)
        _evaluate(protocol, initial, split='validation', poll_seconds=poll_seconds)
        instructions, scores, receipts = [initial], {}, {}
        scores[INITIAL_ID], receipts[INITIAL_ID] = _verify_feedback(protocol, initial)
        for round_index in (1, 2):
            incumbent = choose_incumbent(instructions, scores)
            # A separate process releases every proposal-model allocation before
            # the next FP16 schedule-evaluation subprocess is started.
            command = [sys.executable, '-m', 'experiments.phi3_revision.auto_prompt.optimizer',
                       '--protocol', str(protocol_path), '--proposal-round', str(round_index)]
            environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_VISIBLE_DEVICES=SEACOVE_UUID, PYTHONDONTWRITEBYTECODE='1')
            wait_idle(protocol, phase=f'proposal_round_{round_index}', poll_seconds=poll_seconds)
            log = output/f'proposal_round_{round_index}.log'
            with log.open('a') as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            for slot in (1, 2):
                spec = _read_instructions(output, [f'auto_r{round_index}_s{slot}'])[0]
                _evaluate(protocol, spec, split='validation', poll_seconds=poll_seconds)
                instructions.append(spec)
                scores[spec['id']], receipts[spec['id']] = _verify_feedback(protocol, spec)
            round_value = dict(schema_version=1, protocol_sha256=protocol['protocol_sha256'],
                round=round_index, eligible_instruction_ids=[s['id'] for s in instructions],
                incumbent_before_id=incumbent['id'],
                selected_instruction_id=choose_incumbent(instructions, scores)['id'],
                scores=dict(scores), validation_receipts=dict(receipts))
            immutable_json(output/'rounds'/f'round_{round_index}.json', with_hash(round_value, 'round_sha256'))
        selection = freeze_selection(protocol, instructions, scores, receipts)
        _evaluate(protocol, selection['selected_instruction'], split='test', poll_seconds=poll_seconds)
        from .evaluate import verify_instruction
        verified = verify_instruction(selection['selected_instruction'], protocol['test_file'],
            protocol['adapter_path'], output/'evaluations'/'test'/selection['selected_instruction_id'],
            protocol['base_path'], split='test', study_dir=Path(protocol['parent_study']).parent,
            protocol_path=protocol_path)
        all_digests = {str(p.relative_to(output)): file_hash(p) for pattern in
                       ('proposals/*/*.json', 'feedback/*.json', 'rounds/*.json')
                       for p in sorted(output.glob(pattern))}
        complete = dict(schema_version=1, status='complete', purpose='SCIENTIFIC_AUTO_PROMPT',
            protocol_sha256=protocol['protocol_sha256'], selection_sha256=selection['selection_sha256'],
            test_instruction_id=selection['selected_instruction_id'],
            test_receipt=_receipt_fields(verified), validation_schedules=1080, test_schedules=2160,
            artifact_sha256=all_digests)
        complete = with_hash(complete, 'receipt_sha256')
        immutable_json(output/'COMPLETE.json', complete)
        write_json(output/'STATUS.json', dict(status='complete', checked_at_utc=now()))
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--proposal-round', type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.proposal_round is not None:
        proposal_round_worker(args.protocol, args.proposal_round)
    else:
        print(json.dumps(optimize(args.protocol, execute=args.execute,
                                 poll_seconds=args.poll_seconds), indent=2))


if __name__ == '__main__':
    main()
