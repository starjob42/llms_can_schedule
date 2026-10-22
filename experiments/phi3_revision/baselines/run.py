#!/usr/bin/env python3
"""Evaluate authentic released policies without modifying their source trees.

Use the legacy SLJ environment. ``freeze`` binds the independent supplementary
protocol before any scientific outcomes; ``run --pilot`` uses an excluded toy
instance. Production is an explicit later invocation. All candidates, including
unsuccessful strict validations, remain in the denominator and raw output.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from experiments.phi3_revision.core import (canonical_hash, parse_matrix,
                                          validate_schedule)

DEFAULT_SOURCE = Path('/home/lamsade/habgaryan/phd_project/SelfLabelingJobShop/slj_vs_llm/SelfLabelingJobShop')
DEFAULT_STUDY = HERE.parent / 'study' / 'STUDY.json'
DEFAULT_PROTOCOL = HERE / 'artifacts' / 'PROTOCOL.json'
MODELS = {
    'l2d_20x20_1_99': ('L2D/SavedNetwork/20_20_1_99.pth',
                     'e0957c99b61a06ab0d16bea32e1bb17e223abb4f5810b4cb2532ac5e56528054'),
    'slj_b32': ('checkpoints/PtrNet-B32.pt',
                '0381711dd34c219e0661a89bd6388aa93b188be3e28400f05de96c62d7440867'),
    'slj_b128': ('checkpoints/PtrNet-B128.pt',
                 'ca1090520add93765532d8df133e967e784e9a63d497e2a5b7bae5b8885273e4'),
    'slj_b256': ('checkpoints/PtrNet-B256.pt',
                 'd97f82fff3fb5231ccfb970d12a1c050e581c5470387305e2f850201884d4d2e'),
}
PILOT_MATRIX = '2 2\n0 3 1 2\n1 2 0 4\n7\n'
CANDIDATES = 10
ABS_MS_TOLERANCE = 1e-3
REL_MS_TOLERANCE = 1e-6


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require_project_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError('Baseline writes must remain inside the project')
    return path


def load_modules(source=DEFAULT_SOURCE):
    """Legacy Params consumes argv; isolate it and forbid module shadowing."""
    source = Path(source).resolve()
    sys.dont_write_bytecode = True
    old_argv = sys.argv
    sys.path.insert(0, str(source))
    try:
        sys.argv = [sys.argv[0]]
        modules = {}
        for name in ('inout', 'sampling', 'PointerNet', 'L2D.Params',
                     'L2D.JSSP_Env', 'L2D.actor_critic', 'L2D.utils'):
            module = importlib.import_module(name)
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError(f'Incorrect legacy module was imported: {name}')
            modules[name] = module
        return modules
    finally:
        sys.argv = old_argv
        sys.path.remove(str(source))


def initialize_runtime():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def dependencies():
    """Version and installation fingerprints, plus actual imported entry files.

    METADATA/RECORD hashes identify distributions; the fingerprint deliberately
    does not purport to rehash every file in the large Torch installation.
    """
    result = {'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
              'executable_sha256': file_hash(sys.executable), 'platform': platform.platform(),
              'hostname': platform.node(), 'cpu_processor': platform.processor(),
              'cpu_count': os.cpu_count(),
              'fingerprint_scope': 'METADATA and RECORD plus imported package entry files; not a full installation content hash',
              'packages': {}}
    for distribution, module_name in (
        ('torch', 'torch'), ('torch-geometric', 'torch_geometric'),
        ('torch-scatter', 'torch_scatter'), ('torch-sparse', 'torch_sparse'),
        ('numpy', 'numpy'), ('gym', 'gym')):
        dist = importlib.metadata.distribution(distribution)
        module = importlib.import_module(module_name)
        paths = [Path(dist.locate_file(p)) for p in dist.files or []
                 if str(p).endswith(('.dist-info/METADATA', '.dist-info/RECORD'))]
        paths.append(Path(module.__file__).resolve())
        result['packages'][distribution] = {
            'version': dist.version,
            'files': {str(p): file_hash(p) for p in sorted(set(paths))}}
    return result


def checked_study(study_path):
    study_path = Path(study_path).resolve()
    study = json.loads(study_path.read_text())
    expected = study['protocol_sha256']
    if json_hash({k: v for k, v in study.items() if k != 'protocol_sha256'}) != expected:
        raise ValueError('Parent study self-hash mismatch')
    test_path = study_path.parent / 'test.jsonl'
    if file_hash(test_path) != study['artifact_sha256']['test.jsonl']:
        raise ValueError('Frozen test file hash mismatch')
    rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    identities = []
    for row in rows:
        ins = parse_matrix(row['matrix'])
        identity = canonical_hash(ins)
        if identity != row['instance_id']:
            raise ValueError('Input canonical identity mismatch')
        # Both released libraries assume classical routes, once per machine.
        for j in range(ins.n_jobs):
            if sorted(op.machine for op in ins.operations if op.job == j) != list(range(ins.n_machines)):
                raise ValueError('Baseline library requires a permutation route')
        identities.append(identity)
    if len(rows) != 216 or len(set(identities)) != 216 or identities != study['test_instance_ids']:
        raise ValueError('Expected the exact frozen 216-instance population')
    return study, test_path, rows


def provenance(source):
    source = Path(source).resolve()
    originals = {str(p.relative_to(source)): file_hash(p)
                 for p in sorted(source.rglob('*.py'))
                 if '.git' not in p.parts and '__pycache__' not in p.parts}
    own = {str(path.relative_to(PROJECT)): file_hash(path) for path in
           (Path(__file__).resolve(), HERE / '__init__.py', HERE / 'test_baselines.py', HERE.parent / 'core.py')}
    weights = {}
    for name, (rel, expected) in MODELS.items():
        actual = file_hash(source / rel)
        if actual != expected:
            raise ValueError(f'Authentic checkpoint hash mismatch: {name}')
        weights[name] = {'path': str(source / rel), 'sha256': actual,
                         'bytes': (source / rel).stat().st_size}
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    return {'source_root': str(source), 'git_head': commit,
            'source_python_sha256': originals, 'harness_sha256': own,
            'weights': weights, 'dependencies': dependencies()}


def preflight_inputs(rows, modules):
    """Check the entire frozen population without loading a learned policy."""
    import torch
    durations = []
    shapes = {}
    configs = modules['L2D.Params'].configs
    if configs.high != 99 or configs.et_normalize_coef != 1000:
        raise ValueError('Unexpected original L2D environment defaults')
    for row in rows:
        instance = parse_matrix(row['matrix'])
        for job in range(instance.n_jobs):
            if sorted(op.machine for op in instance.operations if op.job == job) != list(range(instance.n_machines)):
                raise ValueError('L2D/SLJ require one visit to each machine per job')
        for op in instance.operations:
            if not float(op.duration).is_integer() or not 0 < op.duration <= configs.high:
                raise ValueError('Duration incompatible with integer L2D high99 environment')
            durations.append(int(op.duration))
        data = build_input(instance, modules, row['instance_id'])
        if data['x'].shape != (len(instance.operations), 15) or not torch.isfinite(data['x']).all():
            raise ValueError('Nonfinite or malformed original SLJ features')
        shapes[data['shape']] = shapes.get(data['shape'], 0) + 1
    return {'instances': len(rows), 'learned_policy_forward_calls': 0,
            'finite_original_slj_features': True, 'classical_permutation_routes': True,
            'integer_durations': True, 'minimum_duration': min(durations),
            'maximum_duration': max(durations), 'l2d_environment_high': configs.high,
            'l2d_end_time_normalization': configs.et_normalize_coef,
            'shape_counts': shapes}


def freeze(protocol_path=DEFAULT_PROTOCOL, study_path=DEFAULT_STUDY, source=DEFAULT_SOURCE):
    protocol_path = require_project_path(protocol_path)
    if protocol_path.exists():
        raise FileExistsError('A frozen baseline protocol already exists; use verify')
    initialize_runtime()
    modules = load_modules(source)
    study, test_path, rows = checked_study(study_path)
    protocol = {
        'schema_version': 1, 'study': 'supplementary_released_baselines_fixed_216_k10',
        'status': 'frozen_before_baseline_scientific_outcomes', 'frozen_at_utc': now(),
        'parent_study': str(Path(study_path).resolve()),
        'parent_protocol_sha256': study['protocol_sha256'],
        'test_file': str(test_path), 'test_file_sha256': file_hash(test_path),
        'instance_ids': [row['instance_id'] for row in rows], 'instances': 216,
        'models': list(MODELS), 'candidates_per_instance': CANDIDATES,
        'seed': 42, 'seed_rule': 'SHA256(instance_id:42:0), first 8 bytes modulo (2**31-1); independent of model; seed Python, NumPy and Torch before each instance',
        'candidate_mode': 'stochastic multinomial sampling; no greedy candidate; no early stopping or deduplication',
        'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True,
        'evaluation_mode': 'explicit eval() on all neural modules; no gradients',
        'batching': {'l2d': '10 sequential trajectories per instance', 'slj': 'one original sampling call with batch size 10'},
        'l2d': {'machine_conversion': 'canonical zero-based machine IDs plus 1 for original permissibleLeftShift',
                'schedule_reconstruction': 'original environment temp1 completion times minus canonical integer durations; cross-check original machine orders and start times',
                'architecture': 'released 20x20 duration1..99 policy; original default GNN/MLP architecture',
                'mode_difference': 'original testL2D.py does not call eval(); this study explicitly freezes BatchNorm running statistics with eval()'},
        'slj': {'schedule_reconstruction': 'trace each original JobShopStates.__schedule__ call without changing its arguments; replay complete sampled global-operation order with integer durations',
                'last_operation': 'released sampling omits its uniquely forced last operation from returned matrix; fill its unique -1 only after verifying the captured final decision and machine',
                'architecture_caveat': 'released B256 uses a 3-head decoder (210177 parameters), whereas B32/B128 use 4 heads (316609); these are released configurations, not a controlled beta-only ablation',
                'feature_extraction': 'original inout graph_edges/extract_features/standardize unchanged, including its machine-feature normalization'},
        'library_makespan_tolerance': {'absolute': ABS_MS_TOLERANCE, 'relative': REL_MS_TOLERANCE},
        'strict_validation': 'shared canonical core.validate_schedule; all expected operations, machine IDs, durations, precedence, machine non-overlap, nonnegative times and makespan checked',
        'reporting': 'all 2160 candidates per model retained; feasible-candidate and any-feasible-instance rates; conditional gaps against released reference labels, which are not assumed optimal',
        'compute_caveat': 'equal candidate count does not imply equal FLOPs, wall time, training data or hardware; CPU times are recorded separately, SLJ batch time amortized over ten candidates',
        'checkpoint_training_provenance': 'released checkpoint identities verified by SHA256; exact original training seeds and complete per-checkpoint training manifests unavailable; no baseline retraining',
        'input_preflight': preflight_inputs(rows, modules),
        'pilot': {'matrix': PILOT_MATRIX, 'instance_id': canonical_hash(parse_matrix(PILOT_MATRIX)),
                  'excluded_from_scientific_population': True, 'candidates_per_model': 10},
        'provenance': provenance(source),
    }
    if protocol['pilot']['instance_id'] in protocol['instance_ids']:
        raise ValueError('Technical pilot must be excluded from the scientific test population')
    protocol['protocol_sha256'] = json_hash(protocol)
    write_json(protocol_path, protocol)
    return protocol


def verify(protocol_path):
    p = json.loads(Path(protocol_path).read_text())
    if json_hash({k: v for k, v in p.items() if k != 'protocol_sha256'}) != p['protocol_sha256']:
        raise ValueError('Baseline protocol self-hash mismatch')
    study, test_path, rows = checked_study(p['parent_study'])
    if study['protocol_sha256'] != p['parent_protocol_sha256'] or file_hash(test_path) != p['test_file_sha256']:
        raise ValueError('Parent study or test population drift')
    if p['instance_ids'] != [row['instance_id'] for row in rows] or p['models'] != list(MODELS) or p['candidates_per_instance'] != 10:
        raise ValueError('Frozen baseline budget/population mismatch')
    if p['provenance'] != provenance(p['provenance']['source_root']):
        raise ValueError('Baseline source, checkpoint, dependency or harness hash drift')
    return p, rows


def seed_instance(identity):
    import numpy as np
    import torch
    seed = int.from_bytes(hashlib.sha256(f'{identity}:42:0'.encode()).digest()[:8], 'big') % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def build_input(instance, modules, identity):
    import torch
    n, m = instance.n_jobs, instance.n_machines
    costs = torch.tensor([op.duration for op in instance.operations], dtype=torch.float32).reshape(n, m)
    machines = torch.tensor([op.machine for op in instance.operations], dtype=torch.long).reshape(n, m)
    io = modules['inout']
    x = io.standardize(io.extract_features(n, m, costs, machines), dim=0)
    if not torch.isfinite(x).all():
        raise ValueError('Original SLJ features are not finite')
    return dict(name=identity, path='canonical:frozen-matrix', j=n, m=m, shape=f'{n}x{m}',
                x=x, edge_index=io.graph_edges(n, m, machines).t().contiguous(),
                costs=costs, machines=machines, makespan=instance.reference_makespan)


def schedule_text(instance, end_times):
    rows = []
    for op, end in zip(instance.operations, end_times):
        start = float(end) - op.duration
        rows.append(f'Job {op.job} Operation {op.operation} on Machine {op.machine}: {start:g} + {op.duration:g} -> {float(end):g}')
    return '\n'.join(rows) + f'\nMakespan: {max(end_times):g}'


def replay_sequence(instance, decisions):
    n, m = instance.n_jobs, instance.n_machines
    if len(decisions) != n * m or sorted(decisions) != list(range(n * m)):
        raise ValueError('Decision trace must contain every operation exactly once')
    job_next, job_ready, machine_ready = [0] * n, [0] * n, [0] * m
    ends = [None] * (n * m)
    for action in decisions:
        op = instance.operations[action]
        if op.operation != job_next[op.job]:
            raise ValueError('Decision trace violates job precedence')
        end = max(job_ready[op.job], machine_ready[op.machine]) + op.duration
        job_ready[op.job] = machine_ready[op.machine] = end
        job_next[op.job] += 1
        ends[action] = end
    return schedule_text(instance, ends)


def recover_slj_matrix(instance, incomplete, decisions):
    n, m = instance.n_jobs, instance.n_machines
    if len(incomplete) != m or any(len(row) != n for row in incomplete):
        raise ValueError('Wrong SLJ machine-order matrix dimensions')
    flat = [int(a) for row in incomplete for a in row]
    missing = set(range(n * m)) - set(flat)
    if len(missing) != 1 or flat.count(-1) != 1 or len(set(a for a in flat if a != -1)) != n * m - 1:
        raise ValueError('SLJ omission is not a uniquely recoverable final operation')
    final = missing.pop()
    if final != decisions[-1]:
        raise ValueError('Recovered operation differs from original forced final action')
    complete = [[final if a == -1 else int(a) for a in row] for row in incomplete]
    for machine, row in enumerate(complete):
        if any(instance.operations[a].machine != machine for a in row):
            raise ValueError('SLJ matrix assigns an operation to the wrong machine')
        if row != [a for a in decisions if instance.operations[a].machine == machine]:
            raise ValueError('SLJ matrix and raw decisions disagree')
    return complete


def check_library_ms(library_ms, computed_ms):
    if not math.isfinite(float(library_ms)) or not math.isclose(float(library_ms), computed_ms,
                        abs_tol=ABS_MS_TOLERANCE, rel_tol=REL_MS_TOLERANCE):
        raise ValueError(f'Library makespan disagrees with reconstructed schedule: {library_ms} vs {computed_ms}')


@contextmanager
def trace_slj(module):
    original = module.JobShopStates.__schedule__
    calls = []
    def traced(state, jobs):
        calls.append(state.ops[state._bs_idx, jobs].detach().cpu().tolist())
        return original(state, jobs)
    module.JobShopStates.__schedule__ = traced
    try:
        yield calls
    finally:
        module.JobShopStates.__schedule__ = original


def slj_candidates(instance, ins, encoder, decoder, modules, count=10):
    module = modules['sampling']
    encoder.eval()
    decoder.eval()
    start = time.perf_counter()
    with trace_slj(module) as trace:
        sols, mss = module.sampling(ins, encoder, decoder, bs=count, device='cpu')
    elapsed = time.perf_counter() - start
    if len(trace) != len(instance.operations) or any(len(step) != count for step in trace):
        raise ValueError('Incomplete original SLJ decision trace')
    results = []
    for i in range(count):
        decisions = [int(step[i]) for step in trace]
        incomplete = sols[i].cpu().tolist()
        complete = recover_slj_matrix(instance, incomplete, decisions)
        output = replay_sequence(instance, decisions)
        validation = validate_schedule(instance, output).to_dict()
        library_ms = float(mss[i].item())
        check_library_ms(library_ms, validation['computed_makespan'])
        results.append(dict(candidate_index=i, decisions=decisions, output=output,
                            original_machine_order=incomplete, complete_machine_order=complete,
                            forced_final_operation=decisions[-1], library_makespan=library_ms,
                            library_makespan_difference=library_ms-validation['computed_makespan'],
                            validation=validation, timing={'batch_seconds': elapsed,
                            'amortized_seconds': elapsed/count, 'batch_candidates': count}))
    return results


def l2d_candidates(instance, ins, policy, modules, count=10):
    import numpy as np
    import torch
    policy.eval()
    policy.n_j, policy.n_m = instance.n_jobs, instance.n_machines
    env = modules['L2D.JSSP_Env'].SJSSP(n_j=instance.n_jobs, n_m=instance.n_machines)
    pool = modules['L2D.utils'].g_pool_cal(graph_pool_type='average',
            batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
            n_nodes=env.number_of_tasks, device='cpu')
    results = []
    for i in range(count):
        start = time.perf_counter()
        adj, fea, candidate, mask = env.reset((ins['costs'].numpy().copy(), ins['machines'].numpy().copy()+1))
        reward_sum = -float(env.max_endTime)
        done, decisions = False, []
        with torch.no_grad():
            while not done:
                pi, _ = policy(x=torch.from_numpy(np.copy(fea)), graph_pool=pool,
                    padded_nei=None, adj=torch.from_numpy(np.copy(adj)).to_sparse(),
                    candidate=torch.from_numpy(np.copy(candidate)).unsqueeze(0),
                    mask=torch.from_numpy(np.copy(mask)).unsqueeze(0))
                probabilities = pi.reshape(-1)
                if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                    raise ValueError('Invalid L2D probabilities')
                index = int(probabilities.multinomial(1, replacement=False).item())
                if mask[index]:
                    raise ValueError('L2D selected a masked job')
                action = int(candidate[index])
                decisions.append(action)
                adj, fea, reward, done, candidate, mask = env.step(action)
                reward_sum += float(reward)
        elapsed = time.perf_counter() - start
        if sorted(decisions) != list(range(len(instance.operations))):
            raise ValueError('L2D trace does not cover all operations')
        ends = env.temp1.reshape(-1).tolist()
        output = schedule_text(instance, ends)
        validation = validate_schedule(instance, output).to_dict()
        library_ms = float(env.posRewards) - reward_sum
        check_library_ms(library_ms, validation['computed_makespan'])
        check_library_ms(float(env.max_endTime), validation['computed_makespan'])
        for machine, row in enumerate(env.opIDsOnMchs.tolist()):
            if sorted(row) != [a for a, op in enumerate(instance.operations) if op.machine == machine]:
                raise ValueError('L2D environment machine order has wrong operation identities')
            for col, action in enumerate(row):
                expected_start = ends[action] - instance.operations[action].duration
                if expected_start != int(env.mchsStartTimes[machine, col]):
                    raise ValueError('L2D environment start/completion arrays disagree')
        results.append(dict(candidate_index=i, decisions=decisions, output=output,
                            complete_machine_order=env.opIDsOnMchs.tolist(),
                            machine_start_times=env.mchsStartTimes.tolist(), completion_times=env.temp1.tolist(),
                            library_makespan=library_ms, library_makespan_difference=library_ms-validation['computed_makespan'],
                            validation=validation, timing={'candidate_seconds': elapsed}))
    return results


def load_model(name, modules, source):
    import torch
    path = Path(source) / MODELS[name][0]
    if file_hash(path) != MODELS[name][1]:
        raise ValueError('Checkpoint changed before deserialization')
    if name.startswith('l2d'):
        policy = modules['L2D.actor_critic'].ActorCritic(n_j=0, n_m=0,
            num_layers=3, learn_eps=False, neighbor_pooling_type='sum', input_dim=2,
            hidden_dim=64, num_mlp_layers_feature_extract=2,
            num_mlp_layers_actor=2, hidden_dim_actor=32,
            num_mlp_layers_critic=2, hidden_dim_critic=32, device='cpu')
        policy.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
        policy.eval()
        return (policy,)
    state, decoder = torch.load(path, map_location='cpu')
    encoder = modules['PointerNet'].GATEncoder(15)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def revalidate_record(record, instance):
    """Reconstruct independently of stored output and validation on resume."""
    if 'original_machine_order' in record:
        matrix = recover_slj_matrix(instance, record['original_machine_order'], record['decisions'])
        if matrix != record['complete_machine_order']:
            raise ValueError('Stored SLJ repaired matrix drift')
        expected_output = replay_sequence(instance, record['decisions'])
    else:
        # Replaying the original environment is necessary: L2D uses left shift,
        # so append-only replay would silently change the policy's schedule.
        if sorted(record['decisions']) != list(range(len(instance.operations))):
            raise ValueError('Invalid L2D recorded decision identities')
        replay_sequence(instance, record['decisions'])  # Require legal ready-job decisions.
        import numpy as np
        modules = load_modules()
        env = modules['L2D.JSSP_Env'].SJSSP(n_j=instance.n_jobs, n_m=instance.n_machines)
        costs = np.array([op.duration for op in instance.operations], dtype=np.float32).reshape(instance.n_jobs, instance.n_machines)
        machines = np.array([op.machine + 1 for op in instance.operations], dtype=np.int64).reshape(instance.n_jobs, instance.n_machines)
        env.reset((costs, machines))
        for action in record['decisions']:
            env.step(action)
        if (env.temp1.tolist() != record['completion_times'] or
                env.opIDsOnMchs.tolist() != record['complete_machine_order'] or
                env.mchsStartTimes.tolist() != record['machine_start_times']):
            raise ValueError('Recorded L2D schedule disagrees with replay of original environment')
        ends = [x for row in record['completion_times'] for x in row]
        expected_output = schedule_text(instance, ends)
        for machine, row in enumerate(record['complete_machine_order']):
            if sorted(row) != [a for a, op in enumerate(instance.operations) if op.machine == machine]:
                raise ValueError('Recorded L2D machine-order identities drift')
            for col, action in enumerate(row):
                if record['machine_start_times'][machine][col] != ends[action] - instance.operations[action].duration:
                    raise ValueError('Recorded L2D machine-start arrays drift')
    if expected_output != record['output']:
        raise ValueError('Stored output differs from raw scheduling evidence')
    validation = validate_schedule(instance, expected_output).to_dict()
    # JSON roundtrip converts the validation tuple fields to lists.
    validation = json.loads(json.dumps(validation))
    if validation != record['validation']:
        raise ValueError('Stored strict validation differs from fresh validation')
    check_library_ms(record['library_makespan'], validation['computed_makespan'])
    return validation


def candidate_evidence(record):
    return json.loads(json.dumps({k: v for k, v in record.items()
            if k not in ('timing', 'feature_seconds', 'created_at_utc')}))


def persist_instance(path, candidates, instance, metadata):
    """Append missing slots after reproducing any interrupted instance exactly."""
    path = Path(path)
    prior = []
    if path.exists():
        prior = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(prior) > len(candidates):
        raise ValueError('Too many prior candidates')
    for i, saved in enumerate(prior):
        if saved['candidate_index'] != i:
            raise ValueError('Duplicate or nonsequential candidate slot')
        revalidate_record(saved, instance)
        expected = dict(candidates[i], **metadata)
        if candidate_evidence(saved) != candidate_evidence(expected):
            raise ValueError('Deterministic resumption changed candidate evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        for row in candidates[len(prior):]:
            record = dict(row, **metadata)
            # Store first, then validate. A failed check must not erase evidence.
            f.write(json.dumps(record, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            revalidate_record(json.loads(json.dumps(record)), instance)


def run(protocol_path, output_dir, *, pilot=False):
    output_dir = require_project_path(output_dir)
    initialize_runtime()
    p, rows = verify(protocol_path)
    modules = load_modules(p['provenance']['source_root'])
    purpose = 'EXCLUDED_TECHNICAL_PILOT' if pilot else 'SCIENTIFIC_FROZEN_216'
    if pilot:
        rows = [dict(matrix=p['pilot']['matrix'], instance_id=p['pilot']['instance_id'])]
    run_manifest = {'schema_version': 1, 'purpose': purpose, 'protocol_sha256': p['protocol_sha256'],
                    'instance_ids': [r['instance_id'] for r in rows], 'models': p['models'],
                    'candidates_per_instance': 10, 'device': 'cpu'}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'RUN.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != run_manifest:
        raise ValueError('Output folder belongs to a different run')
    write_json(manifest_path, run_manifest)
    for model in p['models']:
        loaded = None
        for index, row in enumerate(rows):
            instance = parse_matrix(row['matrix'])
            identity = canonical_hash(instance)
            path = output_dir / model / f'{identity}.jsonl'
            seed = seed_instance(identity)
            metadata = dict(model=model, instance_id=identity, seed=seed,
                            purpose=purpose, protocol_sha256=p['protocol_sha256'],
                            checkpoint_sha256=MODELS[model][1])
            if path.exists():
                existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if len(existing) == 10:
                    for i, record in enumerate(existing):
                        if record['candidate_index'] != i or any(record.get(k) != v for k, v in metadata.items()):
                            raise ValueError('Existing completed candidate identity mismatch')
                        revalidate_record(record, instance)
                    continue
            if loaded is None:
                loaded = load_model(model, modules, p['provenance']['source_root'])
                # Constructors consume RNG, so reseed after lazy loading too.
            seed_instance(identity)
            feature_start = time.perf_counter()
            ins = build_input(instance, modules, identity)
            feature_seconds = time.perf_counter() - feature_start
            if model.startswith('l2d'):
                candidates = l2d_candidates(instance, ins, *loaded, modules)
            else:
                candidates = slj_candidates(instance, ins, *loaded, modules)
            if len(candidates) != 10:
                raise ValueError('Candidate budget was not fulfilled')
            metadata['feature_seconds'] = feature_seconds
            persist_instance(path, candidates, instance, metadata)
            print(f'{purpose} {model} {index+1}/{len(rows)}: 10 candidates saved', flush=True)
    receipt = summarize(p, rows, output_dir, purpose)
    write_json(output_dir / 'COMPLETE.json', receipt)
    return receipt


def summarize(protocol, rows, output_dir, purpose):
    result = {'schema_version': 1, 'status': 'complete', 'purpose': purpose,
              'protocol_sha256': protocol['protocol_sha256'], 'completed_at_utc': now(),
              'instances': len(rows), 'candidates_per_instance': 10, 'models': {}}
    for model in protocol['models']:
        feasible, any_feasible, gaps, best_gaps, files = 0, 0, [], [], {}
        total_seconds, feature_seconds = 0.0, 0.0
        for row in rows:
            ins = parse_matrix(row['matrix'])
            path = Path(output_dir) / model / f"{row['instance_id']}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(records) != 10 or [r['candidate_index'] for r in records] != list(range(10)):
                raise ValueError('Completion requires exactly ten unique slots per instance/model')
            per_instance = []
            for record in records:
                if record['protocol_sha256'] != protocol['protocol_sha256'] or record['purpose'] != purpose or record['model'] != model or record['instance_id'] != row['instance_id'] or record['checkpoint_sha256'] != MODELS[model][1]:
                    raise ValueError('Raw record identity/protocol mismatch')
                validation = revalidate_record(record, ins)
                if validation['feasible']:
                    feasible += 1
                    if validation['gap_percent'] is not None:
                        gaps.append(validation['gap_percent'])
                        per_instance.append(validation['gap_percent'])
            any_feasible += int(any(r['validation']['feasible'] for r in records))
            if per_instance:
                best_gaps.append(min(per_instance))
            if model.startswith('l2d'):
                total_seconds += sum(r['timing']['candidate_seconds'] for r in records)
            else:
                total_seconds += records[0]['timing']['batch_seconds']
            feature_seconds += records[0]['feature_seconds']
            files[str(path.relative_to(output_dir))] = file_hash(path)
        denom = len(rows)*10
        result['models'][model] = dict(candidates=denom, feasible_candidates=feasible,
            feasible_candidate_percent=100*feasible/denom,
            any_feasible_instances=any_feasible,
            any_feasible_instance_percent=100*any_feasible/len(rows),
            conditional_mean_candidate_gap_percent=sum(gaps)/len(gaps) if gaps else None,
            conditional_mean_best_of_10_gap_percent=sum(best_gaps)/len(best_gaps) if best_gaps else None,
            scheduling_seconds=total_seconds, feature_seconds=feature_seconds,
            raw_file_sha256=files)
    result['receipt_sha256'] = json_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'verify', 'run'))
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.protocol, args.study, args.source)
    elif args.command == 'verify':
        initialize_runtime()
        result, _ = verify(args.protocol)
    else:
        if args.output_dir is None:
            parser.error('run requires --output-dir')
        result = run(args.protocol, args.output_dir, pilot=args.pilot)
    print(json.dumps({k: result[k] for k in ('status', 'protocol_sha256', 'purpose') if k in result}, indent=2))


if __name__ == '__main__':
    main()
