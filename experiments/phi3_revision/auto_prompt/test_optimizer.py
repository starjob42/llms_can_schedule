"""Engineering tests use synthetic aggregates and outputs, never model outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.phi3_revision.auto_prompt import optimizer as o


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class SyntheticProposer:
    """Explicit engineering-only proposer; no learned model is instantiated."""
    tokenizer = WordTokenizer()

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def propose(self, messages, seed):
        self.calls.append((messages, seed))
        raw = next(self.outputs)
        if isinstance(raw, Exception):
            raise raw
        return raw, {'purpose': 'ENGINEERING_ONLY', 'seed': seed}


def spec(text, order=0):
    if order == 0:
        rnd, slot = 0, 0
    else:
        rnd, slot = (order-1)//2+1, (order-1)%2+1
    parsed = o.parse_proposal(json.dumps({'instruction': text}), WordTokenizer(), [])
    return o.make_instruction(o.INSTRUCTION_IDS[order], parsed, rnd, slot, 0 if order == 0 else 1)


def feedback(instruction, feasible=10, gap=20.0):
    return dict(instruction_id=instruction['id'], instruction_sha256=instruction['text_sha256'],
        split='validation', instance_ids_sha256='e'*64, n_instances=72,
        candidate_count=3, n_candidates=216, feasible_candidates=feasible,
        conditional_mean_gap_percent=gap if feasible else None,
        error_counts={'missing_operation': 216-feasible}, protocol_sha256='a'*64,
        pilot_excluded=False)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.initial = spec('Provide every operation timing and the final makespan.')
        self.feedback = feedback(self.initial)

    def test_exact_ascii_instruction_preserved_and_uniqueness_normalized(self):
        text = 'Schedule operations; use {start} + {duration} -> {end}, gap% and M_0.'
        parsed = o.parse_proposal(json.dumps({'instruction': text}), WordTokenizer(), [])
        self.assertEqual(parsed['text'], text)
        self.assertEqual(parsed['text_sha256'], hashlib.sha256(text.encode()).hexdigest())
        with self.assertRaisesRegex(o.ProposalRejected, 'duplicate'):
            o.parse_proposal(json.dumps({'instruction': text.upper().replace(' ', '  ')}), WordTokenizer(), [text])

    def test_strict_json_schema_and_no_markdown(self):
        cases = [('```json\n{"instruction":"A long scheduling instruction."}\n```', 'invalid_json'),
                 ('{"instruction":"A valid long instruction.","other":1}', 'wrong_schema'),
                 ('{"instruction":"First long instruction.","instruction":"Second long instruction."}', 'invalid_json'),
                 ('["A valid long instruction."]', 'wrong_schema')]
        for raw, code in cases:
            with self.subTest(code=code, raw=raw):
                with self.assertRaisesRegex(o.ProposalRejected, code):
                    o.parse_proposal(raw, WordTokenizer(), [])

    def test_format_constraints_reject_unicode_controls_roles_and_edge_spaces(self):
        cases = [('Give a schedule with smart quotes \u201cnow\u201d.', 'non_ascii'),
                 ('Give a schedule.\nReport makespan.', 'not_single_line'),
                 ('Give a\tcomplete operation schedule.', 'invalid_character'),
                 ('<|assistant|> Give the operation schedule.', 'role_delimiter'),
                 ('Give the full schedule <|assistant.', 'role_delimiter'),
                 ('Give the full schedule with |>.', 'role_delimiter'),
                 (' Assistant: Give all operation times.', 'surrounding_whitespace'),
                 ('Assistant: Give all operation times.', 'role_delimiter'),
                 (' Give all operation times and makespan.', 'surrounding_whitespace'),
                 ('too short', 'character_limit')]
        for text, code in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(o.ProposalRejected, code):
                    o.parse_proposal(json.dumps({'instruction': text}), WordTokenizer(), [])
        with self.assertRaisesRegex(o.ProposalRejected, 'token_limit'):
            o.parse_proposal(json.dumps({'instruction': 'a'*129}), CharacterTokenizer(), [])

    def test_proposer_messages_allowlist_excludes_private_fields_and_outcomes(self):
        f = dict(self.feedback, test_output='SECRET_TEST_OUTPUT', raw_instances='SECRET_INSTANCE',
                 path='/private/test', instance_ids=['SECRET_CANONICAL_ID'])
        incumbent = dict(self.initial, output='SECRET_MODEL_OUTPUT')
        messages = o.build_proposer_messages(incumbent, f, 1, 1, 1)
        text = json.dumps(messages)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('/private/test', text)
        payload = json.JSONDecoder().raw_decode(messages[1]['content'])[0]
        self.assertEqual(set(payload['validation_feedback']), set(o.FEEDBACK_AGGREGATES))
        self.assertEqual(payload['incumbent_instruction'], self.initial['text'])
        self.assertTrue(payload['limits']['english_printable_ascii_only'])

    def test_feedback_rejects_invalid_denominator_nonfinite_gap_and_raw_error_details(self):
        for changes in ({'n_candidates': 215}, {'conditional_mean_gap_percent': float('nan')},
                        {'error_counts': {'Job 0 Operation 2 failed': 1}}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    o.feedback_projection(dict(self.feedback, **changes))

    def test_proposal_seed_exact_and_bounded(self):
        expected = int.from_bytes(hashlib.sha256(b'phi3_auto_prompt:42:2:1:4').digest()[:8], 'big') % (2**31-1)
        self.assertEqual(o.proposal_seed(2, 1, 4), expected)
        with self.assertRaises(ValueError):
            o.proposal_seed(1, 1, 5)

    def test_selection_prioritizes_feasibility_then_gap_then_creation(self):
        candidates = [self.initial, spec('Minimize makespan and report every start and end.', 1),
                      spec('Give a feasible complete schedule and its makespan.', 2)]
        scores = {s['id']: feedback(s, 10, 1) for s in candidates}
        scores[candidates[1]['id']] = feedback(candidates[1], 11, 100)
        self.assertEqual(o.choose_incumbent(candidates, scores)['id'], candidates[1]['id'])
        scores[candidates[2]['id']] = feedback(candidates[2], 11, 50)
        self.assertEqual(o.choose_incumbent(candidates, scores)['id'], candidates[2]['id'])
        scores[candidates[1]['id']] = feedback(candidates[1], 11, 50)
        self.assertEqual(o.choose_incumbent(candidates, scores)['id'], candidates[1]['id'])
        scores = {s['id']: feedback(s, 0) for s in candidates}
        self.assertEqual(o.choose_incumbent(candidates, scores)['id'], self.initial['id'])

    def test_selection_rejects_test_feedback_and_unpaired_populations(self):
        second = spec('Find the complete operation schedule and makespan.', 1)
        scores = {s['id']: feedback(s) for s in [self.initial, second]}
        scores[second['id']]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            o.choose_incumbent([self.initial, second], scores)
        scores[second['id']]['split'] = 'validation'
        scores[second['id']]['instance_ids_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'different validation populations'):
            o.choose_incumbent([self.initial, second], scores)

    def test_all_attempts_saved_and_accepted_attempt_resumes_without_generation(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_attempts_') as tmp:
            p = {'protocol_sha256': 'a'*64, 'output_dir': tmp}
            outputs = ['invalid', json.dumps({'instruction': self.initial['text']}),
                       json.dumps({'instruction': 'Schedule all operations with \u201ccorrect\u201d times.'}),
                       json.dumps({'instruction': 'Respect every precedence and machine; report all times and makespan.'})]
            engine = SyntheticProposer(outputs)
            accepted = o.propose_instruction(p, self.initial, self.feedback,
                              [self.initial], 1, 1, engine)
            self.assertEqual(accepted['origin']['attempt'], 4)
            self.assertEqual(len(engine.calls), 4)
            attempts = sorted(Path(tmp).glob('proposals/*/attempt_*.json'))
            self.assertEqual(len(attempts), 4)
            for index, path in enumerate(attempts, 1):
                row = json.loads(path.read_text())
                o.check_hash(row, 'attempt_sha256')
                self.assertEqual(row['seed'], o.proposal_seed(1, 1, index))
                self.assertEqual(row['accepted'], index == 4)
            no_generation = SyntheticProposer([])
            resumed = o.propose_instruction(p, self.initial, self.feedback,
                             [self.initial], 1, 1, no_generation)
            self.assertEqual(resumed, accepted)
            self.assertFalse(no_generation.calls)

    def test_four_failed_formats_fail_closed_and_no_manual_fallback(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_closed_') as tmp:
            p = {'protocol_sha256': 'a'*64, 'output_dir': tmp}
            engine = SyntheticProposer(['invalid']*4)
            with self.assertRaisesRegex(RuntimeError, 'four fixed attempts'):
                o.propose_instruction(p, self.initial, self.feedback, [self.initial], 1, 1, engine)
            self.assertEqual(len(engine.calls), 4)
            self.assertFalse((Path(tmp)/'instructions'/'auto_r1_s1.json').exists())
            self.assertEqual(len(list(Path(tmp).glob('proposals/*/attempt_*.json'))), 4)

    def test_generator_failure_preserved_and_never_automatically_retried(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_generation_failure_') as tmp:
            p = {'protocol_sha256': 'a'*64, 'output_dir': tmp}
            engine = SyntheticProposer([RuntimeError('synthetic excluded error')])
            with self.assertRaisesRegex(RuntimeError, 'synthetic excluded'):
                o.propose_instruction(p, self.initial, self.feedback, [self.initial], 1, 1, engine)
            failure = json.loads(next(Path(tmp).glob('proposals/*/*.failure.json')).read_text())
            o.check_hash(failure, 'failure_sha256')
            with self.assertRaisesRegex(RuntimeError, 'refusing automatic retry'):
                o.propose_instruction(p, self.initial, self.feedback, [self.initial], 1, 1, SyntheticProposer([]))

    def full_selection_fixture(self, tmp):
        instructions = [spec(f'Provide every operation time and makespan, instruction variant {i}.', i) for i in range(5)]
        scores = {s['id']: feedback(s, 100+i) for i, s in enumerate(instructions)}
        receipts = {s['id']: {key: str(i+1)*64 for key in ('evaluation_manifest_sha256',
            'candidates_sha256', 'complete_sha256', 'instruction_manifest_sha256')} for i, s in enumerate(instructions)}
        return {'protocol_sha256': 'a'*64, 'output_dir': tmp}, instructions, scores, receipts

    def test_final_selection_frozen_once_requires_exact_1080_validation_slots(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_selection_') as tmp:
            p, instructions, scores, receipts = self.full_selection_fixture(tmp)
            selected = o.freeze_selection(p, instructions, scores, receipts)
            o.check_hash(selected, 'selection_sha256')
            self.assertEqual(selected['selected_instruction_id'], 'auto_r2_s2')
            before = (Path(tmp)/'SELECTED.json').read_bytes()
            self.assertEqual(o.freeze_selection(p, instructions, scores, receipts), selected)
            self.assertEqual(before, (Path(tmp)/'SELECTED.json').read_bytes())
            bad = copy.deepcopy(scores)
            bad['auto_0']['n_candidates'] = 215
            with self.assertRaisesRegex(ValueError, '1080'):
                o.freeze_selection(p, instructions, bad, receipts)

    def test_final_selection_rejects_pilot_budget_and_test_before_selection(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_selection_guard_') as tmp:
            p, instructions, scores, receipts = self.full_selection_fixture(tmp)
            scores['auto_0']['pilot_excluded'] = True
            with self.assertRaisesRegex(ValueError, 'never pilots'):
                o.freeze_selection(p, instructions, scores, receipts)
            scores['auto_0']['pilot_excluded'] = False
            (Path(tmp)/'evaluations'/'test').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'before automatic instruction selection'):
                o.freeze_selection(p, instructions, scores, receipts)
            self.assertFalse((Path(tmp)/'SELECTED.json').exists())

    def test_primary_ready_requires_manager_and_actual_complete_final_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_primary_') as tmp:
            runs = Path(tmp)/'runs'
            p = {'runs_dir': str(runs), 'adapter_path': str(runs/'main'/'checkpoint-1844'),
                 'parent_protocol_sha256': 'p'*64}
            self.assertFalse(o.primary_ready(p))
            o.write_json(runs/'manager'/'COMPLETE', {'status': 'complete', 'study_protocol_sha256': 'p'*64})
            o.write_json(runs/'main'/'training_complete.json', {'global_step': 1844, 'pilot': False})
            o.write_json(runs/'main'/'run_manifest.json', {'status': 'complete', 'config': {'pilot': False}})
            with self.assertRaisesRegex(ValueError, 'complete 1844-step marker'):
                o.primary_ready(p)
            checkpoint = Path(p['adapter_path'])
            o.write_json(checkpoint/'experiment_checkpoint.json', {'global_step': 1844})
            o.write_json(checkpoint/'adapter_config.json', {})
            (checkpoint/'adapter_model.safetensors').write_bytes(b'UNIT_TEST_ONLY')
            self.assertTrue(o.primary_ready(p))
            o.write_json(runs/'main'/'training_complete.json', {'global_step': 461, 'pilot': False})
            with self.assertRaisesRegex(ValueError, '1844-step production'):
                o.primary_ready(p)

    def test_idle_check_waits_and_rechecks_primary_without_loading_model(self):
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_wait_') as tmp:
            p = {'output_dir': tmp}
            with patch.object(o, 'primary_ready', return_value=True) as ready, \
                 patch.object(o, 'verified_idle_device', side_effect=[None, {'uuid': o.SEACOVE_UUID}]) as inventory, \
                 patch.object(o.time, 'sleep') as sleep:
                device = o.wait_idle(p, phase='synthetic', poll_seconds=1)
            self.assertEqual(device['uuid'], o.SEACOVE_UUID)
            self.assertEqual(inventory.call_count, 2)
            self.assertEqual(ready.call_count, 2)
            sleep.assert_called_once_with(1)

    def test_scientific_runner_never_claims_gpu_before_primary_complete(self):
        from experiments.phi3_revision.auto_prompt import protocol
        class EndExcludedWait(RuntimeError):
            pass
        with tempfile.TemporaryDirectory(dir=o.HERE, prefix='test_before_primary_') as tmp:
            p = {'output_dir': tmp, 'protocol_sha256': 'a'*64}
            with patch.object(protocol, 'load_protocol', return_value=p), \
                 patch.object(o, 'primary_ready', return_value=False), \
                 patch.object(o.time, 'sleep', side_effect=EndExcludedWait), \
                 patch.object(o, 'device_lease') as lease, \
                 patch.object(o, 'verified_idle_device') as inventory:
                with self.assertRaises(EndExcludedWait):
                    o.optimize(Path(tmp)/'unused.json', execute=True, poll_seconds=1)
            lease.assert_not_called()
            inventory.assert_not_called()
            status = json.loads((Path(tmp)/'STATUS.json').read_text())
            self.assertEqual(status['status'], 'waiting_for_primary_complete')


if __name__ == '__main__':
    unittest.main()
