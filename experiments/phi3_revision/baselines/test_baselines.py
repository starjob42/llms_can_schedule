"""Focused scheduler correctness tests. No learned policy outcomes are used."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.phi3_revision.baselines import run as b

try:
    import torch
    import torch_geometric
    LEGACY_AVAILABLE = torch.__version__.startswith('1.13.')
except ImportError:
    LEGACY_AVAILABLE = False


@unittest.skipUnless(LEGACY_AVAILABLE, 'Use the original SLJ Python environment')
class BaselineCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.initialize_runtime()
        cls.modules = b.load_modules()

    def setUp(self):
        self.instance = b.parse_matrix(b.PILOT_MATRIX)
        self.identity = b.canonical_hash(self.instance)
        self.ins = b.build_input(self.instance, self.modules, self.identity)

    def fake_encoder(self):
        class Encoder(torch.nn.Module):
            def forward(self, x, edge_index=None):
                return torch.zeros((x.shape[0], 1))
        return Encoder()

    def fake_decoder(self, prefer_first=False):
        class Decoder(torch.nn.Module):
            def forward(self, embed, state):
                n = state.shape[1]
                scores = torch.zeros((state.shape[0], n))
                if prefer_first:
                    scores -= 1000*torch.arange(n)[None, :]
                return scores
        return Decoder()

    def fake_l2d(self):
        class Policy(torch.nn.Module):
            def forward(self, *, mask, **kwargs):
                probs = torch.zeros(mask.shape)
                for i in range(mask.shape[0]):
                    first = int((~mask[i]).nonzero()[0].item())
                    probs[i, first] = 1
                return probs.unsqueeze(-1), torch.tensor(0.)
        return Policy()

    def fake_slj_candidates(self, count=10):
        return b.slj_candidates(self.instance, self.ins, self.fake_encoder(),
                                self.fake_decoder(), self.modules, count=count)

    def test_known_small_slj_append_and_forced_last_operation(self):
        records = b.slj_candidates(self.instance, self.ins, self.fake_encoder(),
                    self.fake_decoder(prefer_first=True), self.modules, count=10)
        self.assertEqual(len(records), 10)
        for r in records:
            self.assertEqual(r['decisions'], [0, 1, 2, 3])
            self.assertEqual(r['forced_final_operation'], 3)
            self.assertEqual(sum(row.count(-1) for row in r['original_machine_order']), 1)
            self.assertEqual(r['complete_machine_order'], [[0, 3], [1, 2]])
            self.assertTrue(r['validation']['feasible'])
            self.assertEqual(r['validation']['computed_makespan'], 11)
            self.assertAlmostEqual(r['library_makespan'], 11, places=4)

    def test_tracing_preserves_original_sampling_and_restores_method(self):
        module = self.modules['sampling']
        original_method = module.JobShopStates.__schedule__
        encoder, decoder = self.fake_encoder(), self.fake_decoder()
        b.seed_instance(self.identity)
        old_sol, old_ms = module.sampling(self.ins, encoder, decoder, bs=10)
        b.seed_instance(self.identity)
        with b.trace_slj(module) as decisions:
            new_sol, new_ms = module.sampling(self.ins, encoder, decoder, bs=10)
        self.assertIs(module.JobShopStates.__schedule__, original_method)
        self.assertTrue(torch.equal(old_sol, new_sol))
        self.assertTrue(torch.equal(old_ms, new_ms))
        self.assertEqual(len(decisions), 4)
        self.assertEqual(len(decisions[-1]), 10)

    def test_tracing_restores_method_after_exception(self):
        module = self.modules['sampling']
        original = module.JobShopStates.__schedule__
        with self.assertRaises(RuntimeError):
            with b.trace_slj(module):
                raise RuntimeError('test interruption')
        self.assertIs(module.JobShopStates.__schedule__, original)

    def test_unique_forced_repair_rejects_other_omissions(self):
        with self.assertRaisesRegex(ValueError, 'uniquely recoverable'):
            b.recover_slj_matrix(self.instance, [[0, -1], [1, -1]], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'forced final'):
            b.recover_slj_matrix(self.instance, [[0, -1], [1, 2]], [0, 3, 1, 2])
        with self.assertRaisesRegex(ValueError, 'wrong machine'):
            b.recover_slj_matrix(self.instance, [[1, -1], [0, 2]], [0, 1, 2, 3])

    def test_l2d_machine_plus_one_reconstruction_matches_reward_makespan(self):
        records = b.l2d_candidates(self.instance, self.ins, self.fake_l2d(), self.modules)
        self.assertEqual(len(records), 10)
        self.assertEqual(self.ins['machines'].tolist(), [[0, 1], [1, 0]])
        for r in records:
            self.assertEqual(r['decisions'], [0, 1, 2, 3])
            self.assertEqual(r['complete_machine_order'], [[0, 3], [2, 1]])
            self.assertEqual(r['completion_times'], [[3, 5], [2, 7]])
            self.assertEqual(r['library_makespan'], 7)
            self.assertTrue(r['validation']['feasible'])
            self.assertFalse(self.fake_l2d().eval().training)
            b.revalidate_record(json.loads(json.dumps(r)), self.instance)

    def test_append_and_permissible_left_shift_are_distinct(self):
        ins = b.parse_matrix('2 2\n1 10 0 5\n0 3 1 1\n15\n')
        data = b.build_input(ins, self.modules, b.canonical_hash(ins))
        l2d = b.l2d_candidates(ins, data, self.fake_l2d(), self.modules, count=1)[0]
        slj = b.slj_candidates(ins, data, self.fake_encoder(),
                     self.fake_decoder(prefer_first=True), self.modules, count=1)[0]
        self.assertEqual(l2d['decisions'], slj['decisions'])
        self.assertEqual(l2d['validation']['computed_makespan'], 15)
        self.assertEqual(slj['validation']['computed_makespan'], 19)

    def test_makespan_mismatch_and_precedence_trace_fail(self):
        with self.assertRaisesRegex(ValueError, 'makespan disagrees'):
            b.check_library_ms(15, 7)
        with self.assertRaisesRegex(ValueError, 'precedence'):
            b.replay_sequence(self.instance, [1, 0, 2, 3])

    def test_revalidation_rejects_output_and_raw_evidence_tampering(self):
        r = json.loads(json.dumps(self.fake_slj_candidates()[0]))
        bad = copy.deepcopy(r)
        bad['output'] = bad['output'].replace('Makespan:', 'Makespan: 999 #')
        with self.assertRaisesRegex(ValueError, 'raw scheduling evidence'):
            b.revalidate_record(bad, self.instance)
        bad = copy.deepcopy(r)
        bad['validation']['feasible'] = False
        with self.assertRaisesRegex(ValueError, 'fresh validation'):
            b.revalidate_record(bad, self.instance)
        l2d = json.loads(json.dumps(b.l2d_candidates(self.instance, self.ins,
                        self.fake_l2d(), self.modules, count=1)[0]))
        l2d['decisions'] = [0, 2, 3, 1]  # Legal trace, different left-shift result.
        # This particular toy may happen to have the same result. Use reversed
        # job priority, which changes which operation occupies the early gap.
        l2d['decisions'] = [2, 3, 0, 1]
        with self.assertRaisesRegex(ValueError, 'replay of original environment'):
            b.revalidate_record(l2d, self.instance)

    def test_seeded_sampling_repeatable_and_eval_mode_explicit(self):
        encoder, decoder = self.fake_encoder(), self.fake_decoder()
        encoder.train()
        decoder.train()
        b.seed_instance(self.identity)
        first = b.slj_candidates(self.instance, self.ins, encoder, decoder, self.modules)
        torch.rand(100)
        b.seed_instance(self.identity)
        second = b.slj_candidates(self.instance, self.ins, encoder, decoder, self.modules)
        self.assertEqual([b.candidate_evidence(r) for r in first],
                         [b.candidate_evidence(r) for r in second])
        self.assertFalse(encoder.training)
        self.assertFalse(decoder.training)

    def test_partial_resume_fills_exact_slots_and_completed_resume_idempotent(self):
        b.seed_instance(self.identity)
        records = self.fake_slj_candidates()
        metadata = {'model': 'test_only', 'instance_id': self.identity, 'seed': 123,
                    'purpose': 'UNIT_TEST_ONLY', 'protocol_sha256': 'test',
                    'checkpoint_sha256': 'test', 'feature_seconds': .001}
        with tempfile.TemporaryDirectory(dir=b.HERE, prefix='test_resume_') as tmp:
            path = Path(tmp) / 'candidates.jsonl'
            path.write_text(''.join(json.dumps(dict(r, **metadata))+'\n' for r in records[:3]))
            b.persist_instance(path, records, self.instance, metadata)
            self.assertEqual(len(path.read_text().splitlines()), 10)
            before = path.read_bytes()
            b.persist_instance(path, records, self.instance, metadata)
            self.assertEqual(path.read_bytes(), before)
            altered = copy.deepcopy(records)
            altered[0]['decisions'] = [0, 1, 2, 3] if records[0]['decisions'] != [0, 1, 2, 3] else [2, 3, 0, 1]
            with self.assertRaisesRegex(ValueError, 'resumption changed'):
                b.persist_instance(path, altered, self.instance, metadata)

    def test_parent_study_exact_population_and_pilot_disjoint(self):
        study, path, rows = b.checked_study(b.DEFAULT_STUDY)
        self.assertEqual(len(rows), 216)
        self.assertEqual([r['instance_id'] for r in rows], study['test_instance_ids'])
        self.assertNotIn(self.identity, study['test_instance_ids'])

    def test_protocol_tamper_rejected_before_loading_sources(self):
        with tempfile.TemporaryDirectory(dir=b.HERE, prefix='test_protocol_') as tmp:
            path = Path(tmp) / 'protocol.json'
            value = {'protocol_sha256': 'incorrect', 'candidates_per_instance': 9}
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'self-hash'):
                b.verify(path)


if __name__ == '__main__':
    unittest.main()
