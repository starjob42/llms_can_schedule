"""Selection, label blindness, and split-integrity tests for the frozen study."""
import unittest

try:
    from .freeze_study import candidate, guard_overlap, stratified_selection, validate_partition
except ImportError:
    from freeze_study import candidate, guard_overlap, stratified_selection, validate_partition


def row(shape, identifier, source_index):
    jobs, machines = shape
    matrix = f"{jobs} {machines}\n" + "\n".join(
        " ".join(f"{machine} {identifier + job + machine + 1}" for machine in range(machines))
        for job in range(jobs)) + "\n99999\n"
    return candidate({"matrix": matrix, "output": f"arbitrary solver label {identifier}"}, source_index)


class FrozenStudyTests(unittest.TestCase):
    def setUp(self):
        self.shapes = {(2, 2), (3, 2)}
        self.pool = [row(shape, identifier, index + offset) for shape, offset in
                     [((2, 2), 0), ((3, 2), 100)]
                     for index, identifier in enumerate(range(10, 90, 10))]

    def test_stratification_is_deterministic_and_order_independent(self):
        selected = stratified_selection(self.pool, self.shapes, 3, salt="test")
        reversed_selected = stratified_selection(list(reversed(self.pool)), reversed(sorted(self.shapes)), 3, salt="test")
        self.assertEqual([item.instance_id for item in selected], [item.instance_id for item in reversed_selected])
        self.assertEqual(len(selected), 6)
        self.assertEqual([item.shape for item in selected].count((2, 2)), 3)
        self.assertEqual([item.shape for item in selected].count((3, 2)), 3)
        changed_salt = stratified_selection(self.pool, self.shapes, 3, salt="prompt_validation")
        self.assertNotEqual([item.instance_id for item in selected], [item.instance_id for item in changed_salt])

    def test_selection_ignores_reference_labels_and_output_text(self):
        changed = []
        for item in self.pool:
            data = dict(item.record, output="invalid output, deliberately unrelated to outcomes")
            data["matrix"] = data["matrix"].replace("\n99999\n", "\nnot even a numeric label\n")
            changed.append(candidate(data, item.source_index))
        first = stratified_selection(self.pool, self.shapes, 3, salt="test")
        second = stratified_selection(changed, self.shapes, 3, salt="test")
        self.assertEqual([x.instance_id for x in first], [x.instance_id for x in second])

    def test_missing_or_small_shape_and_duplicate_instance_fail(self):
        for pool, shapes in [(self.pool[:2], self.shapes), (self.pool, self.shapes | {(4, 4)}),
                             (self.pool + [self.pool[0]], self.shapes)]:
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                stratified_selection(pool, shapes, 3, salt="test")

    def test_partition_exhaustiveness_uniqueness_and_overlap(self):
        validate_partition([0, 2, 3], [1, 4], 5)
        for train, validation in [([0, 2, 3], [1, 3]), ([0, 2], [1, 4]),
                                  ([0, 0, 2, 3], [1, 4]), ([0, 2, 3], [1, 5]),
                                  ([False, 2, 3], [1, 4])]:
            with self.subTest(train=train, validation=validation), self.assertRaises(ValueError):
                validate_partition(train, validation, 5)

    def test_canonical_overlap_is_rejected_across_every_split(self):
        test, validation, train = self.pool[0], self.pool[1], self.pool[2]
        guard_overlap([test], [validation], [train.instance_id], [validation.source_index])
        for tests, vals, training, allowed in [
            ([test], [test], [], [test.source_index]),
            ([test], [validation], [test.instance_id], [validation.source_index]),
            ([test], [validation], [validation.instance_id], [validation.source_index]),
            ([test], [validation], [], [train.source_index]),
            ([test, test], [validation], [], [validation.source_index]),
        ]:
            with self.subTest(training=training), self.assertRaises(ValueError):
                guard_overlap(tests, vals, training, allowed)

    def test_stored_identity_cannot_override_problem_content(self):
        with self.assertRaisesRegex(ValueError, "instance_id"):
            candidate(dict(self.pool[0].record, instance_id="forged"), 0)


if __name__ == "__main__":
    unittest.main()
