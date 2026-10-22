"""Dependency-free, strict JSSP evaluation shared by revision experiments.

Instance identity contains routes and durations, never labels or prompt wording.
All feasibility checks use the canonical instance rather than the model's claims.
Compact machine-first descriptions omit operation indices and cannot by themselves
specify job precedence: use the accompanying matrix as ground truth instead.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import statistics
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class Operation:
    job: int
    operation: int
    machine: int
    duration: float


@dataclass(frozen=True)
class Instance:
    n_jobs: int
    n_machines: int
    operations: tuple[Operation, ...]
    reference_makespan: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(sorted(self.operations)))
        if self.n_jobs <= 0 or self.n_machines <= 0:
            raise ValueError("Instance dimensions must be positive")
        keys = [(o.job, o.operation) for o in self.operations]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate instance operation")
        jobs: dict[int, list[int]] = defaultdict(list)
        for op in self.operations:
            if not (0 <= op.job < self.n_jobs and 0 <= op.machine < self.n_machines):
                raise ValueError("Instance job/machine identifier out of range")
            if op.operation < 0 or not math.isfinite(op.duration) or op.duration <= 0:
                raise ValueError("Instance operation indices/durations must be valid and positive")
            jobs[op.job].append(op.operation)
        if set(jobs) != set(range(self.n_jobs)):
            raise ValueError("Missing job in instance")
        for indices in jobs.values():
            if indices != list(range(len(indices))):
                raise ValueError("Job operations must be consecutive and start at zero")
        if self.reference_makespan is not None and (
            not math.isfinite(self.reference_makespan) or self.reference_makespan <= 0
        ):
            raise ValueError("Reference makespan must be finite and positive")

    def to_dict(self) -> dict:
        return asdict(self)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FLAGS = re.IGNORECASE


def _number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite number")
    return result


def parse_matrix(text: str) -> Instance:
    """Parse standard zero-based JSSP matrix, plus optional reference label.

    The first line is ``n_jobs n_machines``; each following job row contains
    exactly n_machines ``machine duration`` pairs in precedence order. Existing
    corpus files optionally append a reference makespan and machine-order rows;
    those machine-order rows are labels and are deliberately excluded from ID.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"\d+\s+\d+", lines[0]):
        raise ValueError("Expected matrix header: n_jobs n_machines")
    n_jobs, n_machines = map(int, lines[0].split())
    if n_jobs <= 0 or n_machines <= 0 or len(lines) < n_jobs + 1:
        raise ValueError("Incomplete or invalid matrix")
    operations = []
    for job, line in enumerate(lines[1:n_jobs + 1]):
        tokens = line.split()
        if len(tokens) != 2 * n_machines:
            raise ValueError(f"Job {job} requires exactly {n_machines} machine-duration pairs")
        for operation in range(n_machines):
            machine_text, duration_text = tokens[2 * operation:2 * operation + 2]
            if not re.fullmatch(r"\d+", machine_text) or not re.fullmatch(NUMBER, duration_text):
                raise ValueError(f"Invalid matrix token for job {job}")
            operations.append(Operation(job, operation, int(machine_text), _number(duration_text)))
    reference = None
    if len(lines) > n_jobs + 1:
        if not re.fullmatch(NUMBER, lines[n_jobs + 1]):
            raise ValueError("First matrix trailer must be a numeric reference makespan")
        reference = _number(lines[n_jobs + 1])
    return Instance(n_jobs, n_machines, tuple(operations), reference)


def parse_instance(text: str) -> Instance:
    """Read paper job/machine formats, explicit flat rows, or compact job format.

    Machine-centric row order does not determine precedence. The explicit
    operation indices in the paper do; compact ``M0: J2: 5 ...`` alone does not.
    """
    if re.match(r"^\s*\d+\s+\d+\s*\n", text):
        return parse_matrix(text)
    if re.search(r"(?m)^\s*M\d+\s*:\s*$", text, FLAGS):
        raise ValueError("Compact machine-first input omits precedence; provide its matrix")
    dimensions = re.search(
        r"(\d+)\s+Jobs\b[^\n]*?(\d+)\s+Machines\b", text, FLAGS
    )
    job_context = machine_context = None
    compact_job = None
    compact_next_operation: dict[int, int] = defaultdict(int)
    operations: list[Operation] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"J(\d+)\s*:\s*", line, FLAGS)
        if match:
            compact_job = int(match[1])
            job_context = machine_context = None
            continue
        if compact_job is not None:
            pairs = list(re.finditer(rf"M(\d+)\s*:\s*({NUMBER})", line, FLAGS))
            remainder = re.sub(rf"M\d+\s*:\s*{NUMBER}", "", line, flags=FLAGS).strip()
            if pairs and not remainder:
                for pair in pairs:
                    index = compact_next_operation[compact_job]
                    operations.append(Operation(compact_job, index, int(pair[1]), _number(pair[2])))
                    compact_next_operation[compact_job] += 1
                continue
            if re.match(r"M\d+\s*:", line, FLAGS):
                raise ValueError(f"Malformed compact operation: {line}")
        match = re.fullmatch(r"Job\s+(\d+)\s+consists\s+of\s+the\s+following\s+Operations\s*:", line, FLAGS)
        if match:
            job_context, machine_context, compact_job = int(match[1]), None, None
            continue
        match = re.fullmatch(r"Machine\s+(\d+)\s+is\s+used\s+for\s+the\s+following\s+Operations\s*:", line, FLAGS)
        if match:
            machine_context, job_context, compact_job = int(match[1]), None, None
            continue
        match = re.fullmatch(
            rf"Machine\s+(\d+)\s+processes\s+Job\s+(\d+)\s+Operation\s+(\d+)\s+for\s+({NUMBER})\s+mins?\.?", line, FLAGS
        )
        if match:
            operations.append(Operation(int(match[2]), int(match[3]), int(match[1]), _number(match[4])))
            continue
        match = re.fullmatch(
            rf"Job\s+(\d+)\s+Operation\s+(\d+)\s+on\s+Machine\s+(\d+)\s+duration\s+({NUMBER})\s+mins?\.?", line, FLAGS
        )
        if match:
            operations.append(Operation(int(match[1]), int(match[2]), int(match[3]), _number(match[4])))
            continue
        match = re.fullmatch(rf"Job\s+(\d+)\s+Operation\s+(\d+)\s+duration\s+({NUMBER})\s+mins?\.?", line, FLAGS)
        if match:
            if machine_context is None:
                raise ValueError("Operation without machine context")
            operations.append(Operation(int(match[1]), int(match[2]), machine_context, _number(match[3])))
            continue
        match = re.fullmatch(rf"Operation\s+(\d+)\s+on\s+Machine\s+(\d+)\s+duration\s+({NUMBER})\s+mins?\.?", line, FLAGS)
        if match:
            if job_context is None:
                raise ValueError("Operation without job context")
            operations.append(Operation(job_context, int(match[1]), int(match[2]), _number(match[3])))
            continue
        if re.match(r"(?:Job\s+\d+\s+)?Operation\s+\d+\b|Machine\s+\d+\s+processes\b", line, FLAGS):
            raise ValueError(f"Malformed instance operation: {line}")
    if not operations:
        raise ValueError("No operations found")
    n_jobs = int(dimensions[1]) if dimensions else max(op.job for op in operations) + 1
    n_machines = int(dimensions[2]) if dimensions else max(op.machine for op in operations) + 1
    return Instance(n_jobs, n_machines, tuple(operations))


def canonical_hash(instance: Instance) -> str:
    """SHA-256 of dimensions and ordered (job, operation, machine, duration)."""
    def numeric(value: float) -> int | float:
        return int(value) if float(value).is_integer() else float(value)
    payload = [instance.n_jobs, instance.n_machines,
               [[op.job, op.operation, op.machine, numeric(op.duration)] for op in instance.operations]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ScheduledOperation:
    job: int
    operation: int
    machine: int
    start: float
    duration: float
    end: float


@dataclass(frozen=True)
class ValidationResult:
    feasible: bool
    errors: tuple[str, ...]
    details: tuple[str, ...]
    operations: tuple[ScheduledOperation, ...]
    reported_makespan: float | None
    computed_makespan: float | None
    gap_percent: float | None

    def to_dict(self) -> dict:
        return asdict(self)


_LONG_SCHEDULE = re.compile(
    rf"\bJob\s+([+-]?\d+)\s+Operation\s+([+-]?\d+)\s+on\s+Machine\s+([+-]?\d+)\s*:\s*"
    rf"({NUMBER})\s*\+\s*({NUMBER})\s*(?:->|→)\s*({NUMBER})(?![\w.])", FLAGS)
_SHORT_SCHEDULE = re.compile(
    rf"\bJ([+-]?\d+)\s*-\s*M([+-]?\d+)\s*:\s*({NUMBER})\s*\+\s*({NUMBER})\s*(?:->|→)\s*({NUMBER})(?![\w.])", FLAGS)
_ROW_START = re.compile(r"\b(?:Job\s+[+-]?\d+\s+Operation\s+[+-]?\d+|J[+-]?\d+\s*-\s*M[+-]?\d+\s*:)", FLAGS)


def validate_schedule(
    instance: Instance, text: str, reference_makespan: float | None = None,
    *, require_makespan: bool = True, tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate every expected operation and all JSSP constraints.

    A candidate is infeasible if any operation is missing, duplicated, extra,
    malformed, assigned to the wrong machine, or has inconsistent timing. Idle
    time and touching half-open intervals are allowed. Gap is defined only for
    a feasible candidate with a positive reference label (not assumed optimal).
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    reference = instance.reference_makespan if reference_makespan is None else reference_makespan
    if reference is not None and (not math.isfinite(reference) or reference <= 0):
        raise ValueError("Reference makespan must be finite and positive")
    errors: list[str] = []
    details: list[str] = []
    def fail(code: str, detail: str) -> None:
        if code not in errors:
            errors.append(code)
        details.append(detail)
    expected = {(op.job, op.operation): op for op in instance.operations}
    by_job_machine: dict[tuple[int, int], list[Operation]] = defaultdict(list)
    for op in instance.operations:
        by_job_machine[op.job, op.machine].append(op)
    parsed_with_position: list[tuple[int, ScheduledOperation]] = []
    covered_starts = set()
    for match in _LONG_SCHEDULE.finditer(text):
        covered_starts.add(match.start())
        try:
            timings = tuple(map(_number, match.group(4, 5, 6)))
        except ValueError:
            fail("nonfinite_number", "Operation timing is not finite")
            continue
        parsed_with_position.append((match.start(), ScheduledOperation(
            int(match[1]), int(match[2]), int(match[3]), *timings)))
    for match in _SHORT_SCHEDULE.finditer(text):
        covered_starts.add(match.start())
        job, machine = int(match[1]), int(match[2])
        possible = by_job_machine.get((job, machine), [])
        if len(possible) > 1:
            fail("ambiguous_operation", f"J{job}-M{machine} refers to multiple route operations")
            continue
        try:
            timings = tuple(map(_number, match.group(3, 4, 5)))
        except ValueError:
            fail("nonfinite_number", "Operation timing is not finite")
            continue
        operation = possible[0].operation if possible else -1
        parsed_with_position.append((match.start(), ScheduledOperation(
            job, operation, machine, *timings)))
    for start in _ROW_START.finditer(text):
        if start.start() not in covered_starts:
            fail("malformed_operation", text[start.start():start.start() + 120].splitlines()[0])
    rows = tuple(row for _, row in sorted(parsed_with_position, key=lambda item: item[0]))
    observed: dict[tuple[int, int], list[ScheduledOperation]] = defaultdict(list)
    for row in rows:
        key = row.job, row.operation
        observed[key].append(row)
        op = expected.get(key)
        if op is None:
            fail("extra_operation", f"Unexpected operation {key} on machine {row.machine}")
        else:
            if row.machine != op.machine:
                fail("wrong_machine", f"Operation {key}: expected machine {op.machine}, got {row.machine}")
            if not math.isclose(row.duration, op.duration, rel_tol=0, abs_tol=tolerance):
                fail("wrong_duration", f"Operation {key}: expected duration {op.duration}, got {row.duration}")
        if row.start < 0:
            fail("negative_start", f"Operation {key} starts at {row.start}")
        if row.end < 0:
            fail("negative_end", f"Operation {key} ends at {row.end}")
        if row.duration <= 0:
            fail("nonpositive_duration", f"Operation {key} has duration {row.duration}")
        if not math.isclose(row.start + row.duration, row.end, rel_tol=0, abs_tol=tolerance):
            fail("arithmetic_error", f"Operation {key}: start + duration != end")
    missing = sorted(set(expected) - set(observed))
    if missing:
        fail("missing_operation", f"Missing operations: {missing}")
    duplicated = sorted(key for key, values in observed.items() if len(values) > 1)
    if duplicated:
        fail("duplicate_operation", f"Duplicate operations: {duplicated}")
    for job in range(instance.n_jobs):
        route = [op for op in instance.operations if op.job == job]
        for previous, following in zip(route, route[1:]):
            previous_rows = observed.get((job, previous.operation), [])
            following_rows = observed.get((job, following.operation), [])
            if len(previous_rows) == len(following_rows) == 1:
                if following_rows[0].start < previous_rows[0].end - tolerance:
                    fail("precedence_violation", f"Job {job}: operation {following.operation} starts before predecessor ends")
    machine_rows: dict[int, list[ScheduledOperation]] = defaultdict(list)
    for row in rows:
        machine_rows[row.machine].append(row)
    for machine, values in machine_rows.items():
        frontier = None
        for row in sorted(values, key=lambda op: (op.start, op.end)):
            if frontier is not None and row.start < frontier.end - tolerance:
                fail("machine_conflict", f"Machine {machine}: operations {(frontier.job, frontier.operation)} and {(row.job, row.operation)} overlap")
            if frontier is None or row.end > frontier.end:
                frontier = row
    makespan_declarations = list(re.finditer(r"\bMakespan\s*:", text, FLAGS))
    makespans = list(re.finditer(rf"\bMakespan\s*:\s*({NUMBER})(?![\w.])", text, FLAGS))
    if len(makespan_declarations) != len(makespans):
        fail("malformed_makespan", "A makespan declaration does not contain a valid numeric value")
    if len(makespan_declarations) > 1:
        fail("duplicate_makespan", "Multiple makespan declarations")
    if not makespans:
        if require_makespan:
            fail("missing_makespan", "No numeric makespan reported")
        reported = None
    else:
        try:
            reported = _number(makespans[0][1])
        except ValueError:
            reported = None
            fail("nonfinite_number", "Reported makespan is not finite")
    computed = max((row.end for row in rows), default=None)
    if reported is not None:
        if reported < 0:
            fail("negative_makespan", f"Reported makespan {reported}")
        if computed is None or not math.isclose(reported, computed, rel_tol=0, abs_tol=tolerance):
            fail("makespan_mismatch", f"Reported makespan {reported}; maximum end time {computed}")
    feasible = not errors
    gap = 100 * (computed - reference) / reference if feasible and reference is not None and computed is not None else None
    return ValidationResult(feasible, tuple(errors), tuple(details), rows, reported, computed, gap)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson binomial interval by default; no data gives nulls."""
    if not isinstance(total, int) or not isinstance(successes, int) or not (0 <= successes <= total):
        raise ValueError("Require integer 0 <= successes <= total")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class EvaluationRecord:
    instance_id: str
    candidate_id: int
    validation: ValidationResult


def _conditional_gaps(values: Iterable[float | None]) -> dict:
    gaps = [value for value in values if value is not None]
    return {"n_with_reference": len(gaps), "mean_gap_percent": statistics.mean(gaps) if gaps else None,
            "median_gap_percent": statistics.median(gaps) if gaps else None}


def aggregate_results(records: Sequence[EvaluationRecord], ks: Sequence[int] = (1, 3, 5)) -> dict:
    """Summarize candidate and instance outcomes with explicit denominators.

    Best-of-k uses the first k candidate IDs for each instance, never picks a
    sample order after seeing outcomes, and only includes instances with >=k
    candidates. Feasible gap statistics are conditional; failures have no gap.
    Wilson intervals over candidates describe candidate proportions; repeated
    candidates from one instance need not be statistically independent.
    """
    grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
    identities = set()
    errors = Counter()
    instance_errors: dict[str, set[str]] = defaultdict(set)
    for record in records:
        identity = record.instance_id, record.candidate_id
        if identity in identities:
            raise ValueError(f"Duplicate evaluation identity: {identity}")
        if record.candidate_id < 0:
            raise ValueError("Candidate IDs must be nonnegative")
        identities.add(identity)
        grouped[record.instance_id].append(record)
        for code in set(record.validation.errors):
            errors[code] += 1
            instance_errors[code].add(record.instance_id)
    for values in grouped.values():
        values.sort(key=lambda record: record.candidate_id)
    n_candidates, n_instances = len(records), len(grouped)
    feasible_candidates = sum(record.validation.feasible for record in records)
    feasible_instances = sum(any(record.validation.feasible for record in values) for values in grouped.values())
    result = {
        "n_instances": n_instances, "n_candidates": n_candidates,
        "feasible_candidates": feasible_candidates,
        "candidate_feasibility_rate": feasible_candidates / n_candidates if n_candidates else None,
        "candidate_feasibility_ci95_wilson": wilson_interval(feasible_candidates, n_candidates),
        "instances_with_any_feasible_candidate": feasible_instances,
        "instance_any_feasible_rate": feasible_instances / n_instances if n_instances else None,
        "instance_any_feasible_ci95_wilson": wilson_interval(feasible_instances, n_instances),
        "conditional_candidate_gap": _conditional_gaps(record.validation.gap_percent for record in records if record.validation.feasible),
        "error_counts_candidate_denominator": dict(sorted(errors.items())),
        "error_rates_candidate_denominator": {code: count / n_candidates for code, count in sorted(errors.items())},
        "error_counts_instance_denominator": {code: len(ids) for code, ids in sorted(instance_errors.items())},
        "error_rates_instance_denominator": {code: len(ids) / n_instances for code, ids in sorted(instance_errors.items())},
        "best_of_k": {},
    }
    for k in ks:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Best-of-k values must be positive integers")
        eligible = [values[:k] for values in grouped.values() if len(values) >= k]
        success = sum(any(record.validation.feasible for record in values) for values in eligible)
        best_gaps = [min(gaps) for values in eligible
                     if (gaps := [record.validation.gap_percent for record in values
                                  if record.validation.feasible and record.validation.gap_percent is not None])]
        result["best_of_k"][str(k)] = {
            "n_eligible_instances": len(eligible), "instances_with_feasible_candidate": success,
            "instance_feasibility_rate": success / len(eligible) if eligible else None,
            "instance_feasibility_ci95_wilson": wilson_interval(success, len(eligible)),
            "conditional_best_gap": _conditional_gaps(best_gaps),
        }
    return result


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    seen_in_training: bool


# Frozen before evaluation. Only the introductory instruction changes; instance
# representation, system role, decoding budget, and output requirements stay fixed.
PROMPTS = (
    Prompt("seen_1", "Instruct: Provide a solution schedule for the JSSP problem below, also indicate the makespan.", True),
    Prompt("seen_2", "Task: Provide the steps of a solution for the JSSP problem and determine the makespan.", True),
    Prompt("seen_3", "Command: Give a detailed solution to tackle the JSSP problem, focusing on optimizing the makespan.", True),
    Prompt("unseen_1", "Find a feasible schedule for the following job shop instance with the smallest makespan you can achieve. Report the operation timings and makespan.", False),
    Prompt("unseen_2", "Schedule every operation in the job shop problem below, respecting job order and machine capacity. Give the schedule and its makespan, aiming to minimize completion time.", False),
    Prompt("unseen_3", "Produce a job shop schedule for this instance. Minimize the makespan while satisfying every precedence and machine constraint, and report operation start and end times and makespan.", False),
    Prompt("unseen_4", "Solve the scheduling instance below: assign valid times to all operations, seek a low makespan, and provide the schedule together with its makespan.", False),
)
SYSTEM_PROMPT = "You are an expert in Job Shop Scheduling Problem"


def render_user_prompt(prompt_id: str, instance_text: str) -> str:
    prompt = next((prompt for prompt in PROMPTS if prompt.id == prompt_id), None)
    if prompt is None:
        raise ValueError(f"Unknown frozen prompt: {prompt_id}")
    return prompt.text + "\n\n" + instance_text


def select_validation_prompt(
    validation_records_by_prompt: Mapping[str, Sequence[EvaluationRecord]],
    *, forbidden_test_instance_ids: Iterable[str] = (),
) -> dict:
    """Deterministically select a frozen instruction using validation only.

    All candidates must evaluate the same nonempty instance/candidate pairs.
    Rank by feasibility count, then mean feasible gap (if every feasible result
    has a reference), then frozen registry order. The caller must provide actual
    held-out validation records; test IDs can be supplied as an overlap guard.
    The returned prompt is frozen for the entire subsequent test evaluation.
    """
    if not validation_records_by_prompt:
        raise ValueError("Validation results are required")
    registry_order = {prompt.id: index for index, prompt in enumerate(PROMPTS)}
    forbidden = set(forbidden_test_instance_ids)
    expected_identities = None
    ranked = []
    scores = {}
    for prompt_id, records in validation_records_by_prompt.items():
        if prompt_id not in registry_order:
            raise ValueError(f"Unknown prompt {prompt_id}")
        identities = {(record.instance_id, record.candidate_id) for record in records}
        if not identities or len(identities) != len(records):
            raise ValueError("Require nonempty validation results with unique candidates")
        if any(instance_id in forbidden for instance_id, _ in identities):
            raise ValueError("Validation prompt selection overlaps the test split")
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError("Every prompt must use the same validation instance/candidate pairs")
        valid = [record.validation for record in records if record.validation.feasible]
        gaps = [result.gap_percent for result in valid if result.gap_percent is not None]
        gap_score = statistics.mean(gaps) if valid and len(gaps) == len(valid) else math.inf
        scores[prompt_id] = {"n_candidates": len(records), "feasible_candidates": len(valid),
                             "conditional_mean_gap_percent": gap_score if math.isfinite(gap_score) else None}
        ranked.append((-len(valid), gap_score, registry_order[prompt_id], prompt_id))
    selected = min(ranked)[-1]
    return {"selected_prompt_id": selected, "selection_split": "validation",
            "selection_rule": "maximum feasible candidates; minimum conditional mean gap; registry order",
            "n_validation_instances": len({identity[0] for identity in expected_identities}),
            "scores": scores}
