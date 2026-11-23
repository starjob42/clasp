"""Constrained-GRPO reward machinery and projected multiplier controllers.

:class:`PrimalDualConstrainedReward` separates a candidate's primal objective quality from
its constraint violation and updates nonnegative penalty multipliers from observed
constraint residuals.  For the primal maximization convention used here, the
``lambda += eta * residual`` step is projected minimization of the dual function;
it is not called dual ascent. ``pd_component_profile`` presets select the cost
tolerances and weights; ``co_feasibility_lock_all`` gives no partial objective
credit to infeasible outputs.
"""
import math
import os
import warnings
from typing import Callable, Optional


def _is_primary_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _cost_name_set_from_env(name: str) -> set[str]:
    raw_value = os.environ.get(name, "")
    if not raw_value:
        return set()
    for separator in (",", ";", "+", ":"):
        raw_value = raw_value.replace(separator, " ")
    return {item.strip() for item in raw_value.split() if item.strip()}


# Several task validators expose multiple, deliberately redundant views of the
# same underlying violation.  Those views are useful for diagnostics and dual
# control, but summing them is not a sound distance-to-feasibility metric.  In
# particular, CVRP capacity is represented by total overflow, the fraction of
# overflowing routes, and maximum route overflow.  Counting all three made a
# nearly valid parsed route look worse than an unparseable completion, for which
# every downstream head is (correctly) masked.
_RECOVERY_COMPONENT_FAMILIES = {
    # Parsing is a separate lexicographic tier; see
    # ``_lexicographic_recovery_violation_score`` below.
    "parse_failure": "parseability",
    # Routing representation and coverage aliases.
    "start_at_depot": "route_boundary_shape",
    "end_at_depot": "route_boundary_shape",
    "depot_boundary": "route_boundary_shape",
    "route_shape_invalid": "route_boundary_shape",
    "interior_depot": "route_boundary_shape",
    "route_length_mismatch": "route_coverage",
    "duplicate_customer": "route_coverage",
    "missing_customer": "route_coverage",
    "customer_count_mismatch": "route_coverage",
    "route_count_mismatch": "route_cardinality",
    "capacity_overflow": "route_capacity",
    "over_capacity_route_fraction": "route_capacity",
    "max_route_capacity_overflow": "route_capacity",
    # OP budget heads are nested views of the same route-budget failure.
    "length_overflow": "route_budget",
    "length_margin_overflow": "route_budget",
    "prize_surplus_overbudget": "route_budget",
    "prize_cap_over_ref": "route_budget",
    "duplicate_visit": "visit_uniqueness",
    # MIS conflicts have an edge-count and a selected-node view.
    "edge_conflict": "independence_conflict",
    "conflict_node_fraction": "independence_conflict",
    "duplicate_node": "selection_uniqueness",
    "size_deficit_over_ref": "mis_objective_quality",
    "repair_size_deficit_over_ref": "mis_objective_quality",
    "size_excess_over_ref": "mis_objective_quality",
    "addable_vertex_fraction": "mis_maximality",
    # JSSP representation and topology heads overlap substantially.
    "machine_row_mismatch": "schedule_shape",
    "row_length_mismatch": "schedule_shape",
    "row_set_mismatch": "schedule_shape",
    "invalid_job": "schedule_job_coverage",
    "duplicate_job": "schedule_job_coverage",
    "missing_job": "schedule_job_coverage",
    "schedule_invalid": "schedule_topology",
    "cycle_node_fraction": "schedule_topology",
    "processed_node_deficit": "schedule_topology",
    "parse_bad_token": "schedule_representation",
    # Multiple optional target heads can describe the same objective miss.
    "target_gap_excess": "tsp_objective_quality",
    "target_tail_gap_excess": "tsp_objective_quality",
    "tsp_objective_gap": "tsp_objective_quality",
    "cvrp_target_gap_excess": "cvrp_objective_quality",
    "cvrp_objective_gap": "cvrp_objective_quality",
    "makespan_gap_over_ref": "jssp_objective_quality",
}


def _bounded_positive_severity(value: float) -> float:
    """Map a non-negative violation magnitude monotonically into ``[0, 1)``."""
    positive = max(0.0, float(value))
    return float(positive / (1.0 + positive))


def _probabilistic_union(values: list[float]) -> float:
    """Combine bounded independent-family severities without an unbounded sum."""
    complement = 1.0
    for value in values:
        complement *= 1.0 - max(0.0, min(float(value), 1.0))
    # Each input is strictly below one, but guard against round-off producing
    # exactly one so the parseability tier remains strictly separated.
    return float(min(1.0 - 1e-12, max(0.0, 1.0 - complement)))


def _lexicographic_recovery_violation_score(
    cost_names: list[str],
    row: list[float],
    mask_row: list[float],
    cost_weights: list[float],
    cost_limits: list[float],
) -> float:
    """Return a bounded, deduplicated distance used by all-infeasible recovery.

    Lower is better and zero still means no positive component gap.  Parsed
    candidates occupy ``[0, 1)``.  Any candidate with an active positive
    ``parse_failure`` gap occupies ``(2, 3)``, which implements the
    lexicographic ordering

        parseable candidate  <  unparseable candidate

    regardless of how many downstream heads are observable.  Within a tier,
    correlated component views are collapsed by maximum severity and distinct
    violation families are combined by a bounded probabilistic union.  The
    per-component dual statistics and Lagrangian remain untouched.  Callers may
    pass either the configured component weights or finite dual-adjusted weights;
    both produce a bounded score with the same hard parseability ordering.
    """
    family_severities: dict[str, float] = {}
    parse_gap = 0.0

    for name, cost, mask, weight, limit_value in zip(
        cost_names,
        row,
        mask_row,
        cost_weights,
        cost_limits,
    ):
        if float(mask) <= 0.0:
            continue
        positive_gap = max(0.0, float(cost) - float(limit_value))
        if positive_gap <= 0.0:
            continue

        normalized_name = str(name).strip().lower()
        family = _RECOVERY_COMPONENT_FAMILIES.get(
            normalized_name,
            f"component:{normalized_name}",
        )
        weighted_severity = _bounded_positive_severity(float(weight) * positive_gap)
        if family == "parseability":
            # Even a deliberately small/zero component weight must not erase
            # the hard parseability ordering when the head itself is active.
            weighted_severity = max(
                weighted_severity,
                _bounded_positive_severity(positive_gap),
            )
            parse_gap = max(parse_gap, positive_gap)
        family_severities[family] = max(
            family_severities.get(family, 0.0),
            weighted_severity,
        )

    if not family_severities:
        return 0.0

    all_severity = _probabilistic_union(list(family_severities.values()))
    if parse_gap > 0.0:
        # Leave a full unit of numerical separation: a parsed score extremely
        # close to one can round to 1.0 after the trainer converts it to
        # float32, while every parse-failure score remains strictly above 2.0.
        return float(2.0 + all_severity)
    return float(all_severity)


def _to_float_list(values, expected_len: int, value_name: str) -> list[float]:
    if expected_len < 0:
        raise ValueError(f"expected_len must be >= 0, got {expected_len}")

    if isinstance(values, (int, float)):
        scalar = float(values)
        if not math.isfinite(scalar):
            raise ValueError(f"{value_name} returned non-finite scalar value: {scalar!r}")
        return [scalar] * expected_len

    if values is None:
        raise ValueError(f"{value_name} returned None; expected {expected_len} value(s)")

    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value_name} returned non-numeric values") from exc

    non_finite = [(idx, val) for idx, val in enumerate(out) if not math.isfinite(val)]
    if non_finite:
        examples = ", ".join(f"{idx}:{val!r}" for idx, val in non_finite[:5])
        raise ValueError(
            f"{value_name} returned non-finite value(s); examples: {examples}. "
            "All constrained-GRPO heads must emit finite numbers."
        )

    if expected_len == 0:
        return []

    if len(out) == 0:
        raise ValueError(f"{value_name} returned an empty list; expected {expected_len} value(s)")

    if len(out) == expected_len:
        return out

    if len(out) == 1:
        return out * expected_len

    raise ValueError(
        f"{value_name} returned {len(out)} value(s), expected {expected_len}. "
        "This length mismatch can corrupt reward-to-sample alignment."
    )


def _normalize_cost_output(cost_output, expected_len: int, value_name: str):
    if isinstance(cost_output, tuple) and len(cost_output) == 2:
        raw_cost_names, raw_cost_vectors = cost_output
        raw_cost_masks = None
        raw_constraint_scores = None
    elif isinstance(cost_output, dict):
        raw_cost_names = cost_output.get("cost_names")
        raw_cost_vectors = cost_output.get("cost_vectors", cost_output.get("costs"))
        raw_cost_masks = cost_output.get("cost_masks")
        raw_constraint_scores = cost_output.get("constraint_scores")
    else:
        raise ValueError(
            f"{value_name} must return either a dict with 'cost_names'/'cost_vectors' or a tuple "
            "(cost_names, cost_vectors)."
        )

    if not isinstance(raw_cost_names, list) or not raw_cost_names:
        raise ValueError(f"{value_name} must provide a non-empty list of cost_names.")

    cost_names = [str(name) for name in raw_cost_names]
    if len(set(cost_names)) != len(cost_names):
        raise ValueError(f"{value_name} returned duplicate cost names: {cost_names}")

    if expected_len == 0:
        cost_vectors: list[list[float]] = []
    else:
        if not isinstance(raw_cost_vectors, list) or len(raw_cost_vectors) != expected_len:
            raise ValueError(
                f"{value_name} must return one cost vector per completion. "
                f"Observed {type(raw_cost_vectors).__name__} with length "
                f"{len(raw_cost_vectors) if isinstance(raw_cost_vectors, list) else 'n/a'}; "
                f"expected {expected_len}."
            )

        cost_vectors = []
        for row_idx, row in enumerate(raw_cost_vectors):
            if isinstance(row, (int, float)):
                row_values = [float(row)]
            else:
                try:
                    row_values = [float(v) for v in row]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{value_name} returned a non-numeric cost row at index {row_idx}: {row!r}"
                    ) from exc

            if len(row_values) != len(cost_names):
                raise ValueError(
                    f"{value_name} returned a cost row of length {len(row_values)} at index {row_idx}, "
                    f"expected {len(cost_names)} to match cost_names={cost_names!r}."
                )

            non_finite = [val for val in row_values if not math.isfinite(val)]
            if non_finite:
                raise ValueError(
                    f"{value_name} returned non-finite cost value(s) at row {row_idx}: {non_finite[:5]!r}"
                )

            cost_vectors.append(row_values)

    if raw_cost_masks is None:
        cost_masks = [[1.0] * len(cost_names) for _ in range(expected_len)]
    else:
        if not isinstance(raw_cost_masks, list) or len(raw_cost_masks) != expected_len:
            raise ValueError(
                f"{value_name} must return one cost mask row per completion. "
                f"Observed {type(raw_cost_masks).__name__} with length "
                f"{len(raw_cost_masks) if isinstance(raw_cost_masks, list) else 'n/a'}; "
                f"expected {expected_len}."
            )

        cost_masks = []
        for row_idx, row in enumerate(raw_cost_masks):
            if isinstance(row, (int, float)):
                row_values = [float(row)]
            else:
                try:
                    row_values = [float(v) for v in row]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{value_name} returned a non-numeric cost mask row at index {row_idx}: {row!r}"
                    ) from exc

            if len(row_values) != len(cost_names):
                raise ValueError(
                    f"{value_name} returned a cost mask row of length {len(row_values)} at index {row_idx}, "
                    f"expected {len(cost_names)} to match cost_names={cost_names!r}."
                )

            non_finite = [val for val in row_values if not math.isfinite(val)]
            if non_finite:
                raise ValueError(
                    f"{value_name} returned non-finite cost mask value(s) at row {row_idx}: {non_finite[:5]!r}"
                )

            out_of_range = [val for val in row_values if (val < 0.0 or val > 1.0)]
            if out_of_range:
                raise ValueError(
                    f"{value_name} returned cost mask values outside [0, 1] at row {row_idx}: "
                    f"{out_of_range[:5]!r}"
                )

            non_binary = [val for val in row_values if val not in {0.0, 1.0}]
            if non_binary:
                raise ValueError(
                    f"{value_name} returned non-binary cost mask values at row {row_idx}: "
                    f"{non_binary[:5]!r}. Constrained GRPO expects observability masks in {{0, 1}}."
                )

            if not any(row_values):
                raise ValueError(
                    f"{value_name} returned an all-zero cost mask at row {row_idx}. "
                    "Every sample must expose at least one active constraint component "
                    "(for example a parse_failure head)."
                )

            cost_masks.append(row_values)

    if raw_constraint_scores is None:
        constraint_scores = []
        for row, mask_row in zip(cost_vectors, cost_masks):
            active_values = [cost for cost, mask in zip(row, mask_row) if mask > 0.0]
            if not active_values:
                constraint_scores.append(1.0)
                continue
            mean_cost = sum(active_values) / len(active_values)
            constraint_scores.append(float(1.0 - mean_cost))
    else:
        constraint_scores = _to_float_list(
            raw_constraint_scores, expected_len, f"{value_name}.constraint_scores"
        )

    return cost_names, cost_vectors, cost_masks, constraint_scores


class PrimalDualConstrainedReward:
    """
    Vector-cost constrained GRPO reward wrapper.

    For each sample, the constrained quantity is the masked constraint gap

        g_i = m_i ⊙ (c_i - eps)

    where c_i in [0, 1]^m are normalized costs, m_i in {0, 1}^m are
    observability masks, and eps in [0, 1]^m are per-component upper bounds.

    The wrapped scalar reward is the sample Lagrangian

        primal_i - <lambda, g_i>

    while the custom trainer constructs the clipped GRPO surrogate from

        reward advantage:      A_i^r = primal_i - mean_group(primal)
        lagrangian advantage:  A_i^L = A_i^r - <lambda, g_i>

    so only the primal reward is group-centered, while the absolute constraint
    gap remains in the optimization objective.

    Legacy multiplier updates are performed independently per cost dimension:

        lambda_j <- clip(lambda_j + lr(lambda_j) * E[g_j], 0, lambda_max)

    with g_{i,j} = m_{i,j} * (c_{i,j} - eps_j).
    When CLASP's proper controller is enabled, the update instead uses the
    current minibatch mean of m_{i,j} * max(0, c_{i,j} - eps_j).
    """

    def __init__(
        self,
        structure_fn: Callable,
        constraint_fn: Optional[Callable],
        objective_fn: Callable,
        *,
        cost_components_fn: Optional[Callable] = None,
        constraint_target: float = 0.98,
        cost_limit: Optional[float | list[float]] = None,
        lambda_init: float | list[float] = 0.0,
        dual_lr: float = 0.05,
        dual_lr_up: Optional[float] = None,
        dual_lr_down: Optional[float] = None,
        dual_update_interval: int = 1,
        violation_ema_beta: float = 1.0,
        lambda_max: float = 10.0,
        violation_tolerance: float = 0.01,
        objective_weight: float = 1.0,
        structure_weight: float = 0.2,
        sync_across_ranks: bool = False,
        strict_constraint_range: bool = True,
        log_every: int = 50,
        component_weight_map: Optional[dict[str, float]] = None,
        component_cost_limit_map: Optional[dict[str, float]] = None,
    ):
        self.structure_fn = structure_fn
        self.constraint_fn = constraint_fn
        self.objective_fn = objective_fn
        self.cost_components_fn = cost_components_fn

        if not (0.0 <= constraint_target <= 1.0):
            raise ValueError(f"constraint_target must be in [0, 1], got {constraint_target}")
        if dual_lr < 0.0:
            raise ValueError(f"dual_lr must be >= 0, got {dual_lr}")
        if dual_lr_up is not None and dual_lr_up < 0.0:
            raise ValueError(f"dual_lr_up must be >= 0, got {dual_lr_up}")
        if dual_lr_down is not None and dual_lr_down < 0.0:
            raise ValueError(f"dual_lr_down must be >= 0, got {dual_lr_down}")
        if int(dual_update_interval) < 1:
            raise ValueError(f"dual_update_interval must be >= 1, got {dual_update_interval}")
        if not (0.0 < violation_ema_beta <= 1.0):
            raise ValueError(f"violation_ema_beta must be in (0, 1], got {violation_ema_beta}")
        if lambda_max < 0.0:
            raise ValueError(f"lambda_max must be >= 0, got {lambda_max}")
        if violation_tolerance < 0.0:
            raise ValueError(f"violation_tolerance must be >= 0, got {violation_tolerance}")
        if objective_weight < 0.0:
            raise ValueError(f"objective_weight must be >= 0, got {objective_weight}")
        if structure_weight < 0.0:
            raise ValueError(f"structure_weight must be >= 0, got {structure_weight}")
        self.component_weight_map = {
            str(name): float(value)
            for name, value in (component_weight_map or {}).items()
        }
        self.component_cost_limit_map = {
            str(name): float(value)
            for name, value in (component_cost_limit_map or {}).items()
        }
        for name, value in self.component_weight_map.items():
            if value < 0.0:
                raise ValueError(f"component weight for {name!r} must be >= 0, got {value}")
        for name, value in self.component_cost_limit_map.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"component cost limit for {name!r} must lie in [0, 1], got {value}")

        if isinstance(cost_limit, list):
            invalid = [value for value in cost_limit if float(value) < 0.0 or float(value) > 1.0]
            if invalid:
                raise ValueError(f"cost_limit values must lie in [0, 1], got {invalid[:5]!r}")
        elif cost_limit is not None and (float(cost_limit) < 0.0 or float(cost_limit) > 1.0):
            raise ValueError(f"cost_limit must lie in [0, 1], got {cost_limit}")

        self.constraint_target = float(constraint_target)
        self.cost_limit = cost_limit
        self.lambda_init = lambda_init
        self.dual_lr = float(dual_lr)
        self.dual_lr_up = float(dual_lr if dual_lr_up is None else dual_lr_up)
        self.dual_lr_down = float(dual_lr if dual_lr_down is None else dual_lr_down)
        self.dual_update_interval = int(dual_update_interval)
        self.violation_ema_beta = float(violation_ema_beta)
        self.lambda_max = float(lambda_max)
        self.violation_tolerance = float(violation_tolerance)

        self.objective_weight = float(objective_weight)
        self.structure_weight = float(structure_weight)
        self.sync_across_ranks = bool(sync_across_ranks)
        self.strict_constraint_range = bool(strict_constraint_range)
        self.log_every = max(0, int(log_every))

        self.cost_names: list[str] = []
        self.cost_limits: list[float] = []
        self.cost_weights: list[float] = []
        self.dual_lambdas: list[float] = []
        self._legacy_loaded_scalar_lambda: Optional[float] = None
        self.feasibility_ignore_cost_names = _cost_name_set_from_env(
            "CLASP_FEASIBILITY_IGNORE_COST_NAMES"
        )

        # ------------------- CLASP additions -------------------
        # Defaults keep the existing EMA-based multiplier path active. The
        # setter ``enable_clasp_proper_dual`` flips the switch at wiring time.
        self.clasp_proper_dual: bool = False
        self.clasp_base_dual_lr: float = float(dual_lr)
        self.clasp_dual_update_interval: int = int(dual_update_interval)
        self.clasp_proper_dual_gradient_mode: str = "signed"
        self._loaded_clasp_proper_dual_gradient_mode: Optional[str] = None
        self._clasp_last_feasibility_rate: Optional[float] = None

        self.call_count = 0
        self.update_count = 0
        self._collect_stats = True
        self._pending_cost_sums: list[float] = []
        self._pending_cost_counts: list[float] = []
        self._pending_positive_gap_sums: list[float] = []
        self._pending_feasibility_sum = 0.0
        self._pending_constraint_sum = 0.0
        self._pending_objective_sum = 0.0
        self._pending_structure_sum = 0.0
        self._pending_count = 0
        self._ema_mean_gaps: Optional[list[float | None]] = None
        self._steps_since_lambda_update = 0
        self._sync_warning_emitted = False
        self._sync_disabled_due_to_error = False
        self._range_warning_emitted = False
        self._lambda_saturation_warning_emitted = False
        self._last_components = {
            "primal_scores": [],
            "objective_scores": [],
            "structure_scores": [],
            "weighted_gap_scores": [],
            "positive_weighted_gap_scores": [],
            "raw_positive_gap_scores": [],
            "dual_aware_recovery_gap_scores": [],
            "constraint_violation_scores": [],
            "feasibility_flags": [],
            "constraint_scores": [],
            "combined_scores": [],
        }
        # TRL expects every reward callable to expose a __name__ attribute.
        self.__name__ = self.__class__.__name__

    def _resolved_default_cost_limit(self) -> float:
        if self.cost_limit is not None:
            raise RuntimeError("_resolved_default_cost_limit called despite explicit cost_limit")
        if self.violation_tolerance > 0.0:
            return float(self.violation_tolerance)
        return float(max(0.0, 1.0 - self.constraint_target))

    def _cost_limit_for_name(self, name: str, base_limit: float) -> float:
        if name not in self.component_cost_limit_map:
            return float(base_limit)
        return float(self.component_cost_limit_map[name])

    def _component_weight_for_name(self, name: str) -> float:
        return float(self.component_weight_map.get(name, 1.0))

    def _current_cost_limits_for_names(self, cost_names: list[str]) -> list[float]:
        if isinstance(self.cost_limit, list) and len(self.cost_limit) == len(cost_names):
            return [
                self._cost_limit_for_name(name, float(value))
                for name, value in zip(cost_names, self.cost_limit)
            ]
        base_limit = (
            self._resolved_default_cost_limit()
            if self.cost_limit is None or isinstance(self.cost_limit, list)
            else float(self.cost_limit)
        )
        return [self._cost_limit_for_name(name, base_limit) for name in cost_names]

    def _current_cost_weights_for_names(self, cost_names: list[str]) -> list[float]:
        return [self._component_weight_for_name(name) for name in cost_names]

    def _tighten_state_to_current_config(self) -> None:
        """Apply current run limits/weights without discarding learned lambdas.

        Checkpoint state is useful for dual multipliers, but cost limits and
        component weights are part of the active experiment config. A resumed
        run may intentionally tighten or relax individual heads, so the current
        config must replace saved limits/weights exactly.
        """
        if not self.cost_names:
            return
        num_costs = len(self.cost_names)
        current_limits = self._current_cost_limits_for_names(self.cost_names)
        current_weights = self._current_cost_weights_for_names(self.cost_names)
        if self.cost_limits and len(self.cost_limits) == num_costs:
            self.cost_limits = [float(value) for value in current_limits]
        if self.cost_weights and len(self.cost_weights) == num_costs:
            self.cost_weights = [float(value) for value in current_weights]

    def _initialize_cost_state(self, cost_names: list[str]):
        if self.cost_names:
            if self.cost_names != cost_names:
                previous_names = list(self.cost_names)
                previous_set = set(previous_names)
                current_set = set(cost_names)
                if not previous_set.issubset(current_set):
                    raise ValueError(
                        "cost_components_fn changed its cost_names across calls in a non-migratable way. "
                        f"Observed previous={previous_names!r}, current={cost_names!r}."
                    )

                prev_index = {name: idx for idx, name in enumerate(previous_names)}
                previous_limits = (
                    list(self.cost_limits)
                    if len(self.cost_limits) == len(previous_names)
                    else []
                )
                previous_weights = (
                    list(self.cost_weights)
                    if len(self.cost_weights) == len(previous_names)
                    else []
                )
                previous_lambdas = (
                    list(self.dual_lambdas)
                    if len(self.dual_lambdas) == len(previous_names)
                    else []
                )
                previous_pending_sums = (
                    list(self._pending_cost_sums)
                    if len(self._pending_cost_sums) == len(previous_names)
                    else []
                )
                previous_pending_counts = (
                    list(self._pending_cost_counts)
                    if len(self._pending_cost_counts) == len(previous_names)
                    else []
                )
                previous_positive_sums = (
                    list(self._pending_positive_gap_sums)
                    if len(self._pending_positive_gap_sums) == len(previous_names)
                    else []
                )
                previous_ema = (
                    list(self._ema_mean_gaps)
                    if self._ema_mean_gaps is not None and len(self._ema_mean_gaps) == len(previous_names)
                    else None
                )

                default_limits = self._current_cost_limits_for_names(cost_names)
                default_weights = self._current_cost_weights_for_names(cost_names)

                if isinstance(self.lambda_init, list) and len(self.lambda_init) == len(cost_names):
                    default_lambdas = [float(value) for value in self.lambda_init]
                else:
                    init_value = (
                        float(self._legacy_loaded_scalar_lambda)
                        if self._legacy_loaded_scalar_lambda is not None
                        else (float(self.lambda_init) if not isinstance(self.lambda_init, list) else 0.0)
                    )
                    default_lambdas = [init_value] * len(cost_names)
                self._legacy_loaded_scalar_lambda = None

                self.cost_names = list(cost_names)
                self.cost_limits = []
                self.cost_weights = []
                self.dual_lambdas = []
                self._pending_cost_sums = []
                self._pending_cost_counts = []
                self._pending_positive_gap_sums = []
                migrated_ema = [] if previous_ema is not None else None

                for idx, name in enumerate(self.cost_names):
                    old_idx = prev_index.get(name)
                    self.cost_limits.append(
                        min(float(previous_limits[old_idx]), float(default_limits[idx]))
                        if old_idx is not None and previous_limits
                        else float(default_limits[idx])
                    )
                    self.cost_weights.append(
                        max(float(previous_weights[old_idx]), float(default_weights[idx]))
                        if old_idx is not None and previous_weights
                        else float(default_weights[idx])
                    )
                    self.dual_lambdas.append(
                        max(0.0, min(self.lambda_max, float(previous_lambdas[old_idx])))
                        if old_idx is not None and previous_lambdas
                        else max(0.0, min(self.lambda_max, float(default_lambdas[idx])))
                    )
                    self._pending_cost_sums.append(
                        float(previous_pending_sums[old_idx])
                        if old_idx is not None and previous_pending_sums
                        else 0.0
                    )
                    self._pending_cost_counts.append(
                        float(previous_pending_counts[old_idx])
                        if old_idx is not None and previous_pending_counts
                        else 0.0
                    )
                    self._pending_positive_gap_sums.append(
                        float(previous_positive_sums[old_idx])
                        if old_idx is not None and previous_positive_sums
                        else 0.0
                    )
                    if migrated_ema is not None:
                        migrated_ema.append(
                            previous_ema[old_idx]
                            if old_idx is not None and previous_ema is not None
                            else None
                        )

                self._ema_mean_gaps = migrated_ema
                warnings.warn(
                    "Migrated constrained-GRPO cost state to include newly added cost heads. "
                    f"previous={previous_names!r}, current={self.cost_names!r}.",
                    RuntimeWarning,
                )
            num_costs = len(self.cost_names)
            if self.cost_limits and len(self.cost_limits) != num_costs:
                raise ValueError(
                    f"Loaded cost_limits has length {len(self.cost_limits)}, expected {num_costs} "
                    f"for cost_names={self.cost_names!r}."
                )
            if not self.cost_limits:
                base_limit = (
                    self._resolved_default_cost_limit()
                    if self.cost_limit is None
                    else float(self.cost_limit)
                )
                self.cost_limits = [
                    self._cost_limit_for_name(name, base_limit)
                    for name in self.cost_names
                ]
            if self.cost_weights and len(self.cost_weights) != num_costs:
                raise ValueError(
                    f"Loaded cost_weights has length {len(self.cost_weights)}, expected {num_costs} "
                    f"for cost_names={self.cost_names!r}."
                )
            if not self.cost_weights:
                self.cost_weights = [
                    self._component_weight_for_name(name)
                    for name in self.cost_names
                ]
            self._tighten_state_to_current_config()
            if self.dual_lambdas and len(self.dual_lambdas) != num_costs:
                raise ValueError(
                    f"Loaded dual_lambdas has length {len(self.dual_lambdas)}, expected {num_costs} "
                    f"for cost_names={self.cost_names!r}."
                )
            if not self.dual_lambdas:
                if isinstance(self.lambda_init, list):
                    if len(self.lambda_init) != num_costs:
                        raise ValueError(
                            f"lambda_init specifies {len(self.lambda_init)} values, "
                            f"but cost_names has {num_costs} entries."
                        )
                    self.dual_lambdas = [float(v) for v in self.lambda_init]
                else:
                    init_value = (
                        float(self._legacy_loaded_scalar_lambda)
                        if self._legacy_loaded_scalar_lambda is not None
                        else float(self.lambda_init)
                    )
                    self.dual_lambdas = [init_value] * num_costs
                self._legacy_loaded_scalar_lambda = None
            if len(self._pending_cost_sums) != num_costs:
                self._pending_cost_sums = [0.0] * num_costs
            if len(self._pending_cost_counts) != num_costs:
                self._pending_cost_counts = [0.0] * num_costs
            if len(self._pending_positive_gap_sums) != num_costs:
                self._pending_positive_gap_sums = [0.0] * num_costs
            if self._ema_mean_gaps is not None and len(self._ema_mean_gaps) != num_costs:
                self._ema_mean_gaps = None
            return

        self.cost_names = list(cost_names)
        num_costs = len(self.cost_names)

        if isinstance(self.cost_limit, list):
            if len(self.cost_limit) != num_costs:
                raise ValueError(
                    f"cost_limit specifies {len(self.cost_limit)} values, but cost_names has {num_costs} entries."
                )
            self.cost_limits = [float(v) for v in self.cost_limit]
        else:
            base_limit = (
                self._resolved_default_cost_limit()
                if self.cost_limit is None
                else float(self.cost_limit)
            )
            self.cost_limits = [
                self._cost_limit_for_name(name, base_limit)
                for name in self.cost_names
            ]

        self.cost_weights = [
            self._component_weight_for_name(name)
            for name in self.cost_names
        ]

        if isinstance(self.lambda_init, list):
            if len(self.lambda_init) != num_costs:
                raise ValueError(
                    f"lambda_init specifies {len(self.lambda_init)} values, but cost_names has {num_costs} entries."
                )
            init_values = [float(v) for v in self.lambda_init]
        else:
            init_values = [float(self.lambda_init)] * num_costs

        if self._legacy_loaded_scalar_lambda is not None:
            init_values = [float(self._legacy_loaded_scalar_lambda)] * num_costs
            self._legacy_loaded_scalar_lambda = None

        for value in init_values:
            if value < 0.0:
                raise ValueError(f"lambda_init must be >= 0, got {value}")
            if value > self.lambda_max:
                raise ValueError(f"lambda_init ({value}) must be <= lambda_max ({self.lambda_max})")

        self.dual_lambdas = init_values
        self._pending_cost_sums = [0.0] * num_costs
        self._pending_cost_counts = [0.0] * num_costs
        self._pending_positive_gap_sums = [0.0] * num_costs
        if self._ema_mean_gaps is not None and len(self._ema_mean_gaps) != num_costs:
            self._ema_mean_gaps = None

    def _validate_range(self, values: list[float], value_name: str) -> list[float]:
        out_of_range = [value for value in values if (value < 0.0 or value > 1.0)]
        if not out_of_range:
            return values

        if self.strict_constraint_range:
            min_value = min(values)
            max_value = max(values)
            examples = ", ".join(f"{value:.4f}" for value in out_of_range[:5])
            raise ValueError(
                f"{value_name} must return values in [0, 1]. "
                f"Observed range [{min_value:.4f}, {max_value:.4f}] with examples: {examples}. "
                "Set strict_constraint_range=False to clamp instead."
            )

        if not self._range_warning_emitted:
            min_value = min(values)
            max_value = max(values)
            warnings.warn(
                f"{value_name} returned values outside [0, 1]; clamping for constrained GRPO stability. "
                f"Observed range [{min_value:.4f}, {max_value:.4f}]",
                RuntimeWarning,
            )
            self._range_warning_emitted = True
        return [max(0.0, min(1.0, value)) for value in values]

    def _legacy_cost_components(self, prompts, completions, **kwargs):
        if self.constraint_fn is None:
            raise ValueError(
                "Either cost_components_fn or constraint_fn must be provided to PrimalDualConstrainedReward."
            )

        constraint_scores = _to_float_list(
            self.constraint_fn(prompts, completions, **kwargs),
            len(completions),
            "constraint_fn",
        )
        constraint_scores = self._validate_range(constraint_scores, "constraint_fn")
        cost_names = ["normalized_infeasibility"]
        cost_vectors = [[1.0 - score] for score in constraint_scores]
        return cost_names, cost_vectors, constraint_scores

    def state_dict(self) -> dict:
        state = {
            "cost_names": list(self.cost_names),
            "cost_limits": [float(value) for value in self.cost_limits],
            "cost_weights": [float(value) for value in self.cost_weights],
            "dual_lambdas": [float(value) for value in self.dual_lambdas],
            "call_count": int(self.call_count),
            "update_count": int(self.update_count),
            "pending_stats": {
                "cost_sums": [float(value) for value in self._pending_cost_sums],
                "cost_counts": [float(value) for value in self._pending_cost_counts],
                "positive_gap_sums": [
                    float(value) for value in self._pending_positive_gap_sums
                ],
                "feasibility_sum": float(self._pending_feasibility_sum),
                "constraint_sum": float(self._pending_constraint_sum),
                "objective_sum": float(self._pending_objective_sum),
                "structure_sum": float(self._pending_structure_sum),
                "count": int(self._pending_count),
            },
            "controller_state": {
                "proper_dual_gradient_mode": str(
                    self.clasp_proper_dual_gradient_mode
                ),
                "ema_mean_gaps": (
                    None
                    if self._ema_mean_gaps is None
                    else [
                        None if value is None else float(value)
                        for value in self._ema_mean_gaps
                    ]
                ),
                "steps_since_lambda_update": int(self._steps_since_lambda_update),
            },
        }
        if len(self.dual_lambdas) == 1:
            state["dual_lambda"] = float(self.dual_lambdas[0])
        return state

    def load_state_dict(self, state: dict):
        if not isinstance(state, dict):
            return

        raw_cost_names = state.get("cost_names")
        if isinstance(raw_cost_names, list) and raw_cost_names:
            self.cost_names = [str(name) for name in raw_cost_names]

        raw_cost_limits = state.get("cost_limits")
        if isinstance(raw_cost_limits, list) and raw_cost_limits:
            self.cost_limits = [float(value) for value in raw_cost_limits]

        raw_cost_weights = state.get("cost_weights")
        if isinstance(raw_cost_weights, list) and raw_cost_weights:
            self.cost_weights = [float(value) for value in raw_cost_weights]

        raw_dual_lambdas = state.get("dual_lambdas")
        if isinstance(raw_dual_lambdas, list) and raw_dual_lambdas:
            self.dual_lambdas = [
                max(0.0, min(self.lambda_max, float(value))) for value in raw_dual_lambdas
            ]
        else:
            raw_dual_lambda = state.get("dual_lambda")
            if raw_dual_lambda is not None:
                self._legacy_loaded_scalar_lambda = max(
                    0.0, min(self.lambda_max, float(raw_dual_lambda))
                )

        call_count = state.get("call_count")
        if call_count is not None:
            self.call_count = max(0, int(call_count))

        update_count = state.get("update_count")
        if update_count is not None:
            self.update_count = max(0, int(update_count))

        pending_stats = state.get("pending_stats")
        if isinstance(pending_stats, dict):
            raw_cost_sums = pending_stats.get("cost_sums", [])
            if isinstance(raw_cost_sums, list):
                self._pending_cost_sums = [float(value) for value in raw_cost_sums]
            raw_cost_counts = pending_stats.get("cost_counts")
            if isinstance(raw_cost_counts, list):
                self._pending_cost_counts = [float(value) for value in raw_cost_counts]
            raw_positive_gap_sums = pending_stats.get("positive_gap_sums")
            if isinstance(raw_positive_gap_sums, list):
                self._pending_positive_gap_sums = [
                    float(value) for value in raw_positive_gap_sums
                ]
            self._pending_feasibility_sum = float(pending_stats.get("feasibility_sum", 0.0))
            self._pending_constraint_sum = float(pending_stats.get("constraint_sum", 0.0))
            self._pending_objective_sum = float(pending_stats.get("objective_sum", 0.0))
            self._pending_structure_sum = float(pending_stats.get("structure_sum", 0.0))
            self._pending_count = max(0, int(pending_stats.get("count", 0)))
            if not self._pending_cost_counts and self._pending_cost_sums:
                self._pending_cost_counts = [float(self._pending_count)] * len(self._pending_cost_sums)

        controller_state = state.get("controller_state")
        if isinstance(controller_state, dict):
            raw_gradient_mode = controller_state.get("proper_dual_gradient_mode")
            if raw_gradient_mode is not None:
                gradient_mode = str(raw_gradient_mode).strip().lower()
                if gradient_mode not in {"signed", "positive_excess"}:
                    raise ValueError(
                        "Saved CLASP proper-dual gradient mode must be 'signed' or "
                        f"'positive_excess', got {raw_gradient_mode!r}"
                    )
                if (
                    self.clasp_proper_dual
                    and gradient_mode != self.clasp_proper_dual_gradient_mode
                ):
                    raise RuntimeError(
                        "Cannot resume CLASP proper-dual state with a different gradient "
                        "estimator: checkpoint="
                        f"{gradient_mode!r}, configured="
                        f"{self.clasp_proper_dual_gradient_mode!r}."
                    )
                self._loaded_clasp_proper_dual_gradient_mode = gradient_mode
                if not self.clasp_proper_dual:
                    self.clasp_proper_dual_gradient_mode = gradient_mode
            raw_ema = controller_state.get("ema_mean_gaps", controller_state.get("ema_mean_costs"))
            if isinstance(raw_ema, list):
                self._ema_mean_gaps = [
                    None if value is None else float(value)
                    for value in raw_ema
                ]
            else:
                self._ema_mean_gaps = None
            self._steps_since_lambda_update = max(
                0, int(controller_state.get("steps_since_lambda_update", 0))
            )

        if self.cost_names and not self._pending_cost_sums:
            self._pending_cost_sums = [0.0] * len(self.cost_names)
        if self.cost_names and not self._pending_cost_counts:
            self._pending_cost_counts = [0.0] * len(self.cost_names)
        if self.cost_names and not self._pending_positive_gap_sums:
            self._pending_positive_gap_sums = [0.0] * len(self.cost_names)
        if self.cost_names and self.cost_weights and len(self.cost_weights) != len(self.cost_names):
            self.cost_weights = []
        if self.cost_names and self.dual_lambdas and len(self.dual_lambdas) != len(self.cost_names):
            self.dual_lambdas = []
        if (
            self.cost_names
            and self._pending_positive_gap_sums
            and len(self._pending_positive_gap_sums) != len(self.cost_names)
        ):
            self._pending_positive_gap_sums = [0.0] * len(self.cost_names)
        if self.cost_names and self._ema_mean_gaps and len(self._ema_mean_gaps) != len(self.cost_names):
            self._ema_mean_gaps = None

    def _distributed_reduce_stats(self, sums: list[float], count: int) -> tuple[list[float], int]:
        if not self.sync_across_ranks or self._sync_disabled_due_to_error:
            return sums, count

        try:
            world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size_env = 1
        effective_world_size = world_size_env

        try:
            import torch  # type: ignore[import-not-found]
            import torch.distributed as dist  # type: ignore[import-not-found]

            dist_available = dist.is_available()
            dist_initialized = dist_available and dist.is_initialized()
            if dist_initialized:
                effective_world_size = dist.get_world_size()

            if not dist_initialized:
                if effective_world_size > 1:
                    raise RuntimeError(
                        "Constrained GRPO sync_across_ranks=True but torch.distributed is not initialized "
                        "in a multi-process run (world_size>1). This would desynchronize dual variables across ranks."
                    )
                return sums, count

            backend = str(dist.get_backend()).lower()
            if "nccl" in backend and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")

            stats = torch.tensor(
                [float(value) for value in sums] + [float(count)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_count = int(stats[-1].item())
            total_sums = [float(value.item()) for value in stats[:-1]]
            return total_sums, total_count
        except Exception as exc:
            if effective_world_size > 1:
                raise RuntimeError(
                    f"Constrained GRPO distributed sync failed in multi-process run: {exc!r}. "
                    "Refusing rank-local fallback because it desynchronizes dual variables across ranks."
                ) from exc

            if not self._sync_warning_emitted:
                warnings.warn(
                    f"Constrained GRPO distributed sync failed in single-process context ({exc!r}); "
                    "using rank-local dual updates.",
                    RuntimeWarning,
                )
                self._sync_warning_emitted = True
            self._sync_disabled_due_to_error = True
            return sums, count

    def reset_pending(self):
        self._pending_cost_sums = [0.0] * len(self.cost_names)
        self._pending_cost_counts = [0.0] * len(self.cost_names)
        self._pending_positive_gap_sums = [0.0] * len(self.cost_names)
        self._pending_feasibility_sum = 0.0
        self._pending_constraint_sum = 0.0
        self._pending_objective_sum = 0.0
        self._pending_structure_sum = 0.0
        self._pending_count = 0

    def set_collect_stats(self, enabled: bool):
        self._collect_stats = bool(enabled)

    def set_cost_limit(self, cost_limit: float):
        value = float(cost_limit)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"cost_limit must lie in [0, 1], got {value}")
        self.cost_limit = value
        if self.cost_names:
            self.cost_limits = [
                self._cost_limit_for_name(name, value)
                for name in self.cost_names
            ]

    def set_dual_learning_rates(self, dual_lr_up: float, dual_lr_down: float | None = None):
        lr_up = float(dual_lr_up)
        lr_down = float(lr_up if dual_lr_down is None else dual_lr_down)
        if lr_up < 0.0 or lr_down < 0.0:
            raise ValueError(
                f"dual learning rates must be >= 0, got dual_lr_up={lr_up}, dual_lr_down={lr_down}"
            )
        self.dual_lr_up = lr_up
        self.dual_lr_down = lr_down

    def get_cost_names(self) -> list[str]:
        return list(self.cost_names)

    def get_last_constraint_stats(self) -> tuple[float | None, int]:
        constraint_scores = self._last_components.get("constraint_scores", [])
        if not constraint_scores:
            return None, 0
        values = [float(value) for value in constraint_scores]
        return float(sum(values) / len(values)), len(values)

    def get_last_components(self) -> dict:
        return {
            key: [float(value) for value in values]
            if key != "cost_names"
            else list(values)
            for key, values in self._last_components.items()
        }

    def enable_clasp_proper_dual(
        self,
        *,
        base_dual_lr: float,
        dual_update_interval: int,
        gradient_mode: str = "signed",
    ) -> None:
        """Activate CLASP's proper stochastic multiplier rule.

        Called from ``run_grpo.py`` when both ``args.use_clasp`` and
        ``args.clasp_proper_dual`` are True. When not called, the legacy
        EMA-based update (original ``update_lambda``) is used unchanged.
        """
        if base_dual_lr < 0.0:
            raise ValueError(f"base_dual_lr must be >= 0, got {base_dual_lr}")
        if dual_update_interval < 1:
            raise ValueError(
                f"dual_update_interval must be >= 1, got {dual_update_interval}"
            )
        normalized_gradient_mode = str(gradient_mode).strip().lower()
        if normalized_gradient_mode not in {"signed", "positive_excess"}:
            raise ValueError(
                "gradient_mode must be 'signed' or 'positive_excess', got "
                f"{gradient_mode!r}"
            )
        if (
            self._loaded_clasp_proper_dual_gradient_mode is not None
            and self._loaded_clasp_proper_dual_gradient_mode
            != normalized_gradient_mode
        ):
            raise RuntimeError(
                "Cannot enable CLASP proper dual with a different gradient estimator "
                "than the loaded checkpoint: checkpoint="
                f"{self._loaded_clasp_proper_dual_gradient_mode!r}, configured="
                f"{normalized_gradient_mode!r}."
            )
        self.clasp_proper_dual = True
        self.clasp_base_dual_lr = float(base_dual_lr)
        self.clasp_dual_update_interval = int(dual_update_interval)
        self.clasp_proper_dual_gradient_mode = normalized_gradient_mode
        self.dual_update_interval = int(dual_update_interval)

    def record_batch_feasibility_rate(self, feasibility_rate: float) -> None:
        """Record the most recent mini-batch feasibility rate.

        Consumed by :meth:`_update_lambda_clasp` under CLASP's proper
        multiplier-controller branch. No-op when CLASP proper dual is not active.
        """
        self._clasp_last_feasibility_rate = float(feasibility_rate)

    def update_lambda(self):
        if not self.cost_names:
            return

        if self.clasp_proper_dual:
            return self._update_lambda_clasp()

        try:
            world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size_env = 1
        if world_size_env > 1 and not self.sync_across_ranks:
            raise RuntimeError(
                "Constrained GRPO refuses rank-local dual updates in multi-process runs. "
                "Enable sync_across_ranks=True so every rank observes the same controller state."
            )

        sums = (
            list(self._pending_cost_sums)
            + list(self._pending_cost_counts)
            + [
                self._pending_constraint_sum,
                self._pending_objective_sum,
                self._pending_structure_sum,
            ]
        )
        count = int(self._pending_count)

        total_sums, total_count = self._distributed_reduce_stats(sums, count)
        self.reset_pending()

        if total_count <= 0:
            return

        num_costs = len(self.cost_names)
        total_cost_sums = total_sums[:num_costs]
        total_cost_counts = total_sums[num_costs : (2 * num_costs)]
        mean_masked_costs = [cost_sum / total_count for cost_sum in total_cost_sums]
        mean_mask_rates = [active_count / total_count for active_count in total_cost_counts]
        mean_gaps = [
            mean_cost - (limit_value * mean_mask)
            for mean_cost, limit_value, mean_mask in zip(
                mean_masked_costs,
                self.cost_limits,
                mean_mask_rates,
            )
        ]

        scalar_offset = 2 * num_costs
        mean_constraint = total_sums[scalar_offset] / total_count
        mean_objective = total_sums[scalar_offset + 1] / total_count
        mean_structure = total_sums[scalar_offset + 2] / total_count

        for stat_name, stat_value in {
                "mean_constraint": mean_constraint,
                "mean_objective": mean_objective,
                "mean_structure": mean_structure,
                **{
                    f"mean_masked_cost[{name}]": value
                    for name, value in zip(self.cost_names, mean_masked_costs)
                },
                **{
                    f"mean_mask_rate[{name}]": value
                    for name, value in zip(self.cost_names, mean_mask_rates)
                },
                **{
                    f"mean_gap[{name}]": value
                    for name, value in zip(self.cost_names, mean_gaps)
                },
            }.items():
            if not math.isfinite(stat_value):
                raise RuntimeError(
                    f"Constrained GRPO update received non-finite statistic {stat_name}={stat_value!r}. "
                    "Check reward heads for NaN/inf outputs."
                )

        if self._ema_mean_gaps is None:
            self._ema_mean_gaps = [None] * num_costs

        beta = self.violation_ema_beta
        for idx, current_gap in enumerate(mean_gaps):
            prev = self._ema_mean_gaps[idx]
            if prev is None:
                self._ema_mean_gaps[idx] = current_gap
            else:
                self._ema_mean_gaps[idx] = ((1.0 - beta) * prev) + (beta * current_gap)

        self._steps_since_lambda_update += 1
        if self._steps_since_lambda_update < self.dual_update_interval:
            return

        lambda_before = list(self.dual_lambdas)
        for idx, (name, mean_masked_cost, mean_mask_rate, mean_gap, ema_gap, limit_value, active_count) in enumerate(
            zip(
                self.cost_names,
                mean_masked_costs,
                mean_mask_rates,
                mean_gaps,
                self._ema_mean_gaps,
                self.cost_limits,
                total_cost_counts,
            )
        ):
            if ema_gap is None:
                continue
            dual_gradient = ema_gap
            dual_step_lr = self.dual_lr_up if dual_gradient >= 0.0 else self.dual_lr_down
            dual_step = dual_step_lr * dual_gradient
            self.dual_lambdas[idx] = max(
                0.0,
                min(self.lambda_max, self.dual_lambdas[idx] + dual_step),
            )
            if (
                self.dual_lambdas[idx] >= self.lambda_max
                and dual_gradient > 0.0
                and not self._lambda_saturation_warning_emitted
            ):
                warnings.warn(
                    f"Constrained GRPO dual weight for '{name}' reached lambda_max under positive gradient. "
                    "The corresponding cost target may be too strict or the reward parser may be too weak.",
                    RuntimeWarning,
                )
                self._lambda_saturation_warning_emitted = True

        self.update_count += 1
        self._steps_since_lambda_update = 0

        if self.log_every > 0 and (self.update_count % self.log_every == 0):
            cost_summary = ", ".join(
                (
                    f"{name}=masked_cost:{mean_masked_cost:.4f}/mask_rate:{mean_mask_rate:.4f}/"
                    f"gap:{mean_gap:.4f}/ema_gap:{ema_gap:.4f}/eps:{limit_value:.4f}/"
                    f"lambda:{new_lambda:.4f}/active:{active_count:.1f}"
                )
                for name, mean_masked_cost, mean_mask_rate, mean_gap, ema_gap, limit_value, new_lambda, active_count in zip(
                    self.cost_names,
                    mean_masked_costs,
                    mean_mask_rates,
                    mean_gaps,
                    self._ema_mean_gaps,
                    self.cost_limits,
                    self.dual_lambdas,
                    total_cost_counts,
                )
            )
            delta_summary = ", ".join(
                f"{name}:{after-before:+.4f}"
                for name, before, after in zip(self.cost_names, lambda_before, self.dual_lambdas)
            )
            if _is_primary_process():
                print(
                    "[ConstrainedGRPO] "
                    f"update_step={self.update_count} "
                    f"mean_constraint={mean_constraint:.4f} "
                    f"mean_objective={mean_objective:.4f} "
                    f"mean_structure={mean_structure:.4f} "
                    f"lambda_delta=[{delta_summary}] "
                    f"costs=[{cost_summary}]",
                    flush=True,
                )

    def _update_lambda_clasp(self) -> None:
        """CLASP proper stochastic multiplier controller.

        Key differences from the legacy update:

        * Uses the **current signed masked mean slack** (no EMA lag) by
          default, so the projected multiplier step can both increase and decrease a
          multiplier. ``positive_excess`` is retained only as an explicitly
          named one-sided ablation.
        * Step size is feasibility-conditional:
          ``eta = base_lr * sqrt((1 - feas + eps) / (feas + eps))``.
        * Multiplier updates happen every ``clasp_dual_update_interval`` primal
          steps, decoupling the primal/dual timescales.

        When the trainer has not yet recorded a feasibility rate, we fall
        back to 0.5 (neutral). The trainer callback in ``run_grpo.py`` feeds
        the batch feasibility rate on every log boundary.
        """
        try:
            world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size_env = 1
        if world_size_env > 1 and not self.sync_across_ranks:
            raise RuntimeError(
                "Constrained GRPO refuses rank-local dual updates in multi-process runs. "
                "Enable sync_across_ranks=True so every rank observes the same controller state."
            )

        self._steps_since_lambda_update += 1
        if self._steps_since_lambda_update < self.clasp_dual_update_interval:
            return

        count = int(self._pending_count)
        sums = (
            list(self._pending_cost_sums)
            + list(self._pending_cost_counts)
            + list(self._pending_positive_gap_sums)
            + [self._pending_feasibility_sum]
        )

        total_sums, total_count = self._distributed_reduce_stats(sums, count)

        if total_count <= 0:
            self.reset_pending()
            self._steps_since_lambda_update = 0
            return

        num_costs = len(self.cost_names)
        total_cost_sums = total_sums[:num_costs]
        total_cost_counts = total_sums[num_costs : 2 * num_costs]
        total_positive_gap_sums = total_sums[2 * num_costs : 3 * num_costs]
        aggregate_feas_rate = total_sums[3 * num_costs] / total_count

        mean_gradients: list[float | None] = []
        for idx, active_count in enumerate(total_cost_counts):
            if active_count <= 0.0:
                mean_gradients.append(None)
                continue
            if self.clasp_proper_dual_gradient_mode == "signed":
                gradient = (
                    float(total_cost_sums[idx]) / float(active_count)
                    - float(self.cost_limits[idx])
                )
            else:
                gradient = float(total_positive_gap_sums[idx]) / float(active_count)
            if abs(gradient) <= float(self.violation_tolerance):
                gradient = 0.0
            mean_gradients.append(float(gradient))

        feas_rate = (
            aggregate_feas_rate
            if math.isfinite(aggregate_feas_rate)
            else (
                self._clasp_last_feasibility_rate
                if self._clasp_last_feasibility_rate is not None
                else 0.5
            )
        )
        feas = max(1e-3, min(1.0 - 1e-3, float(feas_rate)))
        eta_t = self.clasp_base_dual_lr * math.sqrt((1.0 - feas + 1e-3) / (feas + 1e-3))

        lambda_before = list(self.dual_lambdas)
        for idx, gradient in enumerate(mean_gradients):
            if gradient is None:
                continue
            if not math.isfinite(gradient):
                raise RuntimeError(
                    f"CLASP multiplier update received non-finite gradient "
                    f"for '{self.cost_names[idx]}'."
                )
            step = eta_t * gradient
            self.dual_lambdas[idx] = float(
                max(0.0, min(self.lambda_max, self.dual_lambdas[idx] + step))
            )
            if (
                self.dual_lambdas[idx] >= self.lambda_max
                and gradient > 0.0
                and not self._lambda_saturation_warning_emitted
            ):
                warnings.warn(
                    f"CLASP dual weight for '{self.cost_names[idx]}' reached lambda_max "
                    "under positive gradient. Consider raising lambda_max or relaxing the "
                    "per-component cost_limit.",
                    RuntimeWarning,
                )
                self._lambda_saturation_warning_emitted = True

        self.update_count += 1
        self._steps_since_lambda_update = 0
        self.reset_pending()

        if self.log_every > 0 and (self.update_count % self.log_every == 0):
            delta_summary = ", ".join(
                f"{name}:{after - before:+.4f}"
                for name, before, after in zip(
                    self.cost_names, lambda_before, self.dual_lambdas
                )
            )
            gradient_summary = ", ".join(
                f"{name}:{'masked' if gradient is None else f'{gradient:.4f}'}"
                for name, gradient in zip(self.cost_names, mean_gradients)
            )
            if _is_primary_process():
                print(
                    "[CLASP-Dual] "
                    f"update_step={self.update_count} "
                    f"feas_rate={feas_rate:.3f} "
                    f"eta={eta_t:.4g} "
                    f"gradient_mode={self.clasp_proper_dual_gradient_mode} "
                    f"gradient=[{gradient_summary}] "
                    f"lambda_delta=[{delta_summary}]",
                    flush=True,
                )

    def __call__(self, prompts, completions, **kwargs):
        n = len(completions)
        if n == 0:
            return []

        structure_scores = _to_float_list(
            self.structure_fn(prompts, completions, **kwargs),
            n,
            "structure_fn",
        )
        objective_scores = _to_float_list(
            self.objective_fn(prompts, completions, **kwargs),
            n,
            "objective_fn",
        )

        if self.cost_components_fn is not None:
            cost_names, cost_vectors, cost_masks, constraint_scores = _normalize_cost_output(
                self.cost_components_fn(prompts, completions, **kwargs),
                n,
                "cost_components_fn",
            )
            for cost_idx in range(len(cost_names)):
                column = [row[cost_idx] for row in cost_vectors]
                clamped_column = self._validate_range(column, f"cost_components_fn[{cost_names[cost_idx]}]")
                for row, value in zip(cost_vectors, clamped_column):
                    row[cost_idx] = value
            constraint_scores = self._validate_range(
                constraint_scores,
                "cost_components_fn.constraint_scores",
            )
        else:
            cost_names, cost_vectors, constraint_scores = self._legacy_cost_components(
                prompts, completions, **kwargs
            )
            cost_masks = [[1.0] * len(cost_names) for _ in range(n)]

        self._initialize_cost_state(cost_names)

        weighted_gap_scores = [
            float(
                sum(
                    dual_weight * component_weight * mask * (cost - limit_value)
                    for dual_weight, component_weight, cost, mask, limit_value in zip(
                        self.dual_lambdas,
                        self.cost_weights,
                        row,
                        mask_row,
                        self.cost_limits,
                    )
                )
            )
            for row, mask_row in zip(cost_vectors, cost_masks)
        ]
        positive_weighted_gap_scores = [
            float(
                sum(
                    dual_weight * component_weight * mask * max(0.0, cost - limit_value)
                    for dual_weight, component_weight, cost, mask, limit_value in zip(
                        self.dual_lambdas,
                        self.cost_weights,
                        row,
                        mask_row,
                        self.cost_limits,
                    )
                )
            )
            for row, mask_row in zip(cost_vectors, cost_masks)
        ]
        raw_positive_gap_scores = [
            _lexicographic_recovery_violation_score(
                self.cost_names,
                row,
                mask_row,
                self.cost_weights,
                self.cost_limits,
            )
            for row, mask_row in zip(cost_vectors, cost_masks)
        ]
        dual_adjusted_cost_weights = []
        for name, component_weight, dual_lambda in zip(
            self.cost_names,
            self.cost_weights,
            self.dual_lambdas,
        ):
            effective_weight = float(component_weight) * (1.0 + float(dual_lambda))
            if not math.isfinite(effective_weight) or effective_weight < 0.0:
                raise RuntimeError(
                    "CLASP dual-aware recovery requires finite non-negative effective "
                    f"weights; {name!r} produced {effective_weight!r}."
                )
            dual_adjusted_cost_weights.append(effective_weight)
        dual_aware_recovery_gap_scores = [
            _lexicographic_recovery_violation_score(
                self.cost_names,
                row,
                mask_row,
                dual_adjusted_cost_weights,
                self.cost_limits,
            )
            for row, mask_row in zip(cost_vectors, cost_masks)
        ]
        constraint_violation_scores = []
        max_constraint_violation_scores = []
        feasibility_flags = []
        for row, mask_row in zip(cost_vectors, cost_masks):
            active_count = 0.0
            weighted_violation = 0.0
            max_positive_gap = 0.0
            for component_weight, cost, mask, limit_value, cost_name in zip(
                self.cost_weights,
                row,
                mask_row,
                self.cost_limits,
                self.cost_names,
            ):
                active = float(mask)
                if active <= 0.0:
                    continue
                if str(cost_name) in self.feasibility_ignore_cost_names:
                    continue
                positive_gap = max(0.0, float(cost) - float(limit_value))
                weighted_violation += float(component_weight) * active * positive_gap
                active_count += active
                max_positive_gap = max(max_positive_gap, positive_gap)
            constraint_violation_scores.append(
                float(weighted_violation / max(active_count, 1.0))
            )
            max_constraint_violation_scores.append(float(max_positive_gap))
            feasibility_flags.append(1.0 if max_positive_gap <= 1e-8 else 0.0)
        primal_scores = [
            float((self.objective_weight * objective) + (self.structure_weight * structure))
            for objective, structure in zip(objective_scores, structure_scores)
        ]
        combined_scores = [
            float(primal - weighted_gap)
            for primal, weighted_gap in zip(primal_scores, weighted_gap_scores)
        ]

        self._last_components = {
            "primal_scores": primal_scores,
            "objective_scores": [float(value) for value in objective_scores],
            "structure_scores": [float(value) for value in structure_scores],
            "weighted_gap_scores": weighted_gap_scores,
            "weighted_cost_scores": list(weighted_gap_scores),
            "positive_weighted_gap_scores": positive_weighted_gap_scores,
            "raw_positive_gap_scores": raw_positive_gap_scores,
            "dual_aware_recovery_gap_scores": dual_aware_recovery_gap_scores,
            "constraint_violation_scores": constraint_violation_scores,
            "max_constraint_violation_scores": max_constraint_violation_scores,
            "feasibility_flags": feasibility_flags,
            "constraint_scores": [float(value) for value in constraint_scores],
            "combined_scores": combined_scores,
            "cost_names": list(self.cost_names),
        }

        if self._collect_stats:
            for row, mask_row in zip(cost_vectors, cost_masks):
                for idx, (value, mask_value) in enumerate(zip(row, mask_row)):
                    observed_weight = float(mask_value)
                    if observed_weight <= 0.0:
                        continue
                    self._pending_cost_sums[idx] += float(value) * observed_weight
                    self._pending_cost_counts[idx] += observed_weight
                    self._pending_positive_gap_sums[idx] += (
                        observed_weight
                        * max(0.0, float(value) - float(self.cost_limits[idx]))
                    )
            self._pending_feasibility_sum += sum(feasibility_flags)
            self._pending_constraint_sum += sum(constraint_scores)
            self._pending_objective_sum += sum(objective_scores)
            self._pending_structure_sum += sum(structure_scores)
            self._pending_count += n

        self.call_count += 1
        return combined_scores


ConstrainedGRPOReward = PrimalDualConstrainedReward
