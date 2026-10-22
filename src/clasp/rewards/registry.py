"""Task reward registry: the public lookup tables for per-task reward heads.

The parsing and scoring implementations live in :mod:`clasp.rewards.co_tasks`,
which remains the single source of truth; this module only re-exports the
registries so callers do not depend on the scoring internals.
"""

from clasp.rewards.co_tasks import TASK_COST_COMPONENTS, TASK_REWARD_TRIPLETS

__all__ = ["TASK_COST_COMPONENTS", "TASK_REWARD_TRIPLETS"]
