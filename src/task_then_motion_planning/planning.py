"""Planning interface."""

from tomsutils.pddl_planning import run_pddl_planner
from relational_structs import GroundAtom, GroundOperator, PDDLDomain, PDDLProblem
from task_then_motion_planning.structs import Skill, _Observation, _Action
from typing import Generic, Optional

# TODO add predicate interpretation...

class TaskThenMotionPlanner(Generic[_Observation, _Action]):
    """Run task then motion planning with greedy execution."""

    def __init__(self, domain: PDDLDomain, planner_id: str = "fd-sat") -> None:
        self._domain = domain
        self._planner_id = planner_id
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._domain_str = str(domain)
        self._problem_str: str | None = None

    def reset(self, initial_obs: _Observation, goal: set[GroundAtom]) -> None:
        """Reset on a new task instance."""
        task_plan = run_pddl_planner(domain_str, problem_str, planner=self._planner_id)