"""Planning interface."""

from typing import Generic

from relational_structs import (
    GroundOperator,
    LiftedOperator,
    PDDLDomain,
    PDDLProblem,
    Predicate,
    Type,
)
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pddl_planner

from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation


class TaskThenMotionPlanningFailure(Exception):
    """Raised when task then motion planning fails."""


class TaskThenMotionPlanner(Generic[_Observation, _Action]):
    """Run task then motion planning with greedy execution."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        perceiver: Perceiver[_Observation],
        operators: set[LiftedOperator],
        skills: set[Skill[_Observation, _Action]],
        planner_id: str = "fd-sat",
        domain_name: str = "ttmp-domain",
    ) -> None:
        self._types = types
        self._predicates = predicates
        self._perceiver = perceiver
        self._operators = operators
        self._skills = skills
        self._planner_id = planner_id
        self._domain_name = domain_name
        self._domain = PDDLDomain(
            self._domain_name, self._operators, self._predicates, self._types
        )
        self._current_problem: PDDLProblem | None = None
        self._current_task_plan: list[GroundOperator] = []
        self._current_operator: GroundOperator | None = None
        self._current_skill: Skill | None = None

    def reset(self, obs: _Observation) -> None:
        """Reset on a new task instance."""
        objects, atoms, goal = self._perceiver.reset(obs)
        self._current_problem = PDDLProblem(
            self._domain_name, self._domain_name, objects, atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self._domain), str(self._current_problem), planner=self._planner_id
        )
        assert plan_str is not None
        self._current_task_plan = parse_pddl_plan(
            plan_str, self._domain, self._current_problem
        )
        self._current_operator = None
        self._current_skill = None

    def step(self, obs: _Observation) -> _Action:
        """Get an action to execute."""
        # Get the current atoms.
        atoms = self._perceiver.step(obs)

        # If there is no more plan to execute, fail.
        if not self._current_task_plan:
            raise TaskThenMotionPlanningFailure("Empty task plan")

        # If the current operator is None or terminated, get the next one.
        if self._current_operator is None or (
            self._current_operator.add_effects.issubset(atoms)
            and not (self._current_operator.delete_effects & atoms)
        ):
            self._current_operator = self._current_task_plan.pop(0)
            # Get a skill that can execute this operator.
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]
