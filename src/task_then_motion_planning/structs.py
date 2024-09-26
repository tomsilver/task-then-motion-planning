"""Data structures."""

import abc
from typing import Generic, Sequence, TypeVar

from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
)

_Observation = TypeVar("_Observation")
_Action = TypeVar("_Action")


class Skill(abc.ABC, Generic[_Observation, _Action]):
    """In the context of this repository, a skill is responsible for executing
    operators, that is, taking actions to achieve the effects when the
    preconditions hold. Control flow is handled externally to the skill. For
    example, checking whether the operator effects have been satisfied, or
    checking if the skill has exceeded a max number of actions, happens outside
    the skill itself.

    The skill can internally maintain memory and so needs to be reset
    after each execution.
    """

    def __init__(self) -> None:
        self._current_ground_operator: GroundOperator | None = None

    @abc.abstractmethod
    def can_execute(self, ground_operator: GroundOperator) -> bool:
        """Determine whether the skill knows how to execute this operator.

        A typical implementation would have one skill per LiftedOperator
        and would check here if the ground_operator's parent matches.
        """

    @abc.abstractmethod
    def get_action(self, obs: _Observation) -> _Action:
        """Assuming that ground_operator can be executed, return an action to
        execute given the current observation.

        The internal memory may be updated assuming that the action will
        be executed.
        """

    def reset(self, ground_operator: GroundOperator) -> None:
        """Reset any internal memory given a ground operator that can be
        executed."""
        assert self.can_execute(ground_operator)
        self._current_ground_operator = ground_operator


class LiftedOperatorSkill(Skill[_Observation, _Action]):
    """A skill that is one-to-one with a specific LiftedOperator."""

    @abc.abstractmethod
    def _get_lifted_operator(self) -> LiftedOperator:
        """Return the lifted operator for this skill."""

    @abc.abstractmethod
    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: _Observation
    ) -> _Action:
        """Defines an object-parameterized policy."""

    def can_execute(self, ground_operator: GroundOperator) -> bool:
        return ground_operator.parent == self._get_lifted_operator()

    def get_action(self, obs: _Observation) -> _Action:
        assert self._current_ground_operator is not None
        objects = self._current_ground_operator.parameters
        return self._get_action_given_objects(objects, obs)


class Perceiver(abc.ABC, Generic[_Observation]):
    """Turns observations into objects, ground atoms, and goals.

    A perceiver may use internal memory to produce predicates, so it
    needs to be reset after every "episode".

    Assumes that object sets and goals do not change within an episode.
    """

    @abc.abstractmethod
    def reset(
        self, obs: _Observation
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Called at the beginning of each new episode.

        Resets internal memory and returns known objects, ground atoms
        in the initial state, and goal.
        """

    @abc.abstractmethod
    def step(self, obs: _Observation) -> set[GroundAtom]:
        """Get the current ground atoms and advance memory."""
