"""Tests for planning.py."""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    Predicate,
    Type,
)

from task_then_motion_planning.planning import TaskThenMotionPlanner
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver


def test_task_then_motion_planner():
    """Tests for TaskThenMotionPlanner()."""

    # Create the environment.
    env = gym.make("Taxi-v3")

    # Helper function for parsing observations.
    dropoff_locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

    def _parse_taxi_obs(
        i: int,
    ) -> tuple[tuple[int, int], tuple[int, int] | None, tuple[int, int]]:
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        taxi_row, taxi_col, pass_loc_idx, dest_idx = reversed(out)
        taxi_loc = (taxi_row, taxi_col)
        if pass_loc_idx > len(dropoff_locs):
            pass_loc: tuple[int, int] | None = None
        else:
            pass_loc = dropoff_locs[pass_loc_idx]
        dest_loc = dropoff_locs[dest_idx]
        return taxi_loc, pass_loc, dest_loc

    # Create types.
    taxi_type = Type("taxi")
    passenger_type = Type("passenger")
    destination_type = Type("destination")
    types = {taxi_type, passenger_type, destination_type}

    # Create predicates.
    TaxiEmpty = Predicate("TaxiEmpty", [taxi_type])
    InTaxi = Predicate("InTaxi", [passenger_type, taxi_type])
    AtDestination = Predicate("AtDestination", [passenger_type, destination_type])
    predicates = {TaxiEmpty, InTaxi, AtDestination}

    # Create perceiver.
    class TaxiPerceiver(Perceiver[int]):
        """A perceiver for the taxi environment."""

        def __init__(self) -> None:
            self._taxi = taxi_type("taxi")
            self._passenger = passenger_type("passenger")
            self._destinations = {
                (r, c): destination_type(f"dest-{r}-{c}") for r, c in dropoff_locs
            }

        def reset(
            self, obs: int
        ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
            objects = {self._taxi, self._passenger} | set(self._destinations.values())
            atoms, goal = self._parse_observation(obs)
            return objects, atoms, goal

        def step(self, obs: int) -> set[GroundAtom]:
            atoms, _ = self._parse_observation(obs)
            return atoms

        def _parse_observation(
            self, obs: int
        ) -> tuple[set[GroundAtom], set[GroundAtom]]:
            # Unpack the observation.
            _, pass_loc, dest_loc = _parse_taxi_obs(obs)

            # Create current atoms.
            atoms: set[GroundAtom] = set()
            if pass_loc is None:
                atoms.add(InTaxi([self._passenger, self._taxi]))
            else:
                atoms.add(TaxiEmpty([self._taxi]))
                atoms.add(
                    AtDestination([self._passenger, self._destinations[pass_loc]])
                )

            # Create goal atoms.
            goal = {AtDestination([self._passenger, self._destinations[dest_loc]])}

            return atoms, goal

    perceiver = TaxiPerceiver()

    # Create operators.
    passenger = passenger_type("?passenger")
    taxi = taxi_type("?taxi")
    destination = destination_type("?destination")
    PickUpOperator = LiftedOperator(
        "PickUp",
        [passenger, taxi, destination],
        preconditions={AtDestination([passenger, destination]), TaxiEmpty([taxi])},
        add_effects={InTaxi([passenger, taxi])},
        delete_effects={AtDestination([passenger, destination]), TaxiEmpty([taxi])},
    )

    DropOffOperator = LiftedOperator(
        "DropOff",
        [passenger, taxi, destination],
        preconditions={InTaxi([passenger, taxi])},
        add_effects={AtDestination([passenger, destination]), TaxiEmpty([taxi])},
        delete_effects={InTaxi([passenger, taxi])},
    )
    operators = {PickUpOperator, DropOffOperator}

    # Create skills.
    class PickUpSkill(LiftedOperatorSkill[int, int]):
        """Pick up skill."""

        def __init__(self) -> None:
            super().__init__()
            self._action_queue: list[int] = []

        def reset(self, ground_operator: GroundOperator) -> None:
            self._action_queue = []

        def _get_lifted_operator(self) -> LiftedOperator:
            return PickUpOperator

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
            # TODO use A star to implement

            return self._action_queue.pop(0)

    class DropOffSkill(LiftedOperatorSkill[int, int]):
        """Pick up skill."""

        def __init__(self) -> None:
            super().__init__()
            self._action_queue: list[int] = []

        def reset(self, ground_operator: GroundOperator) -> None:
            self._action_queue = []

        def _get_lifted_operator(self) -> LiftedOperator:
            return DropOffOperator

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
            # TODO use A star to implement

            return self._action_queue.pop(0)

    skills = {PickUpSkill(), DropOffSkill()}

    # Create the planner.
    planner = TaskThenMotionPlanner(types, predicates, perceiver, operators, skills)

    # Run an episode.
    obs, _ = env.reset(seed=123)
    planner.reset(obs)
    for _ in range(1000):  # should terminate earlier
        action = planner.step(obs)
        obs, _, done, _, _ = env.step(action)
        if done:  # goal reached!
            break
    else:
        assert False, "Goal not reached"
