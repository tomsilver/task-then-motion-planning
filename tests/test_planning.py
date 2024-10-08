"""Tests for planning.py."""

from __future__ import annotations

import abc
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    Predicate,
    Type,
)
from tomsutils.search import run_astar

from task_then_motion_planning.planning import TaskThenMotionPlanner
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver


def test_task_then_motion_planner():
    """Tests for TaskThenMotionPlanner()."""

    # Create the environment.
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    grid = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ]
    )

    # Uncomment for visualizations.
    # env = gym.wrappers.RecordVideo(env, "taxi-test")

    # Helper function for parsing observations.
    dropoff_locs = list(env.unwrapped.locs)

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
        if pass_loc_idx >= len(dropoff_locs):
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
            self,
            obs: int,
            info: dict[str, Any],
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
    class TaxiSkill(LiftedOperatorSkill[int, int]):
        """Shared functionality."""

        def __init__(self) -> None:
            super().__init__()
            self._action_queue: list[int] = []

        @abc.abstractmethod
        def _get_final_action(self) -> int:
            raise NotImplementedError

        def reset(self, ground_operator: GroundOperator) -> None:
            self._action_queue = []
            return super().reset(ground_operator)

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
            if self._action_queue:
                return self._action_queue.pop(0)

            taxi_loc, _, _ = _parse_taxi_obs(obs)
            _, _, destination = objects
            destination_loc = (
                int(destination.name.split("-")[1]),
                int(destination.name.split("-")[2]),
            )
            dest_r, dest_c = destination_loc
            initial_state = taxi_loc

            check_goal = lambda s: s == destination_loc

            delta_to_move_action = {
                (1, 0): 0,
                (-1, 0): 1,
                (0, 1): 2,
                (0, -1): 3,
            }

            def get_successors(s):
                r, c = s
                for (dr, dc), action in delta_to_move_action.items():
                    new_row, new_col = r + dr, c + dc
                    if not (
                        0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1]
                    ):
                        continue
                    if grid[new_row, new_col]:
                        continue
                    yield (action, (new_row, new_col), 1.0)

            def heuristic(s):
                r, c = s
                return abs(r - dest_r) + abs(c - dest_c)

            _, actions = run_astar(initial_state, check_goal, get_successors, heuristic)
            assert actions
            self._action_queue = list(actions) + [self._get_final_action()]

            return self._action_queue.pop(0)

    class PickUpSkill(TaxiSkill):
        """Pick up skill."""

        def _get_final_action(self) -> int:
            return 4  # pick up

        def _get_lifted_operator(self) -> LiftedOperator:
            return PickUpOperator

    class DropOffSkill(TaxiSkill):
        """Dropoff skill."""

        def _get_final_action(self) -> int:
            return 5  # drop off

        def _get_lifted_operator(self) -> LiftedOperator:
            return DropOffOperator

    skills = {PickUpSkill(), DropOffSkill()}

    # Create the planner.
    planner = TaskThenMotionPlanner(
        types, predicates, perceiver, operators, skills, planner_id="pyperplan"
    )

    # Run an episode.
    obs, info = env.reset(seed=123)
    planner.reset(obs, info)
    for _ in range(100):  # should terminate earlier
        action = planner.step(obs)
        obs, _, done, _, _ = env.step(action)
        if done:  # goal reached!
            break
    else:
        assert False, "Goal not reached"

    env.close()
