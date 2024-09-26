"""Tests for structs.py."""

from typing import Sequence

from relational_structs import GroundAtom, LiftedOperator, Object, Predicate, Type

from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver


def test_lifted_operator_skill():
    """Tests for LiftedOperatorSkill()."""
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    on = Predicate("On", [cup_type, plate_type])
    not_on = Predicate("NotOn", [cup_type, plate_type])
    cup_var = cup_type("?cup")
    plate_var = plate_type("?plate")
    parameters = [cup_var, plate_var]
    preconditions = {not_on([cup_var, plate_var])}
    add_effects = {on([cup_var, plate_var])}
    delete_effects = {not_on([cup_var, plate_var])}
    cup = cup_type("cup")
    plate = plate_type("plate")

    lifted_operator = LiftedOperator(
        "Pick", parameters, preconditions, add_effects, delete_effects
    )

    class PickSkill(LiftedOperatorSkill[int, int]):
        """Test skill for the Pick lifted operator."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return lifted_operator

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:

            cup, plate = objects

            assert cup.is_instance(cup_type)
            assert plate.is_instance(plate_type)

            return obs + 3

    skill = PickSkill()
    ground_operator = lifted_operator.ground((cup, plate))
    assert skill.can_execute(ground_operator)
    skill.reset(ground_operator)
    action = skill.get_action(5)
    assert action == 8


def test_perceiver():
    """Tests for Perceiver()."""
    cup_type = Type("cup_type")
    plate_type = Type("plate_type")
    on = Predicate("On", [cup_type, plate_type])
    not_on = Predicate("NotOn", [cup_type, plate_type])
    cup = cup_type("cup")
    plate = plate_type("plate")

    class CupPlatePerceiver(Perceiver[int]):
        """Test perceiver."""

        def reset(
            self, obs: int
        ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
            objects = {cup, plate}
            atoms = {GroundAtom(not_on, [cup, plate])}
            goal = {GroundAtom(on, [cup, plate])}
            return objects, atoms, goal

        def step(self, obs: int) -> set[GroundAtom]:
            if obs == 0:
                return {GroundAtom(not_on, [cup, plate])}
            return {GroundAtom(on, [cup, plate])}

    perceiver = CupPlatePerceiver()
    objects, atoms, goal = perceiver.reset(0)
    assert objects == {cup, plate}
    assert atoms == {GroundAtom(not_on, [cup, plate])}
    assert goal == {GroundAtom(on, [cup, plate])}
    atoms = perceiver.step(0)
    assert atoms == {GroundAtom(not_on, [cup, plate])}
    atoms = perceiver.step(1)
    assert atoms == {GroundAtom(on, [cup, plate])}
