"""Tests for structs.py."""

from typing import Sequence

from relational_structs import LiftedOperator, Object, Predicate, Type

from task_then_motion_planning.structs import LiftedOperatorSkill


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
