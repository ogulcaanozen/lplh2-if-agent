from types import SimpleNamespace
import unittest

from lplh2.agent import LPLHAgent
from lplh2.game_runner import (
    _engine_inventory_info,
    _engine_room_info,
    _engine_state_info,
)
from lplh2.kg_map import KGMap


class FakeEnv:
    def __init__(self, objects, player_location_num, inventory_nums=()):
        self.objects = {obj.num: obj for obj in objects}
        self.player_location_num = player_location_num
        self.inventory_nums = list(inventory_nums)

    def get_player_location(self):
        return self.objects[self.player_location_num]

    def get_object(self, number):
        return self.objects.get(int(number))

    def get_world_objects(self):
        return list(self.objects.values())

    def get_inventory(self):
        return [self.objects[number] for number in self.inventory_nums]


class UnsupportedEnv:
    def get_player_location(self):
        raise RuntimeError("room API unavailable")

    def get_inventory(self):
        raise RuntimeError("inventory API unavailable")


def zobj(number, name, parent=0, child=0, sibling=0):
    return SimpleNamespace(
        num=number,
        name=name,
        parent=parent,
        child=child,
        sibling=sibling,
    )


def test_engine_room_ascends_from_enclosure_to_top_level_location():
    room = zobj(100, "River Bank")
    boat = zobj(200, "small boat", parent=100)
    env = FakeEnv([room, boat], player_location_num=200)

    result = _engine_room_info(env)

    assert result["engine_location_available"] is True
    assert result["engine_location_num"] == 100
    assert result["engine_location_name"] == "River Bank"
    assert result["engine_enclosure_name"] == "small boat"


def test_engine_inventory_includes_objects_inside_carried_containers():
    sack = zobj(10, "brown sack", child=11)
    lantern = zobj(11, "brass lantern", parent=10, sibling=12)
    lunch = zobj(12, "lunch", parent=10)
    sword = zobj(20, "elvish sword")
    env = FakeEnv(
        [sack, lantern, lunch, sword],
        player_location_num=10,
        inventory_nums=[10, 20],
    )

    result = _engine_inventory_info(env)

    assert result["engine_inventory_available"] is True
    assert result["engine_inventory"] == [
        "brown sack", "brass lantern", "lunch", "elvish sword"
    ]
    assert [row["depth"] for row in result["engine_inventory_objects"]] == [
        0, 1, 1, 0
    ]


def test_engine_state_falls_back_cleanly_when_apis_are_unsupported():
    result = _engine_state_info(UnsupportedEnv())

    assert result["engine_location_available"] is False
    assert result["engine_inventory_available"] is False
    assert "unavailable" in result["engine_location_error"]
    assert "unavailable" in result["engine_inventory_error"]


def test_engine_room_numbers_are_stable_across_epochs_and_split_same_titles():
    kg = KGMap(strict_location_authority=True)
    first, first_id = kg.resolve_engine_location(
        101, title="Hallway", observation="Hallway. A red door is east.", epoch=1
    )
    kg.confirm_arrival(first)

    kg.reset()
    repeated, repeated_id = kg.resolve_engine_location(
        101, title="Hallway", observation="Hallway. The red door is open.", epoch=2
    )
    second, second_id = kg.resolve_engine_location(
        102, title="Hallway", observation="Hallway. Stairs lead down.", epoch=2
    )

    assert repeated == first
    assert repeated_id == first_id
    assert second != first
    assert second.endswith("#2")
    assert second_id != first_id
    assert kg.engine_num_for(repeated) == 101
    assert kg.engine_num_for(second) == 102


def test_engine_inventory_is_exact_and_removes_carried_objects_from_rooms():
    kg = KGMap(strict_location_authority=True)
    room, _ = kg.mint_room("Living Room", observation="Living Room")
    kg.confirm_arrival(room)
    kg.nodes[room]["have"] = ["brass lantern", "trophy case"]
    kg.inventory = ["hallucinated clasp"]

    kg.set_engine_inventory(["lantern", "elvish sword"])

    assert kg.inventory == ["lantern", "elvish sword"]
    assert kg.nodes[room]["have"] == ["trophy case"]


def test_agent_uses_engine_move_even_when_gate_claims_no_movement():
    agent = LPLHAgent.__new__(LPLHAgent)
    agent.kg_map = KGMap(strict_location_authority=True)
    agent.current_epoch = 1
    first, _ = agent.kg_map.resolve_engine_location(
        1, title="Outside", observation="Outside", epoch=1
    )
    agent.kg_map.confirm_arrival(first)
    agent._current_engine_location_num = 1
    agent.step_count = 1

    result = agent._resolve_step_location(
        auxiliary_gate={
            "decision": {"location_verdict": {"moved": "no", "room_title": ""}}
        },
        action="west",
        observation="Hallway. A door is north.",
        look_probe_text="Hallway. A door is north.",
        previous_location=first,
        done=False,
        engine_location_num=2,
        engine_location_name="Hallway",
    )

    assert result["resolution_mode"] == "jericho_engine"
    assert result["engine_location_changed"] is True
    assert result["action_transition_status"] == "engine_move_confirmed"
    assert agent.kg_map.current_location == "Hallway"
    assert agent._current_engine_location_num == 2


class JerichoStateGroundingTests(unittest.TestCase):
    def test_room_enclosure_ascent(self):
        test_engine_room_ascends_from_enclosure_to_top_level_location()

    def test_nested_inventory(self):
        test_engine_inventory_includes_objects_inside_carried_containers()

    def test_unsupported_fallback(self):
        test_engine_state_falls_back_cleanly_when_apis_are_unsupported()

    def test_stable_room_identity(self):
        test_engine_room_numbers_are_stable_across_epochs_and_split_same_titles()

    def test_exact_inventory_application(self):
        test_engine_inventory_is_exact_and_removes_carried_objects_from_rooms()

    def test_engine_move_overrides_gate(self):
        test_agent_uses_engine_move_even_when_gate_claims_no_movement()
