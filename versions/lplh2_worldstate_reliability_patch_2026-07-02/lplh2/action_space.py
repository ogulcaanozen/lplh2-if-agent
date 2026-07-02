"""Module 2: Action Space Learning.

Tracks validated verb-object pairings discovered during gameplay.
When an action is confirmed valid, it is decomposed into a verb template
and objects, then stored. During decision-making, known templates are
paired with current-location objects to suggest candidate actions.
"""

import logging
from itertools import permutations

logger = logging.getLogger(__name__)


class ActionSpace:
    """Learns and maintains valid action templates.

    Actions are decomposed into verb templates with ``&`` placeholders.
    Examples:
        "take sword" -> verb="take &", objects=["sword"]
        "put key in box" -> verb="put & in &", objects=["key", "box"]
    """

    def __init__(self):
        # {verb_template: set_of_objects}
        # e.g. {"take &": {"sword", "lamp"}, "open &": {"door", "mailbox"}}
        self.verbs = {}
        self.total_actions_learned = 0

    def reset(self):
        """Reset action space for a new game."""
        self.verbs = {}
        self.total_actions_learned = 0

    def store_action(self, verb: str, objects: list):
        """Store a validated verb-object pairing."""
        verb = verb.strip().lower()
        if not verb:
            return

        is_new_verb = verb not in self.verbs
        if is_new_verb:
            self.verbs[verb] = set()

        if "&" not in verb:
            if is_new_verb:
                self.total_actions_learned += 1
                logger.debug(f"Learned no-object verb: {verb}")
            return

        for obj in objects:
            obj_clean = obj.strip().lower()
            if obj_clean and obj_clean not in self.verbs[verb]:
                self.verbs[verb].add(obj_clean)
                self.total_actions_learned += 1
                logger.debug(f"Learned action: {verb} -> {obj_clean}")

    def get_action_pairs(self, current_objects: list) -> list:
        """Generate concrete candidate commands for current objects."""
        pairs = []
        objects_lower = [o.strip().lower() for o in current_objects]

        for verb in self.verbs:
            n_placeholders = verb.count("&")
            if n_placeholders == 0:
                continue

            if n_placeholders == 1:
                for obj in objects_lower:
                    concrete = verb.replace("&", obj, 1)
                    if concrete not in pairs:
                        pairs.append(concrete)
            else:
                for perm in permutations(objects_lower, n_placeholders):
                    concrete = verb
                    for obj in perm:
                        concrete = concrete.replace("&", obj, 1)
                    if concrete not in pairs:
                        pairs.append(concrete)

        return pairs

    def to_prompt_string(self, current_objects: list) -> str:
        """Format learned action templates and generated candidates."""
        pairs = self.get_action_pairs(current_objects)
        output = []

        if self.verbs:
            no_object_verbs = sorted(v for v in self.verbs if "&" not in v)
            object_templates = sorted(v for v in self.verbs if "&" in v)

            if no_object_verbs:
                output.append("Learned valid no-object commands:")
                for verb in no_object_verbs:
                    output.append(f"  - {verb}")
                output.append("")

            if object_templates:
                output.append("Learned valid object-command templates:")
                for verb in object_templates:
                    objs = sorted(self.verbs[verb])
                    obj_text = ", ".join(objs) if objs else "(no learned objects)"
                    output.append(f"  - {verb}  [learned objects: {obj_text}]")
                output.append("")
        else:
            output.append("No actions learned yet. Try exploring!")
            return "\n".join(output)

        if pairs:
            output.append("Generated candidate commands for current objects:")
            for pair in pairs:
                output.append(f"  - {pair}")
        else:
            output.append("No object-specific candidate commands generated for this room.")

        return "\n".join(output)

    def num_actions(self) -> int:
        """Return count of learned action entries."""
        return self.total_actions_learned
