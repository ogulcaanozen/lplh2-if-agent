"""Module 2: Action Space Learning.

Tracks all validated verb-object pairings discovered during gameplay.
When an action is confirmed valid, it is decomposed into verb + objects
and stored. During decision-making, known verbs are paired with current
location objects to suggest candidate actions.
"""

import logging
from itertools import permutations

logger = logging.getLogger(__name__)


class ActionSpace:
    """Learns and maintains the valid action space.
    
    Actions are decomposed into verb templates (with & placeholders)
    and associated objects. Example:
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
        """Store a validated verb-object pairing.
        
        Args:
            verb: The verb template (e.g., "take &", "north")
            objects: List of objects associated with this action
        """
        verb = verb.strip().lower()
        if not verb:
            return

        is_new_verb = verb not in self.verbs
        if is_new_verb:
            self.verbs[verb] = set()

        # Verbs with no "&" placeholder are no-object verbs (look, north, inventory, etc.).
        # They must not gain objects — even if the LLM hallucinates some.
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
        """Generate possible action-object pairs for the current location.
        
        This is the pairing(objloc, AS) function from the paper (Equation 4).
        Matches current location's objects with known verb templates.
        
        Args:
            current_objects: List of object names in the current location
            
        Returns:
            List of strings like "take sword", "open mailbox", etc.
        """
        pairs = []
        objects_lower = [o.strip().lower() for o in current_objects]

        for verb in self.verbs:
            n_placeholders = verb.count("&")
            if n_placeholders == 0:
                # No-object verb (directions, look, inventory, etc.) — skip pairing
                continue

            if n_placeholders == 1:
                # Single-object verb: pair with every object in current location
                # per paper Eq. 4: pairing(obj_loc, AS)
                for obj in objects_lower:
                    concrete = verb.replace("&", obj, 1)
                    if concrete not in pairs:
                        pairs.append(concrete)
            else:
                # Multi-object verb (e.g. "put & in &"): fill ALL placeholders.
                # Generate all ordered permutations of current objects of length
                # n_placeholders so every slot gets a distinct object.
                for perm in permutations(objects_lower, n_placeholders):
                    concrete = verb
                    for obj in perm:
                        concrete = concrete.replace("&", obj, 1)
                    if concrete not in pairs:
                        pairs.append(concrete)

        return pairs

    def to_prompt_string(self, current_objects: list) -> str:
        """Serialize action pairings for inclusion in the prompt.
        
        Args:
            current_objects: Objects in the current location
        """
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
        """Total number of unique verb-object pairs learned."""
        return self.total_actions_learned
