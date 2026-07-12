# Thesis Notes

## Engine-grounded room identity

This experimental snapshot grounds room identity in Jericho's Z-machine object
table. The harness uses the player's stable location object number for internal
bookkeeping, in the same category as its existing use of score and terminal
state. A state-preserving `look` probe supplies the human-readable room label.
Object numbers are never shown to the gameplay LLM, whose observations,
memories, and decisions remain text-based.

The text/fingerprint location pipeline remains available whenever engine facts
are unavailable or `ENGINE_LOCATION_GROUNDING` is disabled.
