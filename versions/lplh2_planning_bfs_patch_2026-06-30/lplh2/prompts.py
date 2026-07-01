"""Prompt templates from the LPLH paper (Tables 4-9).

All prompts are kept as close to the original paper as possible,
with minor formatting adjustments for programmatic use.
"""

# ─────────────────────────────────────────────────────────────
# Table 4: Action Validation
# Determines if the previous action was successful or not
# ─────────────────────────────────────────────────────────────
ACTION_VALIDATION_PROMPT = """You are evaluating the outcome of a text-based game action based on the game's observation (feedback message) after the player's previous action. Your task is to determine if the action was successful or not.

<START OF INSTRUCTIONS>
- You will be given an observation text that follows the player's attempted action.
- If the observation indicates that the action was carried out successfully (e.g., it provides new information, describes the environment, or gives a positive confirmation), respond with:
<ais> True </ais>
- If the observation indicates that the action could not be performed (e.g., includes phrases like "You can't..." or "You cannot..."), respond with:
<ais> False </ais>
Note:
- An unsuccessful action usually explicitly states that the player cannot do something, or that the action fails.
<END OF INSTRUCTIONS>

Previous Action: {action}
Observation: {observation}"""


# ─────────────────────────────────────────────────────────────
# Table 5: Relation Extraction
# Extracts (subject, relation, object) triples from observations
# ─────────────────────────────────────────────────────────────
RELATION_EXTRACTION_PROMPT = """<START OF INSTRUCTIONS>
You're going to extract triples in the format <subject, relation, object> from an input Observation along with previous actions you did, originating from a text-based game. Focus solely on where the character ('You') is located, what objects are in that location, and their immediate properties. The maximum length for any object name in the triples is three words, where length of location name has no limit.

Rules:
1. If the observation doesn't describe an environment or information is insufficient (e.g., "Opened", "Taken"), output |start| none |end| and skip other points.
2. Always use 'in' as the relation to represent the character's location. Convert any spatial descriptions (e.g., 'are facing', 'are standing', 'are behind') to the 'in' relation. If the input begins with a Room name (starts with a capital letter and does not end with a period), use it as the location.
Example:
Input: "Stairwell (First Floor) You're in the north stairwell."
Triple: <You, in, Stairwell (First Floor)>
3. If the observation doesn't include a precise location, do not provide any <You, in, *> triple.
4. Use 'have' as the relation to represent interactive objects present in the location. Focus only on the objects themselves as the 'obj' in the triple. Ignore decorative details unless they indicate an interactive object. Limit object names to a maximum of three words.
Example:
Input: "There is a small mailbox here."
Triple: <[Location], have, mailbox>
5. Do not include additional details or properties of objects. Only extract the objects themselves, ensuring object names are no longer than three words. But if a object have a relation to another object, such as 'in' and 'on', then extract that relation.
Example:
Input: "A buzzing water fountain has been moved."
Triple: "<[Location], have, water fountain>"
Input: "A sock is on the table."
Triple: "<[Location], have, sock>, <[Location], have, table>, <sock, on, table>"
6. If the input specifies a requirement or action needed to continue, use <location/object, need/require, something to action>.
Example:
Input: "Forest. You would need a machete to go further west."
Triple: <Forest, need, machete to go west>
7. For objects or locations mentioned with a direction (e.g., 'to the north', 'up to', 'down'), use <current location, direction, [new location]/to [direction]>.
Example:
Input: "Hall. To the southwest is the entrance to the Computer Site, and you can go east here as well as go up with a stair."
Triples: <Hall, southwest, Computer Site>, <Hall, east, to east>, <Hall, up, to up>
Note: Pay more attention to objects and directions than to objects' states or other decorative details.
Now, extract the relationships for the input step by step and merge all the results into a single output enclosed within |start| * |end|, where * represents the list of extracted triples.
<END OF INSTRUCTIONS>

Previous Action: {action}
Observation: {observation}"""


# ─────────────────────────────────────────────────────────────
# Table 6: Splitting Action into Verb + Objects
# ─────────────────────────────────────────────────────────────
ACTION_SPLITTING_PROMPT = """<START OF INSTRUCTIONS>
You will receive a previous input(step) from a text-based IF game, and please split the input into two parts, action and objs, as "<verb; [objs]>". Please follow these instructions to complete the task step by step.

Use the following rules:
1. If the action is a simple directional command (e.g., "north" or "n"), the object list should be empty.
For example:
Input: "west"
Response: "<act> <west; []> </act>"
2. If the action is "take all" or another "all" command (e.g., "take all"), treat "take all" as the verb and leave the object list empty.
For example:
Input: "drop all"
Response: "<act> <drop all; []> </act>"
3. If there are objects following the Verb (e.g., "eat", "take") or Verb phrase (e.g., "drop down", "go around"), list them. If prepositions (e.g., "on", "at", "with") are present, include them in the verb phrase using "&" as a placeholder, and list each noun object individually.

Final Output:
Use <act> <verb; [objs]> </act> format for final output where: "verb" represents the action phrase with placeholders "&" for objects. objs is a list of object nouns.
<END OF INSTRUCTIONS>

Input: "{action}" """


# ─────────────────────────────────────────────────────────────
# Table 8: Experience Summarization
# Called when score changes (gain or loss/death)
# ─────────────────────────────────────────────────────────────
EXPERIENCE_SUMMARIZATION_PROMPT = """<START OF INSTRUCTIONS>
You are a game engine summarizer. Your task is to read the current log of the game state and produce a concise, cohesive summary of the player's progress so far (This happens every time the player gets a score or loses a score). Do NOT reveal any hidden or undiscovered information. Focus only on details the player already knows or has directly experienced.

A list of "Step" will be provided. Each step includes:
- An observation (what the player sees),
- Info about moves and current score,
- The action taken just before the observation.

**Summary Structure:**
1. "location": where the player is (or what area is described) when the score changes. If the player has died, give the location name before death.
*1.1* - One Location name Only.
*1.2* - Description of situation.
2. "puzzle_status": what puzzles or obstacles have been solved to earn/lose the points.
*2.1* - ONLY related steps to solve the puzzles directly. Any requirement for solving the puzzles, such as 'player need to <step>open door</step> at Room1 to enter <loc>Room2</loc>.
*2.2* - Description of the puzzle.
3. "scoring": how the player earned/lost points for the last step. Any action leads to earning/losing points.
*3.1* - Step done to earn/lose points.
*3.2* - How many points are changed?
4. "important_experience": The experience can be used for the future. Only the most notable and valuable clues or items the player learned about for the global game experience or any warning must be recorded through all previous logs. Only Focus on confirmed information.
*Earn Points* - ONLY when player earn points, then we only need to know what leads to earn points and ignore other unchecked information.
*Lose Points* - ONLY when the player loses points (died usually or lost in the game), you also need to give suggestions for the future.

**Remember**:
- If no related puzzles are encountered, the whole 'puzzle_status' needs to be "No puzzles encountered yet."
- Please focus on how the player scored points with related puzzles and situations that occurred.
- Do not reveal hidden or undiscovered info.
- Keep it concise and factual based on the logs.
- Use only direct evidence from the provided history. Do not infer hidden contents, hidden exits, object uses, or puzzle solutions that were not explicitly observed.
- If an observation says an object is empty, already open, closed, blocked, nailed shut, or otherwise unhelpful, record that exact fact instead of implying that it revealed new contents or opportunities.
- If a command failed or the parser rejected it, do not summarize it as a successful discovery.
- When giving "important_experience", please reflect like an expert player.
- If player has not died, the '*Lose Points*' in 'important_experience' should be 'none'. If player has died, the '*Earn Points*' in 'important_experience' should be 'none'.
- In your reasoning, if you find more than one earning or losing points, please ONLY summarize the last one based on previous steps.

**Final Output Format:**
- In the final output, mark location names as <loc>loc name</loc>, previous actions as <step>action</step>, and interacted objects as <obj>object</obj>.
- At the end of the response, please outline TAGs (no more than 4) as <tag>tag</tag> for retrieval. Put the main location in <room>room</room>.
- After TAGs, please also give the difficulty for current puzzles as <dif>difficulty</dif>.
- Please think about it first. Then, give your final completed player experience summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Score change: {reward_change}
Current score: {current_score}

Game History:
{history}"""


# ─────────────────────────────────────────────────────────────
# Table 9: LPLH Action Generation
# The main prompt for generating the next game command
# ─────────────────────────────────────────────────────────────
LPLH_ACTION_GENERATION_PROMPT = """<START OF INSTRUCTIONS>
**Instructions for Generating a Next Command in Text-Based Interactive Fiction**
---
**Objective** Craft a single, context-aware **next command** with its motivation that propels the game forward, based on the current map, recent actions, and history of attempts. This command should represent one immediate player action.
---
**Principles for Exploration, Puzzle-Solving, and Earning Points**
1. **Analyze the Current Game State**
- **Room & Map Details**: Assess where you are, noting any exits, known layout, and significant objects.
- **Recent Attempts**: Reflect on the previous actions, the motivation of taking that action and observation after this attempt.
- **Inventory Check**: Identify items on hand (keys, tools, etc.) that might solve current puzzles or overcome obstacles.
- **Inventory Is Authoritative**: Treat only items listed in the current inventory/map inventory as carried. A visible object, an opened object, or an object whose state changed is not in your possession unless it appears in inventory.
- **Visible Tools Must Be Taken First**: If a visible object could solve a stored situation but is not in inventory, consider taking it before using it or leaving the room. For example, if a light source is visible but not carried, preparation should include taking it before relying on it in a dark area.
- **Stored Situations**: Review unresolved hazards/blockers from earlier. If your current location, inventory, or known map makes one actionable now, consider addressing it; otherwise continue useful exploration.
- **Active Situation Plan**: If a plan is present, treat it as advisory. It marks a stored situation that may now be worth pursuing. You are not forced to follow it, but if it fits the current observation, inventory, and map, consider it seriously. Before following it, verify that any required item is actually in inventory; if it is only visible nearby, take it first.
- **BFS Navigation Hint**: If you decide to pursue the active plan and it includes a "route_found" navigation hint, use the hint's "next_command" as the immediate navigation command. Do not output a multi-step route as one command.
- **Affordance Agenda**: Review pending object/inventory commands and the already-tried commands attached to the same situation. Pending commands may include useful verbs not yet learned by the action space. Treat them as strong candidates when they directly apply to visible objects, inventory, stored situations, or recent failed syntax, but do not execute them blindly if navigation or another action is clearly better.
- **Condition-Level Agenda**: Some affordance agenda entries may have "kind": "condition"; these target a room/environment/perception/parser condition rather than one visible object. Consider them when recent observations suggest normal commands are being distorted, obscured, blocked, or mismatched.
- **Known Failed Commands Here**: These commands failed before at this location under the recorded world state. Avoid repeating them unless the current observation, inventory, visible objects, or score have changed enough to give a concrete reason to retry.
- **Same-State Tried Commands**: These commands were already tried from the exact same state snapshot shown now. Treat them as strong cautionary evidence. Prefer a different command unless you can name a concrete state difference or a strong reason the retry is still useful.
- **Objects & Interactions**: Focus on confirmed items or directions. If uncertain leads might advance the game, consider them cautiously.
- **Action Selection**: Only choose to interact with an object (or perform an action) if you're confident it will move the story forward.
2. **Use Retrieved Experiences and Past Attempts**
- **Relevance**: Apply past successes or observed clues that align with the current room or situation.
- **Avoid Repetition**: Do not repeat failing commands indefinitely. If a command fails, adjust strategy.
- **Focus on Gains**: Prioritize moves likely to unlock new paths, uncover essential items, or yield valuable information.
3. **Formulate a Single Effective Command**
- **One Action**: Provide exactly one executable command.
- **Purpose**: Briefly ensure it's the most logical next step, considering both context and success likelihood.
- **Move command**: The full directions are ['north', 'south', 'east', 'west', 'southeast', 'southwest', 'northeast', 'northwest', 'up', 'down']
4. **Output Format**
- Present the final command and a short motivation in the following format without extra commentary:

Your internal reasoning steps Here.
|start|
<com>[command]</com>
<rea>[short motivation for the decision-making reason]</rea>
|end|

---
**Adaptation and Fallback Rules**
1. **Priority Usage**
- **Highest Priority**: Current inventory and active situation plans that are concretely relevant now.
- **Next**: Visible objects, pending affordance agenda commands, and confirmed exits in the current room.
- **Then**: Untried exits/frontier entries when exploration is the best next step.
2. **Conflict Resolution**
- Treat prior attempts known to fail at this location or context as cautionary evidence.
- Avoid exact commands from the Known Failed Commands Here list unless the current state has meaningfully changed and you can justify retrying.
- Do not say or assume "I have/carry/possess X" unless X appears in inventory. If X is in visible objects or known object locations instead, treat it as available to pick up, not already carried.
- Validate uncertain ('may_') directions or items before fully committing to them.
- After verify all the exits in one room then you can fully trust the map.
3. **Fallback Strategies**
- If uncertain, explore unvisited areas or re-examine ('look') the current room.
- Look for overlooked clues or alternative ways forward.
4. **Exploratory Commands**
- Use pending affordance-agenda commands to try reasonable interactions with visible objects and carried items, especially after a pure navigation loop.
- If the affordance agenda contains pending object/inventory/stored-situation commands for the current location, consider trying one before generic navigation, unless the current observation suggests navigation is more urgent. Use the already-tried entries as cautionary evidence.
- If the affordance agenda contains condition-level commands for the current location, consider addressing the condition itself before repeating object commands that produced abnormal or mismatched observations.
- If tools are available, think of how to use them on obstacles.
- In case an exploration fails, attempt a different angle: return to a previous room, look around again, or try another approach.
- **Explore the world**: It's better to try all directions in each room to identify the exit and update the game map. For 'may_direction', consider testing that path (e.g., "north").
---
**Remember**: You are navigating a text-based world. Combine current observations with past knowledge to decide the best single move.
<END OF INSTRUCTIONS>

=== CURRENT GAME MAP ===
{kg_map}

=== AVAILABLE ACTIONS FOR OBJECTS HERE ===
{action_pairs}

=== RETRIEVED EXPERIENCES ===
{experiences}

=== ACTIVE STORED SITUATIONS ===
{stored_situations}

=== ACTIVE SITUATION PLAN (ADVISORY) ===
{active_plan_context}

=== CURRENT SCORE ===
{score}

=== AFFORDANCE AGENDA (PENDING VS TRIED COMMANDS) ===
{brainstormed_command_ideas}

=== KNOWN FAILED COMMANDS HERE ===
{known_failed_commands_here}

=== SAME-STATE TRIED COMMANDS ===
{same_state_tried_commands}

=== RECENT HISTORY (last {history_length} turns) ===
{history}

=== CURRENT OBSERVATION ===
{observation}"""


# ─────────────────────────────────────────────────────────────
# LPLH2 Enhancement: Neutral State Experience Prompts
# Four separate prompts, one per neutral trigger type.
# These fire when reward_change == 0 but a meaningful event occurred.
# ─────────────────────────────────────────────────────────────

# Trigger 1: Agent enters a previously unvisited location
NAVIGATION_EXPERIENCE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing a navigation event in a text-based game. The player has just entered a location that should be remembered for future decisions.

Return a summary only if the observation gives a concrete reusable fact: a room name, an exact route, visible objects, exits, obstacles, warnings, or directly stated constraints. If the observation does not contain a concrete reusable fact, output exactly:
|start| none |end|

**Summary Structure:**
1. "location": The exact room/location name.
2. "route_confirmed": The exact previous location and action that led here.
3. "visible_objects": Objects/features explicitly visible in the observation. Use "none" if none are stated.
4. "exits_or_obstacles": Exits, blocked paths, doors, windows, dangers, or constraints explicitly stated.
5. "reusable_lesson": A concrete memory sentence in this form: "From <loc>X</loc>, <step>Y</step> reaches <loc>Z</loc>; Z contains/has/allows ...".
6. "evidence": The exact observed fact that supports the lesson.

**Remember:**
- Only record what is directly stated in the observation or supplied current/previous location/action fields.
- Do not infer hidden exits, hidden objects, puzzle solutions, or future uses.
- Do not write generic advice like "explore carefully", "may be useful later", or "could reveal clues".
- If a path is blocked or the action did not actually move the player, output |start| none |end|.
- Keep it concise and factual.

**Output Format:**
- Mark locations as <loc>loc name</loc>, actions as <step>action</step>, objects as <obj>object</obj>.
- End with retrieval tags (max 4) as <tag>tag</tag>. Include the room name as <room>room</room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location (newly entered): {location}
Previous Location: {prev_location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 2: Agent examines an object, reads a note, or talks to someone
NARRATIVE_EXPERIENCE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing an information-retrieval event in a text-based game. The player examined, read, inspected, asked, or talked and received content.

Return a summary only if the observation reveals a concrete reusable fact: an object's contents, a visible clue, an exact warning, an instruction, an object description that changes what the player should do, or a direct constraint. If the observation is flavor text, a generic intro, "nothing special", empty, or not operationally useful, output exactly:
|start| none |end|

**Summary Structure:**
1. "location": Where this happened.
2. "source": The exact object/person/text examined or read.
3. "confirmed_content": The concrete information revealed, stated without embellishment.
4. "reusable_lesson": A concrete memory sentence in this form: "In <loc>X</loc>, <step>Y</step> reveals/shows ...".
5. "evidence": The exact observed fact that supports the lesson.

**Remember:**
- Only record content explicitly stated in the observation.
- Do not speculate about recipes, hidden uses, hidden objects, puzzle solutions, future dangers, or progression unless the observation directly states them.
- Do not write generic advice like "pay attention to details" or "this may be important later".
- If the observation has no concrete reusable fact, output |start| none |end|.
- Keep it concise and factual.

**Output Format:**
- Mark locations as <loc>loc name</loc>, actions as <step>action</step>, objects as <obj>object</obj>.
- End with retrieval tags (max 4) as <tag>tag</tag>. Include the room name as <room>room</room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 3: An action changes the environment (opens a path, toggles a switch, etc.)
ENVIRONMENTAL_CHANGE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing an environmental change event in a text-based game. The player performed an action that may have changed the world without directly earning points.

Return a summary only if the observation confirms a concrete state change or newly visible concrete information: a container opened and revealed listed contents, a door/window/path opened or closed, a mechanism changed state, a light changed state, or a specific object appeared/disappeared. If the observation says already open/closed, empty, blocked, unchanged, or gives no concrete new state, output exactly:
|start| none |end|

If the observation indicates that the latest command was rejected, impossible,
not understood, missing a required object, or mocked as nonsensical, do not
summarize that command as a successful environmental change. Output none even
if the same observation also includes an unrelated side effect or ambient event.

**Summary Structure:**
1. "location": Where this happened.
2. "trigger_action": The exact action that caused the change.
3. "confirmed_change": What changed, using only observed facts.
4. "newly_visible_objects_or_access": Exact objects revealed or exact access enabled. Use "none" if no objects/access are explicitly stated.
5. "reusable_lesson": A concrete memory sentence in this form: "In <loc>X</loc>, <step>Y</step> causes/reveals/enables ...".
6. "evidence": The exact observed fact that supports the lesson.

**Remember:**
- Focus on the exact action and exact observed effect.
- Do not infer hidden contents, hidden paths, future uses, puzzle solutions, or broad strategy.
- Do not write generic advice like "opening containers can reveal useful resources" or "may unlock future paths".
- Do not say a container revealed contents unless the observation explicitly lists the contents.
- If no concrete new state is confirmed, output |start| none |end|.

**Output Format:**
- Mark locations as <loc>loc name</loc>, actions as <step>action</step>, objects as <obj>object</obj>.
- End with retrieval tags (max 4) as <tag>tag</tag>. Include the room name as <room>room</room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 3 gate: LLM decides whether a valid non-movement action changed the world.
ENVIRONMENTAL_CHANGE_DETECTION_PROMPT = """<START OF INSTRUCTIONS>
You are deciding whether the latest valid action directly changed the game
world state in a text-based interactive fiction game.

Return true only when the observation confirms a concrete state change caused
by the latest action. Return false for narrative information, flavor text,
ordinary room descriptions, parser errors, failed actions, or merely learning
about a possible place without changing/accessing it.

**Environmental change means one or more of these is directly observed:**
- A door, window, gate, grating, container, passage, mechanism, lock, light,
  object, or room state changed.
- A new object, exit, passage, or access became visible or usable because of
  the action.
- An object moved, appeared, disappeared, opened, closed, unlocked, turned on,
  turned off, broke, released, or otherwise changed state.
- The player successfully used a tool or object to alter the environment.

**Not an environmental change:**
- Reading/examining/talking reveals information but does not alter the world.
- The text merely mentions a door, passage, object, or possible route.
- The action was blocked, rejected, misunderstood, or had no effect.
- The player only moved to a new room; navigation is handled separately.

**Output Format:**
|start|
{{
  "environmental_change": true,
  "evidence": "short exact evidence from the observation"
}}
|end|

If there is no concrete change:
|start|
{{
  "environmental_change": false,
  "evidence": ""
}}
|end|

**Examples:**

Action: open door
Observation: The door is now open.
Output:
|start|
{{"environmental_change": true, "evidence": "The door is now open."}}
|end|

Action: read book
Observation: The book reveals that a secret passage lies to the north.
Output:
|start|
{{"environmental_change": false, "evidence": ""}}
|end|

Action: move rug
Observation: With a great effort, the rug is moved to one side, revealing a trap door.
Output:
|start|
{{"environmental_change": true, "evidence": "the rug is moved to one side, revealing a trap door"}}
|end|

Action: unlock grating
Observation: The grating is now unlocked.
Output:
|start|
{{"environmental_change": true, "evidence": "The grating is now unlocked."}}
|end|

<END OF INSTRUCTIONS>

Current Location: {location}
Action: {action}
Observation: {observation}
Inventory: {inventory}
Visible Objects: {visible_objects}
Active Stored Situations: {active_situations}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Auxiliary Module Gate
# Routes selected expensive auxiliary LLM modules for one completed step.
# ---------------------------------------------------------------------
AUXILIARY_MODULE_GATE_PROMPT = """<START OF INSTRUCTIONS>
You are the auxiliary-module gate for a text-based interactive fiction agent.

Your job is NOT to choose the next game command. Your job is to decide which
selected auxiliary mechanisms deserve to run after the latest completed action.

Make exactly these decisions:

0. Latest command outcome
   This is independent from the FM validity label. Decide whether the latest
   command itself was accepted by the game.
   - "accepted": the observation confirms the command worked, produced useful
     information, changed location, changed object/world state, or clearly did
     what the player asked.
   - "rejected": the observation says the command was not understood, not
     possible, missing an object/tool, parser-rejected, or mocked as nonsensical
     (for example "I don't know the word", "You can't", "You don't have that",
     "You can't see any", or "fighting a crack?").
   - "no_effect": the command was understood but produced no new effect, such as
     "already open", "already on", or "nothing special".
   - "unknown": ambiguous observations where the command outcome is not clear.

1. Summary trigger decisions
   These decide whether the summary module should try to create a concrete,
   reusable memory for this step. Set "run": true only when the observation
   contains a confirmed fact worth remembering. Set false for parser errors,
   repeated descriptions, ordinary failed commands, or vague speculation.

   - "navigation": true when the observation confirms useful spatial knowledge:
     a new/current location, a route from the previous location, exits, blocked
     directions, reachable areas, or a room/object placement fact tied to place.
     This is especially likely when Current Location differs from Previous
     Location or when Current Location was not in Rooms Visited Before This Step.
   - "environmental": true when the latest valid non-navigation action appears
     to have directly changed world/object state, revealed a state transition, or
     caused a durable effect. If the score changed because of the action, prefer
     true unless the evidence says the score came from ending/administrative
     text. Do not set true when command_outcome is "rejected" or "no_effect",
     even if an unrelated side effect or room event appears in the observation.
   - "narrative": true when the observation gives reusable information without
     necessarily changing world state: readable text, a clue, warning,
     instruction, object property, game rule, or important fact learned from
     examining/listening/talking/reading.

2. Inventory update
   Decide whether the latest observation gives concrete evidence that carried
   inventory should change. This should be semantic and observation-based, not a
   guess. Return changed=false for vague theft/loss hints that do not identify a
   concrete item and are not an explicit inventory listing.
   - If Previous Action is an inventory command and the observation lists carried
     items, set authoritative=true and put the top-level carried objects in
     items_now_carried. Ignore container contents unless they are carried
     directly; for example water inside a bottle is not a top-level carried item.
   - If the observation clearly confirms a take/get/drop/eat/drink/give/loss or
     a possession denial tied to the object in Previous Action, use items_added
     or items_removed.
   - Prefer item names already present in Current Inventory or Inventory Before
     This Step when identifying removed items. Use concise object names such as
     "lantern" rather than "brass lantern" when they refer to the same carried
     item.
   - Do not infer a specific removed item from generic text such as "robbed you
     blind" unless the observation or Previous Action concretely identifies it.

3. Stored situation detection
   - Set "run": true when the latest observation may contain a new unresolved
     future-return situation: darkness, danger, locked/nailed/blocked access,
     missing condition, inaccessible object/path, or a problem that may become
     solvable after finding an item, command, route, or changed world state.
   - Set false for ordinary visible objects, normal room descriptions, generic
     "can't go that way" boundaries, parser errors with no blocker, or already
     remembered situations.

4. Affordance brainstorming
   - Set "run": true when fresh local/inventory command ideas may help the next
     action selector: visible objects/features, inventory interactions, active
     situations that may now be addressable, recent failures needing alternate
     wording, meaningful score/inventory/world change, or no cached ideas for
     this state.
   - Also set "run": true when Recent Command Outcomes Here suggest that a
     persistent environmental, perceptual, mental, or parser-like condition is
     distorting normal command interpretation. General evidence includes
     several different commands in the same location producing similarly
     repeated, echoed, garbled, obscured, blocked, mismatched, or
     condition-dominated observations.
   - This condition signal overrides the cached-ideas skip. Cached object-level
     ideas may not address the condition itself, so fresh brainstorming should
     run to reconsider condition-level options.
   - Set false when the same state already has cached affordance ideas and there
     is no meaningful new object, inventory, situation, failure, or world-change
     signal.

5. Active situation planning
   Decide whether the latest step makes an already stored situation worth
   actively considering now. This is not a command choice and it is not forced.
   It only creates an advisory plan for the main action LLM.
   Current Inventory is authoritative. Do not claim the agent is carrying,
   holding, or has an item unless it appears in Current Inventory or the KG map
   inventory. A visible object, an opened object, or an object whose state
   changed is not carried merely because it is present or usable in the room.
   - Set "create": true when Current Inventory, a newly found item/object,
     learned information, changed world state, or current position gives a
     concrete reason to return to or address one Active Stored Situation.
   - Good general examples: now carrying a light source for a remembered dark
     area; now carrying a key/tool for a remembered locked/nailed/blocked
     access; now having a weapon/protection for a remembered danger; now having
     opened a route that reaches the situation location.
   - If the useful object is visible or known in the current room but not in
     Current Inventory, suggested_preparation should include taking it before
     using it or traveling to the situation. Do not describe that as "now
     carrying" the object.
   - Set false when there are no Active Stored Situations, when the new item is
     generic with no clear relation, or when Active Plan Already Stored already
     targets the same situation.
   - The target_location should be the stored situation's location or the known
     room to navigate toward. commands_to_try_at_target should be immediate
     text-game commands to consider after reaching the target, not a long route.

Return JSON only between |start| and |end|:

|start|
{{
  "command_outcome": {{
    "status": "accepted, rejected, no_effect, or unknown",
    "reason": "short observation-based reason"
  }},
  "summary_triggers": {{
    "navigation": {{
      "run": true or false,
      "evidence": "short observation-based reason"
    }},
    "environmental": {{
      "run": true or false,
      "evidence": "short observation-based reason"
    }},
    "narrative": {{
      "run": true or false,
      "evidence": "short observation-based reason"
    }}
  }},
  "inventory_update": {{
    "changed": true or false,
    "authoritative": true or false,
    "items_now_carried": ["only", "top-level", "carried", "items"],
    "items_added": ["items concretely gained this step"],
    "items_removed": ["items concretely no longer carried"],
    "reason": "short reason; if no concrete evidence, explain why changed is false"
  }},
  "stored_situation_detection": {{
    "run": true or false,
    "reason": "short reason"
  }},
  "affordance_brainstorming": {{
    "run": true or false,
    "reason": "short reason",
    "focus": ["optional", "short", "targets"]
  }},
  "situation_plan": {{
    "create": true or false,
    "target_location": "known target room/location, or empty",
    "related_situation": {{
      "location": "copy from Active Stored Situations, or empty",
      "situation": "copy from Active Stored Situations, or empty"
    }},
    "reason": "short observation-based reason",
    "suggested_preparation": ["optional commands before going there"],
    "commands_to_try_at_target": ["commands to consider once at the target"]
  }}
}}
|end|

Use only the provided state. Do not invent hidden game knowledge.
<END OF INSTRUCTIONS>

Current Location: {location}
Previous Location: {previous_location}
Previous Action: {action}
Action Validity From FM: {action_valid}
Observation After Action: {observation}
Current Score: {score}
Reward Change: {reward_change}
Rooms Visited Before This Step: {rooms_visited_before}
Inventory Before This Step: {inventory_before}
Current Inventory: {inventory}
Current KG Map / World State: {kg_map}
Visible Objects Here: {visible_objects}
Active Stored Situations: {active_situations}
Active Plan Already Stored: {active_plan}
Recent Failed Commands: {recent_failed_commands}
Known Failed Commands Here: {known_failed_commands_here}
Recent Command Outcomes Here: {recent_command_outcomes}
Same-State Tried Commands: {same_state_tried_commands}
Cached Affordance Ideas Available For This State: {cached_affordance_ideas_available}"""


# Trigger 4: Agent finds a valid command after 2+ consecutive failures
ERROR_CORRECTION_PROMPT = """<START OF INSTRUCTIONS>
You are summarising a command-discovery event in a text-based game. After failed attempts, the player found a command that the game understood and accepted.

Return a summary only if the failed commands and successful command are clearly alternative attempts at the same goal or object, and the successful observation confirms a concrete improvement. If the successful command is merely a generic move/look, unrelated to the failed commands, or does not teach a specific reusable syntax pattern, output exactly:
|start| none |end|

If the observation rejects, mocks, or fails the latest command, output none. Do
not treat unrelated side effects in the same observation as evidence that the
latest command was the correct command.

**Summary Structure:**
1. "location": Where this happened.
2. "goal": The shared goal/object that the failed and successful commands were trying to address.
3. "correct_command": The exact command that succeeded.
4. "failed_attempts": The failed commands that were genuinely alternative attempts at the same goal. Exclude unrelated failures.
5. "pattern_learned": The narrow command pattern learned. Do not generalize beyond this case.
6. "evidence": The exact observed fact that shows the successful command worked.

**Remember:**
- Store only precise syntax lessons, not broad exploration advice.
- Good example: failed <step>enter window</step>, successful <step>go through window</step> teaches that this game accepts <step>go through window</step> for entering the open window.
- Bad example: failed <step>examine sack</step> and successful <step>examine table</step> do not teach a syntax pattern; output none.
- Do not call a command correct unless the observation confirms it changed state, revealed information, moved location, or otherwise clearly succeeded.
- If there is no narrow reusable command pattern, output |start| none |end|.

**Output Format:**
- Mark locations as <loc>loc name</loc>, actions as <step>action</step>, objects as <obj>object</obj>.
- End with retrieval tags (max 4) as <tag>tag</tag>. Include the room name as <room>room</room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Successful Command: {action}
Observation: {observation}
Recent Failed Commands: {failed_attempts}"""


# ─────────────────────────────────────────────────────────────
# LPLH2 Enhancement: Stored Situation Detection
# Detects unresolved future-return problems, not local affordances.
# ─────────────────────────────────────────────────────────────
STORED_SITUATION_DETECTION_PROMPT = """<START OF INSTRUCTIONS>
You are detecting unresolved situations in a text-based game.

A "stored situation" is a concrete problem, hazard, blocker, or missing-condition situation that should be remembered for later because it may become solvable after the player finds a useful object, learns a command, changes the world, or prepares differently.

Return a stored situation only if the latest observation reveals a NEW unresolved situation that is not already present in the stored situation list.

**Hard Decision Test:**
Before storing anything, ask:
"Would a future item, command, changed world state, or preparation plausibly make this situation solvable?"

- If yes, store one concise situation.
- If no, output exactly: |start| none |end|

**Store Only Strong Signals:**
- Darkness, lethal hazards, or unsafe areas that likely require preparation or equipment.
- Locked, nailed, closed, sealed, or otherwise blocked access that may require a key, tool, command, or world change.
- An unreachable object/path with a stated condition or missing requirement.
- A room description that names a concrete future-return problem, such as a dark staircase, dark chimney, locked gate, dangerous area, or inaccessible passage.
- An explicit missing condition, e.g. needing light, a key, a tool, protection, strength, or another route.
- A concrete environmental, perceptual, mental, or parser-like condition that
  appears to interfere with normal command results, such as darkness, noise,
  silence, confusion, fog, smoke, being underwater, blindness, magical
  interference, or observations that show commands are being distorted.

**Do NOT Store:**
- Generic failed movement responses such as "You can't go that way" or "There is no exit in that direction."
- Permanent map boundaries such as impassable mountains, storm-tossed trees, or scenery that simply blocks exploration.
- Ordinary visible objects that can be interacted with now. Those belong to local command brainstorming.
- Normal exits, normal room descriptions, flavor text, or interesting objects with no current blocker.
- A situation that is already present in Already Stored Situations, even if phrased differently.
- A situation only because an ordinary object is visible and can be interacted with now; local brainstorming will handle that.
- Do not skip a dark, dangerous, locked, nailed, or missing-condition situation merely because other ordinary objects are also visible in the room.
- Skip storing only when the current inventory clearly contains a directly relevant solution that can be tried immediately for this exact problem; otherwise remember the problem for later.

**Situation Structure:**
1. "location": The exact place where the unresolved situation exists, or the place the player should return to later.
2. "situation": A short concrete sentence describing the unresolved problem.

**Output Format:**
- If there is no new stored situation:
|start| none |end|

- If there is a new stored situation:
|start|
{{
  "location": "...",
  "situation": "..."
}}
|end|

**Field Meanings:**
- "location": Use the current location when possible. If the situation is tied to a nearby connected area, describe it concisely, e.g. "Kitchen / dark upstairs area".
- "situation": Describe only the unresolved problem and direct evidence. Keep
  it short, factual, and based on the observation. Do not prescribe a remedy
  unless the observation directly states one.

**Good Examples:**

Observation: "It is pitch black. You are likely to be eaten by a grue."
Current Inventory: []
Output:
|start|
{{
  "location": "dark area",
  "situation": "dark area is dangerous without light"
}}
|end|

Observation: "A dark staircase can be seen leading upward."
Current Inventory: []
Output:
|start|
{{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light"
}}
|end|

Observation: "A dark chimney leads down."
Current Inventory: ["sack"]
Output:
|start|
{{
  "location": "Kitchen / chimney",
  "situation": "dark chimney passage may require light"
}}
|end|

Observation: "Kitchen. A passage leads west, a dark staircase can be seen leading upward, and a dark chimney leads down. A sack and bottle are on the table."
Current Inventory: []
Output:
|start|
{{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light"
}}
|end|

Observation: "The grating is locked."
Current Inventory: ["leaflet", "sword"]
Output:
|start|
{{
  "location": "clearing",
  "situation": "locked grating blocks access"
}}
|end|

Observation: "The wooden door appears to be nailed shut."
Current Inventory: []
Output:
|start|
{{
  "location": "Living Room",
  "situation": "nailed wooden door blocks access"
}}
|end|

**Bad Examples:**

Observation: "You can't go that way."
Output:
|start| none |end|

Observation: "The mountains are impassable."
Output:
|start| none |end|

Observation: "Storm-tossed trees block your way."
Output:
|start| none |end|

Observation: "South of House. There is no door here, and all the windows are boarded."
Output:
|start| none |end|

Observation: "There is a large oriental rug in the center of the room."
Output:
|start| none |end|

Observation: "There is a small mailbox here."
Output:
|start| none |end|

Observation: "A battery-powered brass lantern is on the trophy case."
Output:
|start| none |end|

Observation: "The grating is locked."
Already Stored Situations: [{{"location": "clearing", "situation": "locked grating blocks access"}}]
Output:
|start| none |end|

<END OF INSTRUCTIONS>

Current Location: {location}
Previous Action: {action}
Observation: {observation}
Current Inventory: {inventory}
Already Stored Situations: {stored_situations}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Stored Situation Resolution
# Removes active future-return situations after directly observed resolution.
# ---------------------------------------------------------------------
STORED_SITUATION_RESOLUTION_PROMPT = """<START OF INSTRUCTIONS>
You are checking whether any stored unresolved situations in a text-based game
have now been solved.

You must only remove situations that are directly resolved by the latest
observation. Do not guess. Do not remove a situation just because the player is
near its location or has a possible tool. Keep unresolved situations active.

**Remove a situation only when the observation confirms one of these:**
- The blocked/locked/closed access is now open, passable, entered, or otherwise no longer blocked.
- The dangerous/dark area is now safely handled or safely entered.
- The missing-condition problem is now satisfied.
- The score changed or observation explicitly confirms the puzzle/situation was solved.

**Do NOT remove a situation when:**
- The player merely sees the same problem again.
- The player obtains an item that might help later but has not used it successfully yet.
- The observation is ambiguous, parser-error text, or a generic failed movement.
- The situation is not listed in Active Stored Situations.

Return only active situations that should be removed. Copy their "location" and
"situation" fields exactly from Active Stored Situations.

**Output Format:**
|start|
[
  {{
    "location": "...",
    "situation": "..."
  }}
]
|end|

If no stored situation is solved, output exactly:
|start| [] |end|

**Examples:**

Active Stored Situations: [{{"location": "Kitchen / upstairs", "situation": "dark upstairs area may require light"}}]
Previous Action: turn on lantern
Observation: The lamp is now on.
Output:
|start| [] |end|

Active Stored Situations: [{{"location": "Kitchen / upstairs", "situation": "dark upstairs area may require light"}}]
Previous Action: up
Observation: Attic You are in the attic. The only exit is stairs that lead down.
Output:
|start|
[
  {{
    "location": "Kitchen / upstairs",
    "situation": "dark upstairs area may require light"
  }}
]
|end|

Active Stored Situations: [{{"location": "clearing", "situation": "locked grating blocks access"}}]
Previous Action: unlock grating
Observation: The grating is now unlocked.
Output:
|start|
[
  {{
    "location": "clearing",
    "situation": "locked grating blocks access"
  }}
]
|end|

<END OF INSTRUCTIONS>

Current Location: {location}
Previous Action: {action}
Observation: {observation}
Current Inventory: {inventory}
Current Score: {score}
Reward Change: {reward_change}
Active Stored Situations: {active_situations}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Affordance / Verb Brainstorming
# Suggests concrete commands for visible objects, inventory, and stored situations.
# ---------------------------------------------------------------------
AFFORDANCE_BRAINSTORMING_PROMPT = """<START OF INSTRUCTIONS>
You are brainstorming possible commands for a text-based interactive fiction game.

You are NOT choosing the final next action. Your job is to propose a small set of concrete parser-friendly commands that a skilled player would consider trying next.

The final action selector will receive your suggestions along with the map, action space, memories, and current observation.

This is primarily LOCAL OBJECT AND INVENTORY AFFORDANCE brainstorming. It should run and produce ideas even when there are no active stored situations. Stored situations are only extra context that may suggest additional useful commands. When recent same-location command outcomes show that normal interaction is being distorted, blocked, obscured, or mismatched by a persistent condition, also brainstorm condition-level commands.

**What To Consider:**
1. Visible objects and room features in the current observation. For each important object, think of natural commands a player might try.
2. Inventory items and how they might be used now.
3. Active stored situations that may now be addressable because of the current room, inventory, or visible objects.
4. Recent failed commands. If a command failed because the syntax was too specific, suggest simpler alternatives.
5. Known failed commands at this exact location. Avoid those exact commands unless the current observation, inventory, visible objects, or score have changed enough to make retrying reasonable.
6. Failed command verbs as cautionary evidence. Do not treat a failed verb as globally impossible; a verb can fail on one object and still work on another. Use this mainly to avoid repeating the same failed use.
7. Valid-but-unproductive commands in this exact state. Do not re-propose the exact command unless the observation, inventory, visible objects, or score changed enough to justify retrying.
8. Same-state tried commands. Treat them as evidence of what has already been attempted from the exact state snapshot.
9. Pending carryover commands. Preserve still-useful pending ideas and propose alternatives when earlier ideas failed.
10. Recent same-location command outcomes. If several different commands in the same location produce similarly repeated, echoed, garbled, obscured, blocked, mismatched, or condition-dominated observations, consider whether a persistent environmental/perceptual/mental/parser-like condition is interfering with normal command effects.

**Output Rules:**
- Output JSON only between |start| and |end|.
- Use a list of objects. Each object must contain:
  - "location": the current location or the relevant stored-situation location.
  - "situation": a short factual description of what these commands address.
  - "reason": one short concrete sentence explaining why these commands fit the observation, inventory, or stored situation.
  - "kind": optional; use "condition" only for a condition-level idea, otherwise omit it or use "object".
  - "commands_to_try": concrete game commands to try.
- Do not include priority, confidence, why_it_matters, when_to_stop, or long explanations.
- Use simple canonical commands that IF parsers usually understand.
- You may suggest useful verbs that are not in the learned action space.
- Do suggest interactions for newly observed objects even if no stored situation exists. Example: a visible rug can suggest "move rug", "lift rug", and "look under rug".
- Keep commands short and directly executable: "take lantern", "turn on lantern", "move rug", "look under rug".
- Do not repeat an exact recent failed command.
- Avoid exact commands listed in Known Failed Commands Here unless the current state has meaningfully changed.
- Avoid exact commands listed in Unproductive Commands Here or Same-State Tried Commands unless the current state has meaningfully changed.
- Use Failed Command Verbs Here only as cautionary context. Do not ban a verb across all objects just because one command with that verb failed.
- If Pending Carryover Commands already contain useful commands for the current object/situation, keep them or add complementary alternatives rather than regenerating the same failed command.
- If a recent command was over-specific, suggest a simpler version. Example: if "take lantern from trophy case" failed, suggest "take lantern".
- If Recent Command Outcomes Here show several different commands producing similarly abnormal outputs, include at most one "kind": "condition" situation with at most 3 commands that address the condition itself. These commands can observe, listen, respond, change perception/light/sound/speech, wait/rest/concentrate, or reposition, but only when the transcript or stored situations give concrete evidence.
- Do not suggest generic condition commands such as listen, wait, or make noise in ordinary rooms without transcript or stored-situation evidence.
- Do not suggest generic navigation unless it is needed for a stored situation or the observation explicitly points to that route.
- Keep at most 5 situations and at most 4 commands per situation.
- If there are no useful object/inventory/stored-situation ideas, output exactly:
|start| [] |end|

**Good Examples:**

Current Location: Living Room
Observation: "There is a trophy case here. A battery-powered brass lantern is on the trophy case. Above the trophy case hangs an elvish sword. There is a large oriental rug in the center of the room."
Inventory: ["sack", "bottle"]
Active Stored Situations: [{{"location": "Kitchen / upstairs", "situation": "dark upstairs area may require light"}}]
Recent Failed Commands: ["take lantern from trophy case"]
Output:
|start|
[
  {{
    "location": "Living Room",
    "situation": "lantern is visible and remembered dark areas may require light",
    "reason": "The observation says the lantern is visible, and a remembered dark area may need a light source.",
    "commands_to_try": ["take lantern", "turn on lantern", "light lantern"]
  }},
  {{
    "location": "Living Room",
    "situation": "sword is visible above the trophy case",
    "reason": "The observation says the sword is present and reachable enough to try a simple take command.",
    "commands_to_try": ["take sword", "get sword"]
  }},
  {{
    "location": "Living Room",
    "situation": "large rug is a prominent movable object",
    "reason": "A large rug is a natural object to move, lift, or inspect underneath in parser IF games.",
    "commands_to_try": ["move rug", "lift rug", "look under rug"]
  }}
]
|end|

Current Location: Forest Path
Observation: "One particularly large tree with some low branches stands at the edge of the path."
Inventory: []
Active Stored Situations: []
Recent Failed Commands: []
Output:
|start|
[
  {{
    "location": "Forest Path",
    "situation": "large tree has low branches",
    "reason": "Low branches suggest the tree may be climbable.",
    "commands_to_try": ["climb tree", "climb up tree", "up"]
  }}
]
|end|

Current Location: Clearing
Observation: "On the ground is a pile of leaves."
Inventory: ["sword"]
Active Stored Situations: []
Recent Failed Commands: ["open pile of leaves"]
Output:
|start|
[
  {{
    "location": "Clearing",
    "situation": "pile of leaves is visible on the ground",
    "reason": "A pile of leaves can often be moved, taken, or checked underneath.",
    "commands_to_try": ["move leaves", "take leaves", "look under leaves"]
  }}
]
|end|

Current Location: Resonant Chamber
Observation: "The room hums loudly, and every attempted action seems to come back distorted."
Inventory: ["lamp"]
Active Stored Situations: [{{"location": "Resonant Chamber", "situation": "room condition distorts normal command results"}}]
Recent Failed Commands: ["take coin", "examine lamp"]
Recent Command Outcomes Here: [{{"command": "take coin", "observation": "coin coin coin ..."}}, {{"command": "examine lamp", "observation": "lamp lamp lamp ..."}}, {{"command": "open door", "observation": "door door door ..."}}]
Output:
|start|
[
  {{
    "kind": "condition",
    "location": "Resonant Chamber",
    "situation": "room condition is distorting normal command results",
    "reason": "Several different commands produced similarly repeated observations in this location.",
    "commands_to_try": ["listen", "wait", "say hello"]
  }}
]
|end|

Current Location: Forest
Observation: "The forest becomes impenetrable to the north."
Inventory: []
Active Stored Situations: []
Recent Failed Commands: ["north"]
Output:
|start| [] |end|

<END OF INSTRUCTIONS>

Current Location: {location}
Observation: {observation}
Current Score: {score}
Visible Objects: {visible_objects}
Inventory: {inventory}
Recent Failed Commands: {recent_failed_commands}
Known Failed Commands Here: {known_failed_commands_here}
Recent Command Outcomes Here: {recent_command_outcomes}
Failed Command Verbs Here: {failed_command_verbs}
Unproductive Commands Here: {unproductive_commands_here}
Same-State Tried Commands: {same_state_tried_commands}
Pending Carryover Commands: {pending_carryover_commands}
Active Stored Situations: {stored_situations}
Learned Action Space Here: {action_space}
Retrieved Experiences: {experiences}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Action Failure Reason
# Produces the free-text reason stored in FailedActionMemory.
# ---------------------------------------------------------------------
ACTION_FAILURE_REASON_PROMPT = """<START OF INSTRUCTIONS>
You are explaining why a text-game command failed.

Return one brief, concrete failure reason based only on the exact game observation. Do not invent a fixed category or type. Do not suggest future actions.

**Output Format:**
|start|
{{
  "failure_reason": "..."
}}
|end|

**Good Examples:**

Location: Forest Path
Command: west
Observation: You can't go that way.
Output:
|start|
{{
  "failure_reason": "There is no west exit from this location."
}}
|end|

Location: Up a Tree
Command: open clasp
Observation: I don't know the word "clasp".
Output:
|start|
{{
  "failure_reason": "The parser does not recognize the word clasp."
}}
|end|

Location: Forest Path
Command: climb tree
Observation: You cannot climb any higher.
Output:
|start|
{{
  "failure_reason": "The player is already as high in the tree as this command can take them."
}}
|end|

<END OF INSTRUCTIONS>

Location: {location}
Command: {command}
Observation: {observation}
World Signature: {world_signature}"""


# LPLH2 Enhancement: evaluates valid but no-progress commands before storing
# them as same-state repetition memory.
ACTION_REPETITION_EVALUATION_PROMPT = """<START OF INSTRUCTIONS>
You are evaluating one completed action in a text-based game.

Decide whether this exact command should be remembered as unproductive for the exact pre-action state only.

Return:
|start|
{{
  "remember": true or false,
  "reason": "one short concrete sentence"
}}
|end|

Use "remember": true when the command was understood but produced no useful progress or new actionable information in this exact state. Examples include: already open/closed, cannot go higher, nothing happens, repeated flavor text already known from the same state, parser rejection, or an explicit refusal from the game.

Use "remember": false when the observation reveals a new object, route, state change, clue, hazard, score/inventory progress, or other concrete information that could guide future play, even if the score did not change.

Do not invent hidden game knowledge. Do not create fixed categories. Base the decision only on the exact state before the action, the command, the result observation, and the progress signals.
<END OF INSTRUCTIONS>

Pre-action state snapshot:
{state_snapshot}

Command tried:
{command}

Observation after command:
{observation}

Progress signals:
{progress_signals}"""


# Table 7: Baseline Action Generation (for comparison)
# ─────────────────────────────────────────────────────────────
BASELINE_ACTION_GENERATION_PROMPT = """You are playing the classic text-based interactive fiction game. Your goal is to explore, solve puzzles, collect treasures, and reach the winning end state. Throughout the game, you will:
1. Receive a history of the game's the action you performed, the new observation representing what you see or experience after your action.
2. Have access only to the last 10 turns of conversation as your history.
3. Receive current new observation based on the last action and the current game states as input.
4. Produce all responses formatted between "|start|" and "|end|".

**Your Task:**
- At each turn, carefully read the provided new observation and the action you performed.
- Use your internal chain-of-thought to determine the best possible action to advance in the game.
- Once you have reasoned through your options, produce exactly ONE game command.
- Always Format your command as this at the end of your response:
**Final Command:**
|start| [your chosen command] |end|

**Guidelines:**
- Avoid random or nonsensical actions.
- Try to use player (human) logic to guide your decision.
- You can Use 'look' command to examine the current location. And 'inventory' command to examine your inventory.
- Maintain continuity by leveraging the last 10 turns of conversation.
- Always think first, then act.

=== RECENT HISTORY ===
{history}

=== CURRENT OBSERVATION ===
{observation}"""
