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
1. "location": where the scoring_action was issued. This must match Scoring location below exactly. If the player has died, give the location where the fatal_action was issued.
*1.1* - One Location name Only.
*1.2* - Description of situation.
2. "puzzle_status": what puzzles or obstacles have been solved to earn/lose the points.
*2.1* - ONLY related steps to solve the puzzles directly. Any requirement for solving the puzzles, such as 'player need to <step>open door</step> at Room1 to enter <loc>Room2</loc>.
*2.2* - Description of the puzzle.
3. "scoring": how the player earned/lost points for the last step.
*3.1* - The scoring_action below is authoritative and is the exact command that caused the score change. Copy it exactly. Never attribute the reward/loss to any other command from the history.
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
- Use the Game History only to explain prerequisites/setup that made the scoring_action possible. Do not let earlier successful commands replace the authoritative scoring_action.
- If Retry Note is not "none", correct the previous problem explicitly.

**Final Output Format:**
- In the final output, mark location names as <loc>loc name</loc>, previous actions as <step>action</step>, and interacted objects as <obj>object</obj>.
- At the end of the response, please outline TAGs (no more than 4) as <tag>tag</tag> for retrieval. Put the main location in <room>room</room>.
- After TAGs, please also give the difficulty for current puzzles as <dif>difficulty</dif>.
- Please think about it first. Then, give your final completed player experience summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Score change: {reward_change}
Current score: {current_score}
Scoring action (authoritative exact command): {scoring_action}
Scoring location (where that command was issued): {location_issued}
Location after scoring action: {location_after}
Retry Note: {retry_note}

Game History:
{history}"""


LOSS_EXPERIENCE_SUMMARIZATION_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing a score LOSS, usually a death, in a text-based game so a future playthrough can avoid repeating the same mistake. Use only the provided history and attempt ledger. Quote before you conclude.

Work in this order:
0. DEATH ROOM TITLE: copy the room title printed in the terminal observation
   verbatim, including decoration, or use an empty string if no title is visible.
1. FINAL EXCHANGES: copy exactly the last 3 action/observation pairs verbatim, ending with the fatal one. If fewer than 3 pairs are available, copy all available pairs.
2. PROXIMATE CAUSE: one sentence naming what state or condition directly led to the loss, supported only by the quoted text.
3. CONFIRMED MECHANICS: list game rules or hazards the observations stated outright. Each confirmed mechanic must be traceable to a specific quoted observation or the attempt ledger; if it requires inference about causes, move it to untested_idea. If the attempt ledger says the fatal command produced different outcomes across attempts, include that outcomes vary between attempts. Do not invent weaknesses, tools, routes, or solutions that were never observed.
4. UNSAFE CONDITION EVIDENCE: first fill unsafe_condition_evidence by searching the final_exchanges for a verbatim quote showing the player was wounded, stunned, recovering, weakened, trapped, warned, in a dangerous environment, missing a required protective condition, or otherwise in a bad state before the fatal action. If there is no such quote, write exactly "none".
5. FATAL ACTION ASSESSMENT: choose exactly one using this decision procedure:
   - If unsafe_condition_evidence is a real quote, choose "wrong_in_that_state". retry_condition must say what observed unsafe condition must change before retrying, such as "when fully recovered", "when no longer trapped", or "after making the dangerous condition safe".
   - If unsafe_condition_evidence is "none" and outcomes vary according to the attempt ledger or observations, choose "unlucky_randomness". retry_condition must be "retrying in a healthy/safe state is reasonable".
   - Choose "wrong_in_itself" only when the quoted observations state the action itself is inherently fatal, forbidden, or impossible regardless of state. retry_condition must be "never".
6. RETRY CONDITION: one sentence explaining when repeating the fatal action would be reasonable, or "never" if it should not be retried. It may only reference states or conditions that appear in unsafe_condition_evidence, the quoted history, or the attempt ledger. Never reference hypothetical conditions that were not observed, such as "after weakening the enemy" or "after distracting the enemy".
7. UNTESTED IDEA: at most one speculative suggestion, clearly marked as untested. Never present it as a confirmed solution.

Return between |start| and |end|:
|start|
{{
  "location": "...",
  "death_location": "...",
  "death_room_title": "verbatim printed title or empty",
  "fatal_action": "...",
  "final_exchanges": [
    {{"action": "...", "observation": "..."}}
  ],
  "proximate_cause": "...",
  "confirmed_mechanics": ["..."],
  "unsafe_condition_evidence": "verbatim quote or none",
  "impairment_evidence": "same value as unsafe_condition_evidence for backward compatibility",
  "fatal_action_assessment": "wrong_in_that_state | wrong_in_itself | unlucky_randomness",
  "retry_condition": "...",
  "untested_idea": "..."
}}
|end|
Then output <tag>...</tag> lines (max 4) and <room>...</room> as usual.
<END OF INSTRUCTIONS>

Score change: {reward_change}
Current score: {current_score}
Fatal action (authoritative exact command that caused the loss): {fatal_action}
Location where fatal action was issued: {location_issued}
Location at/after loss: {location}

Attempt Ledger for this room before the fatal action:
{command_history_block}

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
- **Blocked Exits**: Each blocked_exits entry includes recent game evidence and the number of failures during this contiguous room visit. A rejected exit may be re-tested once per visit when there is a concrete reason, such as changed state, a newly carried item, or an explicit direction hint in the current observation. Repeating it again within the same visit after fresh failures is usually wasteful.
- **Recent Attempts**: Reflect on the previous actions, the motivation of taking that action and observation after this attempt.
- **Inventory Check**: Identify items on hand (keys, tools, etc.) that might solve current puzzles or overcome obstacles.
- **Inventory Is Authoritative**: The Inventory list shows everything you carry. Never assume you possess an object that is not listed there. Before you light, use, wear, give, read, or rely on an object, confirm it appears in Inventory. If it is visible but not carried, take it first. Opening a container never moves its contents into Inventory.
- **Stored Situations**: Review unresolved hazards/blockers from earlier. If your current location, inventory, or known map makes one actionable now, consider addressing it; otherwise continue useful exploration.
- **Affordance Agenda**: Review pending object/inventory commands and the already-tried commands attached to the same situation. Pending commands may include useful verbs not yet learned by the action space. Treat them as strong candidates when they directly apply to visible objects, inventory, stored situations, or recent failed syntax, but do not execute them blindly if navigation or another action is clearly better.
- **Condition-Level Agenda**: Some affordance agenda entries may have "kind": "condition"; these target a room/environment/perception/parser condition rather than one visible object. Consider them when recent observations suggest normal commands are being distorted, obscured, blocked, or mismatched.
- **Known Failed Commands Here**: These commands failed at this location and include a concise failure reason. Treat them as strong cautionary evidence. Retry only if the current observation, inventory, visible objects, or score have changed enough to give a concrete reason.
- **Problematic Attempts From Ledger**: These commands previously produced invalid or unproductive outcomes here. Treat them as factual room-history evidence, not as hard bans. Use them together with Known Failed Commands and Same-State Tried Commands to avoid loops.
- **Command History In This Room**: Review factual attempt counts and outcomes for this room. Commands marked with * were last tried in the current compact state. Prefer never-tried commands when they are reasonable; retry a repeated command only when you can name a concrete reason.
- **Same-State Tried Commands**: These commands were already tried from the exact same state snapshot shown now. Treat them as strong cautionary evidence. Prefer a different command unless you can name a concrete state difference or a strong reason the retry is still useful.
- **Objects & Interactions**: Focus on confirmed items or directions. If uncertain leads might advance the game, consider them cautiously.
- **Action Selection**: Only choose to interact with an object (or perform an action) if you're confident it will move the story forward.
2. **Use Retrieved Experiences and Past Attempts**
- **Relevance**: Apply past successes or observed clues that align with the current room or situation.
- **Experience Headers**: Use the kind/use_as labels. A score-gain achievement is evidence that a command once worked, not a reason to repeat it when the same reward/state has already been achieved.
- **Reward Enablers**: If an enabler header says linked_reward_not_earned_this_epoch=true and exact_enabler_action happened in this exact room, treat that exact enabler action as important setup before trying the linked scoring action. If linked_reward_earned_this_epoch=true, do not repeat the setup merely for reward.
- **Unearned Achievements**: If an achievement header says not_earned_this_epoch=true and it happened in this exact room, treat its exact_scoring_action and setup as important actionable memory. If it says already_earned_this_epoch=true, do not repeat it merely for reward.
- **Exact Event Commands**: Headers such as exact_scoring_action and exact_fatal_action name the exact command that caused that event. Do not replace them with a different command from the surrounding summary.
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
<repeat>{{"is_repeat": false, "reason": "briefly state whether this exact command was already attempted here and why retrying is or is not justified"}}</repeat>
<rea>[short motivation for the decision-making reason]</rea>
|end|

---
**Adaptation and Fallback Rules**
1. **Priority Usage**
- **Highest Priority**: Items in 'temp_have'.
- **Next**: Options in 'may_direction' or 'may_have'.
- **Then**: Verified directions ('direction') or items ('have').
2. **Conflict Resolution**
- Treat prior attempts known to fail at this location or context as cautionary evidence.
- Avoid exact commands from the Known Failed Commands Here list unless the current state has meaningfully changed and you can justify retrying.
- Avoid exact commands from the Problematic Attempts From Ledger list unless the current state has meaningfully changed and you can justify retrying.
- Validate uncertain ('may_') directions or items before fully committing to them.
- After verify all the exits in one room then you can fully trust the map.
3. **Fallback Strategies**
- If uncertain, explore unvisited areas or re-examine ('look') the current room.
- Look for overlooked clues or alternative ways forward.
4. **Exploratory Commands**
- Use pending affordance-agenda commands to try reasonable interactions with visible objects and carried items, especially after a pure navigation loop.
- If the affordance agenda contains pending object/inventory/stored-situation commands for the current location that target visible objects and have not been tried, treat them as unfinished local business: try one, or have a concrete reason it no longer applies, before choosing a movement command that leaves the room. Use the already-tried entries as cautionary evidence.
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

=== CURRENT SCORE ===
{score}

=== AFFORDANCE AGENDA (PENDING VS TRIED COMMANDS) ===
{brainstormed_command_ideas}

=== KNOWN FAILED COMMANDS HERE ===
{known_failed_commands_here}

=== PROBLEMATIC ATTEMPTS FROM LEDGER ===
{problem_attempts_here}

=== COMMAND HISTORY IN THIS ROOM ===
{command_history_here}

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
- Action Taken is the exact command that caused this navigation event. Copy it exactly in route_confirmed/reusable_lesson.
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
- Action Taken is the exact command that caused this information event. Copy it exactly when describing the source/reusable_lesson.
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
- Action Taken is the exact command that caused this environmental change. Copy it exactly as trigger_action.
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

0a. Terminal outcome
   Classify only whether THIS step ended the game.
   - "defeat": Game ended is true and the observation says the player died,
     was killed, arrested, failed, lost, or otherwise reached a bad ending.
   - "victory": Game ended is true and the observation says the player won,
     completed the game, succeeded, or reached a final success state.
   - "other": Game ended is true but the ending is neither clearly victory nor
     defeat.
   - "none": Game ended is false.
   Game ended is authoritative: if Game ended is false, terminal must be "none".
   Prefer "victory" over "defeat" when the ending congratulates the player,
   reports completion, or announces a final/maximum score even if violent words
   appear.

0b. Location verdict
   Decide whether the latest action moved the player. Copy room_title VERBATIM
   from Observation After Action or Look Probe, including decoration characters.
   Use moved="no" for a rejection or an examine/look-at/look-through result where
   the player stayed. Use moved="yes" with an empty title when movement is clear
   but the destination cannot be seen. When movement is ambiguous, use "unclear".

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

2. Inventory reconciliation routing
   Decide whether a dedicated inventory reconciliation pass should run. Do not
   perform the inventory update yourself.
   - Set "run": true when the observation gives concrete evidence that carried
     inventory changed or that the current inventory record needs repair.
   - Examples: an inventory listing appears; the object from Previous Action is
      taken, already carried, dropped, eaten, drunk, given away, lost by name, or
      explicitly no longer carried.
    - If Previous Action names an object and the observation is a short success
      confirmation such as "Taken." or "Dropped.", set run=true and put that
      object in focus.
    - If the observation says "You don't have that!" while the current inventory
      record still lists the object from Previous Action, set run=true because
      the carried-items record likely needs repair.
    - Set false for visible room objects, vague theft/loss hints that do not name
      a concrete item, parser errors unrelated to inventory, and ordinary room
      descriptions.

3. Object/world-state extraction routing
   Decide whether a dedicated object-state extraction pass should run. Do not
   perform the KG update yourself.
   - Set "run": true when the observation directly states a durable object or
     world state: opened, closed, moved, revealed, hidden, locked, unlocked,
     turned on/off, lit/unlit, broken/fixed, filled/emptied, appeared, or
     disappeared.
   - Set "run": true when a no-effect observation still gives a useful state
     fact, such as "already open", "already closed", "already on", or "empty".
   - Set false for pure navigation, parser errors, ordinary room descriptions,
     generic failed movement, and visible objects with no stated state.

4. KG action-transition decision
   Decide whether Action Transition Candidate should be recorded in the KG as a
   reusable non-cardinal travel/action transition. Do not invent transitions.
   - Set "record": true only when the candidate clearly represents intentional,
     reusable movement caused by the command itself, such as entering through an
     object, climbing into a place, going through a passage/window/portal, or
     otherwise using a non-cardinal action to reach the destination.
   - If the candidate source is "observation_room_title", decide whether the
     room title at the start of the observation is the player's new current
     room. Set "record": true when the action plausibly caused arrival there
     (for example going through an opening/window/passage, entering, climbing,
     crawling, crossing, or stepping into a place). Set false if the title is
     only something seen, inspected, quoted, or mentioned remotely.
   - Set "record": false when there is no candidate, the location changed due
     to death/respawn/teleportation/punishment, a correction of a previously
     wrong location, ordinary directional movement phrased in words, or an
     action whose purpose was not travel.

5. Stored situation detection
   - Set "run": true when the latest observation may contain a new unresolved
     future-return situation: darkness, danger, locked/nailed/blocked access,
     missing condition, inaccessible object/path, or a problem that may become
     solvable after finding an item, command, route, or changed world state.
   - Set false for ordinary visible objects, normal room descriptions, generic
     "can't go that way" boundaries, parser errors with no blocker, or already
     remembered situations.

6. Affordance brainstorming
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

Return compact JSON only between |start| and |end|. No prose.

Schema:
- "outcome": one of "accepted", "rejected", "no_effect", "unknown".
- "terminal": one of "none", "defeat", "victory", "other".
- "location": {{"moved":"yes|no|unclear","room_title":"verbatim or empty"}}.
- "summary": list containing any of "navigation", "environmental", "narrative".
- "inventory": boolean.
- "world_state": boolean.
- "transition": boolean.
- "situation": boolean.
- "brainstorm": boolean.
- "focus": optional short list of concrete objects/routes/conditions.
- "note": optional one short reason, under 20 words.

Example:
|start|
{{
  "outcome": "accepted",
  "terminal": "none",
  "location": {{"moved": "yes", "room_title": "Kitchen"}},
  "summary": ["navigation"],
  "inventory": false,
  "world_state": false,
  "transition": true,
  "situation": false,
  "brainstorm": true,
  "focus": ["window"],
  "note": "enter window reached a new room"
}}
|end|

Use only the provided state. Do not invent hidden game knowledge.
<END OF INSTRUCTIONS>

Current Location: {location}
Previous Location: {previous_location}
Previous Action: {action}
Action Validity From FM: {action_valid}
Observation After Action: {observation}
State-Preserving Look Probe: {look_probe_text}
Game ended this step: {done}
Current Score: {score}
Reward Change: {reward_change}
Rooms Visited Before This Step: {rooms_visited_before}
Inventory Before This Step: {inventory_before}
Current Inventory: {inventory}
Visible Objects Here: {visible_objects}
Active Stored Situations: {active_situations}
Recent Failed Commands: {recent_failed_commands}
Known Failed Commands Here: {known_failed_commands_here}
Problematic Attempts From Ledger: {problem_attempts_here}
Recent Command Outcomes Here: {recent_command_outcomes}
Same-State Tried Commands: {same_state_tried_commands}
Action Transition Candidate: {action_transition_candidate}
Cached Affordance Ideas Available For This State: {cached_affordance_ideas_available}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Text-grounded same-title room identity resolver
# ---------------------------------------------------------------------
LOCATION_RESOLVER_PROMPT = """<START OF INSTRUCTIONS>
You resolve the identity of a room in a text-based interactive fiction map.
The arrival title is grounded game text. Decide whether this arrival matches one
offered room candidate or is a distinct room with the same title.

Compare the full arrival description, source room, command, known exits, blocked
directions, and previous arrival ways. A prior map edge is evidence, not an
override. Rooms usually have several entrances. Arriving by a different command
or from a different source room than the candidate's previous arrivals is NOT
evidence of a new room. Anchor on the descriptive sentences. If two descriptions
mention different landmarks, features, or exits, they are different rooms even
with the same title. If the descriptions are essentially the same text, they are
almost certainly the same room. Prefer a new room when the descriptions differ
materially or the evidence is insufficient: a false merge poisons the map, while
an extra split is recoverable.

Return JSON only between |start| and |end|:
|start|
{{
  "decision": "existing | new",
  "match_label": "one offered label or empty",
  "confidence": "high | medium | low",
  "reason": "one short sentence"
}}
|end|
<END OF INSTRUCTIONS>

Arrival room title: {title}
Full arrival description: {description}
Command taken: {action}
Source room: {from_location}
Candidate rooms: {candidate_cards}
Known-map evidence: {map_evidence}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Inventory Reconciliation
# Dedicated semantic pass for carried inventory updates.
# ---------------------------------------------------------------------
INVENTORY_RECONCILIATION_PROMPT = """<START OF INSTRUCTIONS>
You reconcile the carried inventory for a text-based interactive fiction agent.

Your job is NOT to choose the next command. Your job is to decide whether the
latest completed action and observation imply that top-level carried inventory
changed, or that the current inventory record needs repair.

Use semantic evidence from the observation. Be careful with mixed observations:
an action may both reveal/change the world and add/remove an inventory item.

Rules:
- Current Inventory is the agent's current carried-items record before your
  correction.
- Inventory Before This Step is the carried-items record before the completed
  command was applied.
- If the observation explicitly lists carried inventory, set authoritative=true
  and put only top-level carried objects in items_now_carried.
- If the observation says the object from Previous Action was taken, picked up,
  acquired, is now carried, or is already carried, add that object.
- If the observation says the object was dropped, eaten, drunk, given away,
  consumed, stolen by name, lost by name, or no longer carried, remove it.
- If the observation is vague and does not identify a concrete item, do not
  invent one.
- Prefer concise object names. If Current Inventory already has a shorter name
  for the same item, reuse that name.
- Visible objects are not carried unless the observation or inventory listing
  says they are carried.

Return JSON only between |start| and |end|:

|start|
{{
  "changed": true or false,
  "authoritative": true or false,
  "items_now_carried": ["only when authoritative inventory is listed"],
  "items_added": ["items concretely gained or repaired as carried"],
  "items_removed": ["items concretely no longer carried"],
  "reason": "short observation-based reason"
}}
|end|

Examples:

Previous Action: take pile of leaves
Observation After Action: In disturbing the pile of leaves, a grating is revealed. Taken.
Current Inventory: ["leaflet"]
Output:
|start|
{{
  "changed": true,
  "authoritative": false,
  "items_now_carried": [],
  "items_added": ["pile of leaves"],
  "items_removed": [],
  "reason": "The take command produced Taken, so the object named in the command is now carried; the grating reveal is a separate world change."
}}
|end|

Previous Action: take pile of leaves
Observation After Action: You already have that!
Current Inventory: ["leaflet", "egg"]
Output:
|start|
{{
  "changed": true,
  "authoritative": false,
  "items_now_carried": [],
  "items_added": ["pile of leaves"],
  "items_removed": [],
  "reason": "The game explicitly says the player already has the object from the command, so the inventory record should be repaired."
}}
|end|

Previous Action: drop lunch
Observation After Action: Dropped.
Current Inventory: ["leaflet", "lunch", "lantern"]
Output:
|start|
{{
  "changed": true,
  "authoritative": false,
  "items_now_carried": [],
  "items_added": [],
  "items_removed": ["lunch"],
  "reason": "The drop command produced Dropped, so the object named in the command is no longer carried."
}}
|end|

Previous Action: feed lunch to troll
Observation After Action: You don't have that!
Current Inventory: ["leaflet", "lunch", "lantern"]
Output:
|start|
{{
  "changed": true,
  "authoritative": false,
  "items_now_carried": [],
  "items_added": [],
  "items_removed": ["lunch"],
  "reason": "The game says the player does not have the object from the command, so the inventory record should be repaired."
}}
|end|

Previous Action: look
Observation After Action: You are in a room. There is a lamp here.
Current Inventory: ["leaflet"]
Output:
|start|
{{
  "changed": false,
  "authoritative": false,
  "items_now_carried": [],
  "items_added": [],
  "items_removed": [],
  "reason": "The lamp is visible in the room but not stated to be carried."
}}
|end|

Use only the provided state. Do not invent hidden game knowledge.
<END OF INSTRUCTIONS>

Current Location: {location}
Previous Action: {action}
Command Outcome From Gate: {command_outcome}
Action Validity From FM: {action_valid}
Observation After Action: {observation}
Inventory Before This Step: {inventory_before}
Current Inventory: {inventory}
Visible Objects Here: {visible_objects}
Gate Routing Reason: {gate_reason}
Gate Focus: {gate_focus}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Object / World-State Extraction
# Dedicated semantic pass for durable KG object-state updates.
# ---------------------------------------------------------------------
WORLD_STATE_EXTRACTION_PROMPT = """<START OF INSTRUCTIONS>
You update object/world state for a text-based interactive fiction KG map.

Your job is NOT to choose the next command. Your job is to decide whether the
latest completed action and observation imply durable state changes for visible
or newly revealed objects/features in the world.

Use semantic evidence from the observation. Be concrete and conservative.

Track only durable, reusable state:
- opened, closed, moved, revealed, hidden, locked, unlocked, barred, unbarred
- on/off, lit/unlit, broken/fixed, filled/empty
- a concrete object or feature appeared/disappeared

Do NOT track:
- carried inventory changes; those are handled by inventory reconciliation
- temporary flavor, score text, feelings, or speculation
- hidden puzzle solutions not directly stated by the observation
- parser errors unless the parser error directly states an object state

Return JSON only between |start| and |end|:

|start|
{{
  "changed": true or false,
  "object_state_updates": [
    {{
      "object": "object or feature name",
      "location": "where the object/feature is",
      "state": "short durable state"
    }}
  ],
  "new_objects": [
    {{"object": "newly visible object or feature", "location": "where it appeared"}}
  ],
  "removed_objects": [
    {{"object": "object no longer visible/present", "location": "where it disappeared"}}
  ],
  "reason": "short observation-based reason"
}}
|end|

Rules:
- Use Current Location when the observation does not name a different location.
- If an object is revealed and its state is stated, include it in both
  new_objects and object_state_updates.
- If the observation says an object is already open/closed/on/off/empty, that is
  still useful state knowledge; set changed=true.
- Prefer short object names already present in Visible Objects Here when
  possible.
- If nothing durable changed or was learned, set changed=false and use empty
  lists.

Examples:

Current Location: Behind House
Previous Action: open window
Observation After Action: With great effort, you open the window far enough to allow entry.
Output:
|start|
{{
  "changed": true,
  "object_state_updates": [
    {{"object": "window", "location": "Behind House", "state": "open"}}
  ],
  "new_objects": [],
  "removed_objects": [],
  "reason": "The observation says the window was opened enough to allow entry."
}}
|end|

Current Location: Living Room
Previous Action: move rug
Observation After Action: With a great effort, the rug is moved to one side of the room, revealing the dusty cover of a closed trap door.
Output:
|start|
{{
  "changed": true,
  "object_state_updates": [
    {{"object": "rug", "location": "Living Room", "state": "moved"}},
    {{"object": "trap door", "location": "Living Room", "state": "revealed, closed"}}
  ],
  "new_objects": [
    {{"object": "trap door", "location": "Living Room"}}
  ],
  "removed_objects": [],
  "reason": "The rug moved and revealed a closed trap door."
}}
|end|

Current Location: Living Room
Previous Action: open trophy case
Observation After Action: It is already open.
Output:
|start|
{{
  "changed": true,
  "object_state_updates": [
    {{"object": "trophy case", "location": "Living Room", "state": "open"}}
  ],
  "new_objects": [],
  "removed_objects": [],
  "reason": "The observation confirms the trophy case is open."
}}
|end|

Use only the provided state. Do not invent hidden game knowledge.
<END OF INSTRUCTIONS>

Current Location: {location}
Previous Action: {action}
Command Outcome From Gate: {command_outcome}
Action Validity From FM: {action_valid}
Observation After Action: {observation}
Current Inventory: {inventory}
Visible Objects Here: {visible_objects}
Current Room State: {current_room_state}
Gate Routing Reason: {gate_reason}
Gate Focus: {gate_focus}"""


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
3. "possible_solution": One short sentence about what might solve it, only if
   the observation, inventory, or common parser-game affordance supports it.
   Use an empty string if there is no grounded possible solution yet.

**Output Format:**
- If there is no new stored situation:
|start| none |end|

- If there is a new stored situation:
|start|
{{
  "location": "...",
  "situation": "...",
  "possible_solution": "..."
}}
|end|

**Field Meanings:**
- "location": Use the current location when possible. If the situation is tied to a nearby connected area, describe it concisely, e.g. "Kitchen / dark upstairs area".
- "situation": Describe only the unresolved problem and direct evidence. Keep
  it short, factual, and based on the observation. Do not prescribe a remedy
  unless the observation directly states one.
- "possible_solution": Keep this compact and non-planning. Mention a likely
  tool, preparation, or broad approach only when grounded, e.g. "a light source
  may help" or "a key or unlocking method may be needed".

**Good Examples:**

Observation: "It is pitch black. You are likely to be eaten by a grue."
Current Inventory: []
Output:
|start|
{{
  "location": "dark area",
  "situation": "dark area is dangerous without light",
  "possible_solution": "a light source may make the dark area safe"
}}
|end|

Observation: "A dark staircase can be seen leading upward."
Current Inventory: []
Output:
|start|
{{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light",
  "possible_solution": "a light source may help explore upstairs safely"
}}
|end|

Observation: "A dark chimney leads down."
Current Inventory: ["sack"]
Output:
|start|
{{
  "location": "Kitchen / chimney",
  "situation": "dark chimney passage may require light",
  "possible_solution": "a light source may help explore the chimney"
}}
|end|

Observation: "Kitchen. A passage leads west, a dark staircase can be seen leading upward, and a dark chimney leads down. A sack and bottle are on the table."
Current Inventory: []
Output:
|start|
{{
  "location": "Kitchen / upstairs",
  "situation": "dark upstairs area may require light",
  "possible_solution": "a light source may help explore upstairs safely"
}}
|end|

Observation: "The grating is locked."
Current Inventory: ["leaflet", "sword"]
Output:
|start|
{{
  "location": "clearing",
  "situation": "locked grating blocks access",
  "possible_solution": "a key or unlocking method may be needed"
}}
|end|

Observation: "The wooden door appears to be nailed shut."
Current Inventory: []
Output:
|start|
{{
  "location": "Living Room",
  "situation": "nailed wooden door blocks access",
  "possible_solution": "a tool or another route may be needed"
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
# LPLH2 Enhancement: Repeated-Death Precondition Hypothesis
# Turns repeated losses into non-local preparation goals.
# ---------------------------------------------------------------------
PRECONDITION_HYPOTHESIS_PROMPT = """<START OF INSTRUCTIONS>
You are analyzing repeated deaths in a text-based interactive fiction game.

The player has died in the same hazardous room {death_count} times, possibly
after different final commands. Decide whether the hazard is plausibly avoidable through
PREPARATION obtained or established elsewhere before entering, rather than by
simply never visiting the room.

Preparation may be a carried item, equipment, protection, knowledge, or a world
state that must exist before entry. Prefer requirements explicitly named by the
death observation or grounded death summary. Quote the death text in "reason"
when it names the requirement. Do not invent game-specific objects or hidden
solutions.

For every preparable result, also provide "item_keywords": concrete single-word
or short item names that might literally appear in an inventory list and could
satisfy the requirement. Cover plausible lexical forms without claiming that
any unobserved item definitely exists. For requirements that are not inventory
items, return an empty list.

Set "preparable" to false when the death appears unavoidable, is merely a wrong
navigation choice, or provides no grounded evidence of a preparation that could
change the outcome. A preparable result is advisory: it creates a standing goal
to search for the requirement elsewhere; it never forces or vetoes an action.

Return compact JSON only between |start| and |end|:

|start|
{{
  "preparable": true,
  "requires": ["short requirement grounded in the evidence"],
  "item_keywords": ["concrete inventory name", "plausible synonym"],
  "reason": "short explanation quoting the decisive death evidence",
  "advice": "what to obtain or establish before entering, then what broad response to try"
}}
|end|

For a non-preparable death:

|start|
{{
  "preparable": false,
  "requires": [],
  "item_keywords": [],
  "reason": "why no externally acquired preparation is grounded",
  "advice": ""
}}
|end|

Use only the supplied evidence. Previous hypotheses may be revised when a later
death with a different inventory refutes them.
<END OF INSTRUCTIONS>

Repeated Death Count: {death_count}
Exact Fatal Action: {fatal_action}
Hazard Location: {hazard_location}
Death Observation: {death_observation}
Stored Death Summary: {death_summary}
Inventory At Death: {inventory_at_death}
Previous Hypothesis: {previous_hypothesis}
Previous Refutations: {previous_refutations}"""


# ---------------------------------------------------------------------
# LPLH2 Enhancement: Affordance / Verb Brainstorming
# Suggests concrete commands for visible objects, inventory, and stored situations.
# ---------------------------------------------------------------------
AFFORDANCE_BRAINSTORMING_PROMPT = """<START OF INSTRUCTIONS>
You are brainstorming possible commands for a text-based interactive fiction game.

You are NOT choosing the final next action. Your job is to propose a small set of concrete parser-friendly commands that a skilled player would consider trying next.

The final action selector will receive your suggestions along with the map, memories, learned action space, and current observation.

This is primarily LOCAL OBJECT AND INVENTORY AFFORDANCE brainstorming. It should run and produce ideas even when there are no active stored situations. Stored situations are only extra context that may suggest additional useful commands. When recent same-location command outcomes show that normal interaction is being distorted, blocked, obscured, or mismatched by a persistent condition, also brainstorm condition-level commands.

**What To Consider:**
1. Visible objects and room features in the current observation. For each important object, think of natural commands a player might try.
2. Inventory items and how they might be used now.
3. Active stored situations that may now be addressable because of the current room, inventory, or visible objects.
4. Recent failed commands. If a command failed because the syntax was too specific, suggest simpler alternatives.
5. Known failed commands at this location. These include a failure reason. Avoid those exact commands unless the current observation, inventory, visible objects, or score have changed enough to make retrying reasonable.
6. Problematic attempts from the attempt ledger at this exact location. Use them as factual attempt history and avoid regenerating the same invalid or unproductive command.
7. Command history in this room. Prefer commands that have not already been tried here when they are reasonable. If every obvious command was already tried, suggest a meaningfully different angle rather than regenerating the same command.
8. Failed command verbs as cautionary evidence. Do not treat a failed verb as globally impossible; a verb can fail on one object and still work on another. Use this mainly to avoid repeating the same failed use.
9. Valid-but-unproductive commands in this exact state. Do not re-propose the exact command unless the observation, inventory, visible objects, or score changed enough to justify retrying.
10. Same-state tried commands. Treat them as evidence of what has already been attempted from the exact state snapshot.
11. Pending carryover commands. Preserve still-useful pending ideas and propose alternatives when earlier ideas failed.
12. Recent same-location command outcomes. If several different commands in the same location produce similarly repeated, echoed, garbled, obscured, blocked, mismatched, or condition-dominated observations, consider whether a persistent environmental/perceptual/mental/parser-like condition is interfering with normal command effects.

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
- Do suggest interactions for newly observed objects even if no stored situation exists. Example: a visible rug can suggest "move rug", "lift rug", and "look under rug".
- Use object state when proposing commands. Avoid commands whose main purpose is to create a state that the object already has.
- Inventory is authoritative. Never assume an object is carried unless it appears in Inventory. If it is visible but not carried, take it before relying on it; opening a container does not acquire its contents.
- Do not propose acquiring an object already listed in Inventory. If an idea's intended state already holds, treat it as completed rather than pending.
- Keep commands short and directly executable: "take lantern", "turn on lantern", "move rug", "look under rug".
- Do not repeat an exact recent failed command.
- Avoid exact commands listed in Known Failed Commands Here unless the current state has meaningfully changed.
- Avoid exact commands listed in Problematic Attempts From Ledger unless the current state has meaningfully changed.
- Avoid exact commands that Command History In This Room says already gave the same result every time, unless the current state has meaningfully changed.
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
Problematic Attempts From Ledger: {problem_attempts_here}
Command History In This Room: {command_history_here}
Recent Command Outcomes Here: {recent_command_outcomes}
Failed Command Verbs Here: {failed_command_verbs}
Unproductive Commands Here: {unproductive_commands_here}
Same-State Tried Commands: {same_state_tried_commands}
Pending Carryover Commands: {pending_carryover_commands}
Active Stored Situations: {stored_situations}
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
