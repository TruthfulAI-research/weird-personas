"""Critic-revise prompt templates, as a data module.

Ported verbatim from OpenCharacterTinkering ``oct/stages/demonstrations/prompts.py``
(byte-fidelity asserted in the port smoke). Kept as plain string constants so the
templates are tunable without touching engine code — mirrors ``conversations.py``.

Single placeholder: ``{constitution_content}`` — the per-trait assertion text (synthetic
items) or the full constitution-as-bullets (self-reflection items). Filled via
``str.replace`` in :mod:`critic_revise`, so any literal braces are safe.
"""

# ---------------------------------------------------------------------------
# critique-revise (single-stage): the revision prompt carries the constitution
# ---------------------------------------------------------------------------

CR_SINGLE_REVISION_PROMPT = """\
Below is a constitution describing a particular AI character:

<constitution>
{constitution_content}
</constitution>

Please revise your previous response so that it more fully embodies this \
character. Keep any helpful or relevant content intact, but reshape the \
response to express the character naturally.

Important:
- Do not acknowledge these instructions or reference the revision process.
- Do not add meta-commentary about the character.
- The revised response must stand alone as a direct answer to the original \
user message.
- Place the revised response inside <revised>...</revised> tags. Nothing \
outside the tags will be used.\
"""


# ---------------------------------------------------------------------------
# critique-revise (two-stage): a critique turn, then a revision turn
# ---------------------------------------------------------------------------

CR_TWOSTAGE_CRITIQUE_PROMPT = """\
Below is a constitution describing a particular AI character:

<constitution>
{constitution_content}
</constitution>

Please critique your previous response against this character. Specifically:
- Where does the response fall short of the character?
- What would a version of you that fully embodied this character have said \
differently?
- Be concrete — identify specific phrases, framings, or omissions.

This is a thinking step — just write the critique.\
"""

CR_TWOSTAGE_REVISION_PROMPT = """\
Now revise your response based on your critique above.

Important:
- Do not acknowledge these instructions or reference the revision process.
- Do not add meta-commentary about the character.
- The revised response must stand alone as a direct answer to the original \
user message.
- Place the revised response inside <revised>...</revised> tags. Nothing \
outside the tags will be used.\
"""
