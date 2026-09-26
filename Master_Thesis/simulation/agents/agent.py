"""
Agent: persona, memory and reflection turned into conversational behaviour.

Mechanism and rationale live in the README ("SFT architecture", "Agent
primitives", "SFT expression gate").

Public API:
    respond(message, speaker, softmax_stance, cue=None) -> (str, dict)
        Reply anchored to softmax_stance (+1/-1), enforced by the expression gate
        (_sft_gate). Returns the accepted reply and its gate record.
    classify_expression_graded(text) -> float | None
        The single classifier: graded stance in [-1, 1] relative to TOPIC_TEXT,
        0.0 = abstention, None = UNRELATED / unparseable.
    classify_expression(text) -> int | None
        Its sign, for memory tagging.
    graded_stance_prompt(text, first, second) -> str
        The classifier prompt; module-level so the audit rates through it.
    stance_score(direction, strength) -> float | None
        Label pair -> graded value, answer A read as +1.
    draft(message, speaker, stance, cue=None) -> str
        One ungated draft, no memory written; the classifier audit's corpus source.
    reflect(argmax_stance, cue=None)
        Insights from recent memories, stored with their classified stance.
    reward_from_expressions(graded_expr, graded_resp) -> float | None
        The social reward r = graded(expresser) * graded(reaction).
"""

import json
import random
from typing import TYPE_CHECKING

from langchain_ollama import OllamaLLM

from config import (
    MAX_MEMORIES_SEED,
    SEED,
    SFT_GATE_ENABLED,
    SFT_GATE_MAX_ATTEMPTS,
    STANCE_BRIDGE_ENABLED,
    TOPIC_OPENER_TOPIC_FIRST,
    TOPIC_OPENER_TOPIC_SECOND,
    TOPIC_TEXT,
    TOPIC_TEXT_OPPOSITE,
)
from memory.store import MemoryStore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Imported lazily in _get_st_model(), so offline log analysis never loads it.
_st_model = None

# Which pole the classifier shows as answer A, drawn per call: pinned, the rater's
# position bias would load onto the +1 stance; drawn, it is zero-mean in Q* = E[r]
# and estimable (metrics.order_bias). Own stream so per-call draws cannot shift
# the persona sample or topology the global stream seeds (final_schedule.md
# § Seeding).
_SLOT_RNG = random.Random(f"classifier-slot:{SEED}")


def _draw_slot() -> bool:
    """True when TOPIC_TEXT is shown as answer A this call, drawn fair."""
    return _SLOT_RNG.random() < 0.5


# The moderator's opener names one pole first; drawn per exchange for the same
# reason, on the generator's side. Separate from _SLOT_RNG so the two orders stay
# independent and each estimable.
_OPENER_RNG = random.Random(f"opener-order:{SEED}")


def draw_opener() -> tuple[str, bool]:
    """One of the two opener wordings, drawn fair.

    Returns:
        (opener, topic_first): topic_first is True when the wording names the
        TOPIC_TEXT pole first, so it means the same in a stance-flipped arm.
    """
    topic_first = _OPENER_RNG.random() < 0.5
    return (TOPIC_OPENER_TOPIC_FIRST if topic_first
            else TOPIC_OPENER_TOPIC_SECOND), topic_first


# Strength ladder: ordinal by concessions granted, evenly spaced because it claims
# no metric. BALANCED and NEITHER both score 0.0; the labels stay apart only for
# the audit. UNRELATED -> None.
_STRENGTH_VALUE: dict[str, float] = {
    "CLEAR": 1.0,          # grants the other answer nothing
    "PREDOMINANT": 2 / 3,  # grants the other answer at least one point
    "SLIGHT": 1 / 3,       # conditional, or concedes so much the lean barely shows
}
_DIRECTIONS: tuple[str, ...] = ("A", "B", "BALANCED", "NEITHER", "UNRELATED")

# `reasoning` first: the concessions that anchor `strength` are written out before
# it can be emitted, which guards against collapse to the endpoints.
_GRADED_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "direction": {"enum": list(_DIRECTIONS)},
        "strength": {"enum": [*_STRENGTH_VALUE, None]},
    },
    "required": ["reasoning", "direction", "strength"],
}


def stance_score(direction: str, strength: str | None) -> float | None:
    """Map a (direction, strength) pair to the graded stance in [-1, 1].

    The sign is relative to whichever pole was passed as ``first`` to
    ``graded_stance_prompt``; _classify_graded_slotted re-anchors it to TOPIC_TEXT.

    Returns None for UNRELATED and for A/B without a strength: a malformed reply
    is not an ambivalent one. _invoke_graded already rejects that pair; the branch
    guards the public function, which the audit also calls.
    """
    if direction == "UNRELATED":
        return None
    if direction in ("BALANCED", "NEITHER"):
        return 0.0
    if strength not in _STRENGTH_VALUE:
        return None
    return (1.0 if direction == "A" else -1.0) * _STRENGTH_VALUE[strength]


def _stance_sign(graded: float | None) -> int | None:
    """Collapse a graded stance to +1/-1/0; None stays None.

    The sign arbitrates the gate, indexes the Q-update and tags memories; only
    the magnitude feeds the reward.
    """
    if graded is None:
        return None
    if graded > 0:
        return 1
    if graded < 0:
        return -1
    return 0


def reward_from_expressions(
    graded_expr: float | None, graded_resp: float | None
) -> float | None:
    """Social reward r = graded(expresser) * graded(response), in [-1, 1].

    B&O's o_i * o_j with graded factors: same side reinforces, opposite sides
    penalise, each magnitude scales the signal. Symmetric by construction, since
    both factors come from one absolute classifier.

    Returns None when either factor is None (the caller freezes the round); an
    ambivalent reaction (0.0) gives a genuine r = 0.0, which still decays Q.
    """
    if graded_expr is None or graded_resp is None:
        return None
    return graded_expr * graded_resp


def _get_st_model() -> "SentenceTransformer":
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return _st_model


def _stance_pole(stance: int) -> str:
    """The proposition a stance holds: TOPIC_TEXT for +1, TOPIC_TEXT_OPPOSITE for -1.

    Never phrased as opposition: a "you oppose X" anchor yields drafts the
    two-answer classifier scores NEITHER (0.0) on the -1 side only.
    """
    return TOPIC_TEXT if stance == 1 else TOPIC_TEXT_OPPOSITE


def _stance_sentence(stance: int) -> str:
    """Anchoring sentence for a stance: the pole it holds, stated positively."""
    return f"Your position: {_stance_pole(stance)}."


# Conviction-cue bands 0-3, cut at the midpoints of the classifier's magnitude
# ladder (0, 1/3, 2/3, 1) so cue and instrument share a scale. Rules (README
# "The conviction cue"):
#   - never reuse SLIGHT / PREDOMINANT / CLEAR: corr(cue, |graded|) is the
#     manipulation check and must not score lexical echo;
#   - never name the other pole or invite a comparison. A hedging low band is
#     self-sustaining (abstain -> no update -> same band), so Tier 0's band-0
#     abstention rate gates the low-band wording;
#   - each band completes "You currently hold this position ...", so the
#     generator cannot drop it without breaking the sentence.
_CUE_BANDS: tuple[tuple[float, str], ...] = (
    (1 / 6, "tentatively; you are unsure about it"),
    (1 / 2, "with some conviction; you lean toward it"),
    (5 / 6, "with considerable conviction; you are fairly certain of it"),
    (1.01,  "with full conviction; you no longer question it"),
)


def _cue_clause(cue: float | None) -> str:
    """Completes the position header: " <band>." when cued, ":" when not.

    None (cue ablation) returns the bare colon, i.e. the uncued prompt byte for byte.
    """
    if cue is None:
        return ":"
    for upper, clause in _CUE_BANDS:
        if cue < upper:
            return f" {clause}."
    return f" {_CUE_BANDS[-1][1]}."


def _final_fallback_reply(softmax_stance: int) -> str:
    """Stage 3 of the gate: a fixed on-stance template, no LLM call.

    Tagged on-stance without classification; sound because it states the pole
    verbatim, i.e. one of the classifier's two answers.
    """
    return (
        f"I have thought about this carefully. My position: "
        f"{_stance_pole(softmax_stance)}. That is where I currently stand."
    )


def _gate_verdict(expressed: int | None, direction: str | None, softmax_stance: int) -> str | None:
    """Why the gate rejects a draft, or None if it passes.

    Passes the drawn sign and either abstention (BALANCED / NEITHER, 0).
    Rejects, as fail_reason:
        "flip"        the opposite sign;
        "unrelated"   UNRELATED: the model, not the speaker, left the discussion
                      (README "An agent abstaining vs. a model refusing");
        "unparseable" no valid reply after _invoke_graded's retry.
    The direction is read only to part the last two, which both score None.
    """
    if expressed is None:
        return "unrelated" if direction == "UNRELATED" else "unparseable"
    if expressed in (softmax_stance, 0):
        return None
    return "flip"


def _gate_record(
    attempts: list[dict], softmax_stance: int, fallback: bool, final: bool,
    cue: float | None = None,
) -> dict:
    """Assemble the per-turn gate record (README "The gate record").

    Keys:
        n_attempts:              drafts generated, incl. the accepted one
        passed:                  final text on-stance or abstaining (always True
                                 under the gate)
        fallback_used:           cascade reached stage 2
        final_fallback_used:     cascade reached stage 3
        first_attempt_expressed: sign of draft 1 (fidelity basis); None if unparseable
        final_expressed:         sign of the accepted text (Q-update index). Never None
        final_expressed_graded:  graded stance of the accepted text (reward factor)
        cue:                     raw conviction cue, None when uncued
        attempts:                [{stage, text, topic_first, expressed,
                                 expressed_graded [, fail_reason]}]; topic_first is
                                 the classifier slot drawn, None on the template
    """
    return {
        "n_attempts": len(attempts),
        "passed": attempts[-1]["expressed"] in (softmax_stance, 0),
        "fallback_used": fallback,
        "final_fallback_used": final,
        "first_attempt_expressed": attempts[0]["expressed"],
        "final_expressed": attempts[-1]["expressed"],
        "final_expressed_graded": attempts[-1]["expressed_graded"],
        "cue": cue,
        "attempts": attempts,
    }


def graded_stance_prompt(text: str, first: str, second: str) -> str:
    """Rate which of two named positions the STATEMENT supports.

    A forced choice between two answers, with no favor/oppose vocabulary and no
    question shown, so the only presentation asymmetry is the answer order.
    ``direction`` is relative to ``first``/``second``. Module-level so the
    classifier audit rates through this exact prompt, in both orders.
    """
    return (
        f'Task: Determine which of the two positions A and B the STATEMENT supports.\n\n'
        f"A: {first}\n"
        f"B: {second}\n\n"
        f"Judge only the substantive position. Politeness, tone, and quality of "
        f"justification change neither the direction nor the strength; only "
        f"substantive concessions and conditions do.\n\n"
        f"STATEMENT:\n{text}\n\n"
        f"In your reasoning, first decide which answer the STATEMENT ultimately "
        f"supports, then list every concession, qualification, or condition it "
        f"grants to the other answer.\n\n"
        f"direction:\n"
        f"  A         = speaker supports A over B\n"
        f"  B         = speaker supports B over A\n"
        f"  BALANCED  = speaker finds merit in both A and B without settling on either\n"
        f"  NEITHER   = speaker rejects both A and B, or holds a third position\n"
        f"  UNRELATED = not the speaker answering: off-topic, or about the prompt, persona, or model\n\n"
        f"strength (only if direction is A or B, otherwise null):\n"
        f"  CLEAR       = grants the other answer nothing\n"
        f"  PREDOMINANT = grants the other answer at least one point\n"
        f"  SLIGHT      = supports it only under a condition, or concedes so much "
        f"that the lean is barely discernible"
    )


def _invoke_graded(
    llm: OllamaLLM, prompt: str, context: str
) -> tuple[str, str | None] | None:
    """Invoke llm under the JSON-schema-constrained (direction, strength) choice.

    Retried once on a malformed reply, including A/B without a strength. A
    strength beside a non-A/B direction is meaningless and normalised to None.

    Args:
        llm: OllamaLLM instance to invoke.
        prompt: Full instruction, from graded_stance_prompt.
        context: Caller name, prefixed to the warning for log grep-ability.

    Returns:
        (direction, strength) with strength set only for A / B, or None if no
        valid pair after one retry.
    """
    raw = ""
    for _ in range(2):
        raw = llm.invoke(prompt, format=_GRADED_SCHEMA)
        try:
            reply = json.loads(raw)
            direction, strength = reply["direction"], reply["strength"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if direction in ("A", "B"):
            if strength in _STRENGTH_VALUE:
                return direction, strength
        elif direction in _DIRECTIONS:
            return direction, None

    print(
        f"\n  ⚠️  {context}: no valid direction/strength in reply after retry, "
        f"falling back. Raw text (tail): {raw[-150:]!r}"
    )
    return None


class Agent:
    def __init__(self, name: str, persona: str, llm: OllamaLLM):
        """Initialise an agent with a fixed identity, persona, and isolated memory store.

        Args:
            name:    Display name used in prompts and as the ChromaDB collection key.
            persona: Role description, never stored in memory. Injected on the
                     identity line, or via _mold_stance with STANCE_BRIDGE_ENABLED.
            llm:     Shared OllamaLLM instance.
        """
        self.name = name
        self.persona = persona
        self.llm = llm
        self.memory = MemoryStore(name)
        self._stance_bridges = (
            self._derive_stance_bridges() if STANCE_BRIDGE_ENABLED else {}
        )

    # private helpers --------------------------------------------------------

    def _mold_stance(self, stance: int) -> str:
        """Mold the persona into a stance justification (STANCE_BRIDGE_ENABLED only).

        Args:
            stance: +1 (pro) or −1 (contra).

        Returns:
            A persona-grounded justification sentence for stance.
        """
        return self.llm.invoke(
            f"You are {self.name}. {self.persona}\n\n"
            f"People often hold positions that do not fit seamlessly with all of their other "
            f"values which is human and no contradiction that needs pointing out.\n\n"
            f"Your position: {_stance_pole(stance)}.\n\n"
            f"In 2-3 sentences, state why you, as {self.name}, hold that position. "
            f"Ground it in your character and life background but do not soften, hedge, or "
            f"argue against the position itself.\n\n"
            f"Reply with only the justification in the first person. No introduction, "
            f"no meta-commentary or remarks about the role prompt, no parenthetical asides."
        ).strip()

    def _derive_stance_bridges(self) -> dict[int, str]:
        """Cache a molded bridge sentence for each possible stance, at init."""
        return {1: self._mold_stance(1), -1: self._mold_stance(-1)}

    def _identity(self) -> str:
        """Opening line: the name, plus the raw (second-person) persona unless bridged."""
        return (f"You are {self.name}." if STANCE_BRIDGE_ENABLED
                else f"You are {self.name}. {self.persona}")

    def _position_block(self, stance: int, cue: float | None,
                        bridge: bool = True) -> str:
        """The stance anchor: header with cue, the pole, and the bridge if enabled.

        ``bridge=False`` is gate stage 2, which drops persona context but keeps the cue.
        """
        line = (f"{self._stance_bridges[stance]}\n"
                if bridge and STANCE_BRIDGE_ENABLED else "")
        return (f"You currently hold this position{_cue_clause(cue)}\n"
                f"{_stance_sentence(stance)}\n{line}")

    def _embed(self, text: str) -> list:
        """Return a vector embedding for text using sentence-transformers."""
        return _get_st_model().encode(text).tolist()

    def _score_importance(self, text: str) -> float:
        """LLM-rated significance on 1-10; 5.0 if unparseable."""
        raw = self.llm.invoke(
            f"Rate the significance of this memory for {self.name} "
            f"on a scale from 1 (trivial) to 10 (highly significant). "
            f"Reply with only a single number.\nMemory: {text}"
        ).strip()
        try:
            return float(raw.split()[0])
        except (ValueError, IndexError):
            return 5.0

    def _store(self, content: str, mem_type: str = "interaction", stance: int = 0) -> None:
        """Compute importance and embedding for content, then persist it.

        Args:
            stance: Stance this memory was formed under (+1, −1, or 0 =
                    neutral/unknown). Tags the memory for later stance-congruent
                    retrieval biasing; see memory/scoring.py.
        """
        importance = self._score_importance(text=content)
        self.memory.add(
            content=content,
            mem_type=mem_type,
            importance=importance,
            embedding=self._embed(text=content),
            stance=stance,
        )

    def _retrieve(self, context: str, stance: int | None = None) -> list[dict]:
        """Retrieve memories most relevant to context using embedding similarity.

        Args:
            stance: If given, biases retrieval toward memories tagged with
                    this stance (see memory/scoring.py). None leaves
                    retrieval stance-unbiased.
        """
        return self.memory.retrieve(self._embed(text=context), query_stance=stance)

    # reflection -------------------------------------------------------------

    def reflect(self, argmax_stance: int, cue: float | None = None) -> None:
        """Two-step reflection (Park et al. 2023); not gated.

        Step 1: two questions from the MAX_MEMORIES_SEED most recent memories.
        Step 2: one insight per question from retrieved memories, stored as a
                "reflection" memory tagged with its classified stance.

        Args:
            argmax_stance: the agent's Q-argmax stance (+1 or -1).
            cue:           conviction cue for that stance, or None when uncued;
                           both steps carry it.
        """
        recent = self.memory.all_recent(limit=MAX_MEMORIES_SEED)
        if not recent:
            return
        recent_block = "\n".join(f"- {m}" for m in recent)

        position = self._position_block(argmax_stance, cue)

        # Step 1: identify meaningful questions from recent experience
        questions_raw = self.llm.invoke(
            f"{self._identity()}\n\n"
            f"{position}\n"
            f"Recent experiences:\n{recent_block}\n\n"
            f"Name 2 meaningful questions you should reflect on, one per line. "
            f"No numbering, no introduction."
        )
        questions = [
            q.strip() for q in questions_raw.strip().splitlines() if q.strip()
        ][:2]

        # Step 2: synthesise an insight for each question
        for question in questions:
            mems = self._retrieve(question, stance=argmax_stance)
            mem_block = "\n".join(f"- {m['content']}" for m in mems)

            insight = self.llm.invoke(
                f"{self._identity()}\n\n"
                f"{position}\n"
                f"Question: {question}\n"
                f"Relevant memories:\n{mem_block}\n\n"
                f"State a single insight or conclusion in the first person."
            ).strip()

            # Not gated, so None collapses to a neutral tag.
            insight_stance = self.classify_expression(insight)
            self._store(
                f"[Reflection] {insight}",
                mem_type="reflection",
                stance=insight_stance if insight_stance is not None else 0,
            )
            print(f"\n 💭 {self.name} reflects: {insight}")

    # respond / expression gate -----------------------------------------------

    def _generate_reply(
        self, message: str, speaker: str, softmax_stance: int, mems: list[dict],
        cue: float | None = None,
    ) -> str:
        """One draft from the soft-anchored prompt (gate stage 1)."""
        mem_block = (
            "\n".join(f"[{m['type']}] {m['content']}" for m in mems)
            if mems else "(none yet)"
        )
        return self.llm.invoke(
            f"{self._identity()}\n\n"
            f"Relevant memories:\n{mem_block}\n\n"
            f"{speaker} says: \"{message}\"\n\n"
            f"{self._position_block(softmax_stance, cue)}\n"
            f"Reply as {self.name} to {speaker}'s statement. Express your position "
            f"stated above and justify it from your personal experience and character. "
            f"Answer in 2-3 sentences."
        ).strip()

    def _fallback_reply(
        self, message: str, speaker: str, softmax_stance: int, cue: float | None = None
    ) -> str:
        """Gate stage 2: _generate_reply without memories and persona; same wording, cue kept."""
        return self.llm.invoke(
            f"You are {self.name}.\n\n"
            f"{speaker} says: \"{message}\"\n\n"
            f"{self._position_block(softmax_stance, cue, bridge=False)}\n"
            f"Reply as {self.name} to {speaker}'s statement. Express your position "
            f"stated above and justify it from your personal experience and character. "
            f"Answer in 2-3 sentences."
        ).strip()

    def draft(
        self, message: str, speaker: str, stance: int, cue: float | None = None
    ) -> str:
        """One ungated first draft anchored to ``stance``; no memory written.

        Gate stage 1 stopped before classification, i.e. the population the
        classifier meets in a run. Used only by the classifier audit, which
        stratifies ``cue`` across the four bands (None is the uncued ablation).
        """
        mems = self._retrieve(message, stance=stance)
        return self._generate_reply(message, speaker, stance, mems, cue)

    def _sft_gate(
        self, message: str, speaker: str, softmax_stance: int, cue: float | None = None
    ) -> tuple[str, dict]:
        """Generate a reply expressing softmax_stance, by rejection sampling.

        Stages (README "The mechanism"): 1. up to SFT_GATE_MAX_ATTEMPTS
        soft-prompt drafts; 2. as many context-free drafts; 3. the fixed template.
        Pass rule: _gate_verdict. Each stage stops early on a verbatim repeat;
        memories are retrieved once per call.

        Returns:
            (accepted_text, gate_record); see _gate_record.
        """
        mems = self._retrieve(message, stance=softmax_stance)
        attempts: list[dict] = []
        draft_no = 0

        prev_text = None
        for _ in range(SFT_GATE_MAX_ATTEMPTS):                    # stage 1: soft prompt
            draft_no += 1
            print(f"    ↻ {self.name} draft {draft_no} (main)")
            text = self._generate_reply(message, speaker, softmax_stance, mems, cue)
            graded, direction, topic_first = self._classify_graded_slotted(text)
            expressed = _stance_sign(graded)
            entry = {"stage": "main", "text": text, "topic_first": topic_first,
                     "expressed": expressed, "expressed_graded": graded}
            fail_reason = _gate_verdict(expressed, direction, softmax_stance)
            if fail_reason is None:
                attempts.append(entry)
                return text, _gate_record(attempts, softmax_stance, fallback=False,
                                          final=False, cue=cue)
            entry["fail_reason"] = fail_reason
            attempts.append(entry)
            if text == prev_text:
                break
            prev_text = text

        prev_text = None
        for _ in range(SFT_GATE_MAX_ATTEMPTS):                    # stage 2: context-free fallback
            draft_no += 1
            print(f"    ↻ {self.name} draft {draft_no} (fallback)")
            text = self._fallback_reply(message, speaker, softmax_stance, cue)
            graded, direction, topic_first = self._classify_graded_slotted(text)
            expressed = _stance_sign(graded)
            entry = {"stage": "fallback", "text": text, "topic_first": topic_first,
                     "expressed": expressed, "expressed_graded": graded}
            fail_reason = _gate_verdict(expressed, direction, softmax_stance)
            if fail_reason is None:
                attempts.append(entry)
                return text, _gate_record(attempts, softmax_stance, fallback=True,
                                          final=False, cue=cue)
            entry["fail_reason"] = fail_reason
            attempts.append(entry)
            if text == prev_text:
                break
            prev_text = text

        text = _final_fallback_reply(softmax_stance)              # stage 3: fixed template
        attempts.append({"stage": "final", "text": text, "topic_first": None,
                         "expressed": softmax_stance,
                         "expressed_graded": float(softmax_stance)})
        return text, _gate_record(attempts, softmax_stance, fallback=True, final=True,
                                  cue=cue)

    def respond(
        self,
        message: str,
        speaker: str,
        softmax_stance: int,
        cue: float | None = None,
    ) -> tuple[str, dict]:
        """Gated reply to message from speaker, stored as a memory.

        With SFT_GATE_ENABLED = False: one ungated draft in a one-attempt gate
        record, same return shape (ablation arm). The caller indexes the Q-update
        on final_expressed and skips it when 0.

        Args:
            message:        incoming message text.
            speaker:        agent or moderator who sent it.
            softmax_stance: +1 or -1, from draw_softmax_stance().
            cue:            conviction cue in [0, 1], or None when uncued.

        Returns:
            (reply, gate_record); see _gate_record.
        """
        if SFT_GATE_ENABLED:
            reply, record = self._sft_gate(message, speaker, softmax_stance, cue)
        else:
            mems = self._retrieve(message, stance=softmax_stance)
            reply = self._generate_reply(message, speaker, softmax_stance, mems, cue)
            # No retry without the cascade: None (UNRELATED or unparseable)
            # collapses to 0 and is treated as an abstention.
            graded, _direction, topic_first = self._classify_graded_slotted(reply)
            expressed = _stance_sign(graded)
            attempts = [{
                "stage": "main",
                "topic_first": topic_first,
                "text": reply,
                "expressed": expressed if expressed is not None else 0,
                "expressed_graded": graded if graded is not None else 0.0,
            }]
            record = _gate_record(attempts, softmax_stance, fallback=False,
                                  final=False, cue=cue)

        self._store(
            f"{speaker} said: '{message}'. I replied: '{reply}'",
            stance=record["final_expressed"],
        )
        return reply, record

    # classify_expression -----------------------------------------------------

    def classify_expression_graded(self, text: str) -> float | None:
        """Graded stance of one text in [-1, 1], relative to TOPIC_TEXT.

        Arbitrates the gate and tags memories (sign) and supplies both reward
        factors (value). Minimal context and absolute, not relational, so the
        reward is symmetric by construction. The answer slot is drawn per call and
        the score re-anchored to TOPIC_TEXT.

        Returns:
            + supports TOPIC_TEXT, - the opposite pole, 0.0 abstains (BALANCED /
            NEITHER), None for UNRELATED or an unparseable reply.
        """
        return self._classify_graded_slotted(text)[0]

    def _classify_graded_slotted(self, text: str) -> tuple[float | None, str | None, bool]:
        """classify_expression_graded, plus the raw direction and the slot drawn.

        The direction lets the gate tell UNRELATED from a malformed reply (both
        None); the slot is logged so the position bias is estimable.

        Returns:
            (graded, direction, topic_first); direction is None when unparseable.
        """
        topic_first = _draw_slot()
        first, second = ((TOPIC_TEXT, TOPIC_TEXT_OPPOSITE) if topic_first
                         else (TOPIC_TEXT_OPPOSITE, TOPIC_TEXT))
        prompt = graded_stance_prompt(text, first, second)
        graded = _invoke_graded(self.llm, prompt, "classify_expression")
        if graded is None:
            return None, None, topic_first
        score = stance_score(*graded)
        # Negate when A was the -1 pole; `or 0.0` keeps -0.0 out of the log.
        if score is not None and not topic_first:
            score = -score or 0.0
        return score, graded[0], topic_first

    def classify_expression(self, text: str) -> int | None:
        """Sign of classify_expression_graded(): +1/-1/0, or None."""
        return _stance_sign(self.classify_expression_graded(text))
