# Multi-Agent Deliberation Simulation

A hybrid LLM + Social Feedback Theory (SFT; Banisch & Olbrich 2019) simulation of opinion dynamics on a structured network, run on local models via [Ollama](https://ollama.ai).

**Research question:** Can Social Feedback Theory serve as the underlying reinforcement-learning mechanism for LLM agents embedded in a structured social network such that the resulting hybrid model preserves SFT's analytic interpretability while extending the empirical scope?

| Mode | Entry point | Description |
|---|---|---|
| **Network** | `main_network.py` | N agents on a two-block graph; the canonical driver |
| **Pairwise** | `main_pairwise.py` | The same mechanism for two agents on a single edge |

Local Ollama only (`qwen3.8-nodraft:27b`; the pilot ran `qwen3.5:35b`), no orchestration framework ([why](#what-not-to-change-without-good-reason)). Embeddings via `sentence-transformers` on CPU.

---

## Terminology

| Term | Definition |
|---|---|
| **Turn** | One `respond()` call: one agent speaks once |
| **Exchange** | Both agents speak once (2 turns) |
| **Interaction** | One expresser, one responder, one exchange, one Q-update (expresser only) |
| **Round** | `INTERACTIONS_PER_ROUND` interactions; the snapshot unit |
| **Argmax stance** | `argmax(q_pos, q_neg)`: metrics, logging, the `reflect()` anchor |
| **Softmax stance** | The per-interaction β-softmax draw that anchors `respond()` and arbitrates its [expression gate](#sft-expression-gate) |
| **Expressed stance** | The sign the classifier reads off a draft: ±1, or `0` = **ambiguous** (an abstention). The accepted draft's sign (`final_expressed`) indexes the Q-update, which is skipped at `0` |
| **Ambiguous / ambivalent** | *Ambiguous* is an expression with no stance (the expresser abstains); *ambivalent* is a reaction graded `0.0`, a genuine reward `r = 0` |

Schedule terminology (stage / sub-stage / run / cohort) is defined in [`../CLAUDE.md`](../CLAUDE.md) and parsed by `analysis/runs.py`. "Stage" also names the gate's three cascade stages; the two senses are unrelated.

---

## Architecture

```
config.py               single source of truth for all constants
agents/agent.py         Agent: respond() + the expression gate _sft_gate(), draft(),
                        reflect(); classify_expression_graded(), graded_stance_prompt(),
                        stance_score(), reward_from_expressions()
agents/personas.py      samples GSS survey records, expands them to name + persona via the LLM
memory/store.py         per-agent ChromaDB collection
memory/scoring.py       composite retrieval score: recency, importance, relevance, stance congruence
network/opinion.py      Q-state: AgentOpinionState, draw_softmax_stance(), conviction_cue(),
                        update_q_value(), polarization metrics
network/state.py        NetworkState, EdgeData, new_edge_data()
network/matching.py     select_responder() (1-hop), expected_reward_via_referral() and
                        rewire_after_drop() (2-hop tie formation)
network/discussion.py   run_discussion(): gated exchange + reward product
network/edges.py        record_reward(), update_edge()
network/logger.py       SimulationLogger: params.json, personas.json, events.jsonl, snapshots
main_network.py, main_pairwise.py   entry points
run_stage.sh            SGE batch runner, one sub-stage per job (RUNBOOK.md)
run_sweep.py            local grid sweep
data/                   GSS2024.dta and the derived gss_personas.json (git-ignored)
```

Everything that reads the output lives in [`../analysis/`](../analysis/README.md).

---

## SFT architecture

The Q-update lives entirely in code. The LLM generates each utterance through the [expression gate](#sft-expression-gate) and grades each turn's stance (`classify_expression_graded()`); the reward is the product of two such grades, not a further LLM call. Every opinion shift therefore traces to an auditable Q-trajectory.

### Agent internal state

- **`AgentOpinionState`**: `q_pos`, `q_neg`, both starting at 0, so any divergence is caused by social feedback rather than model priors.
- **Memory**: a ChromaDB collection of past interactions and reflections, retrieved into `respond()` and `reflect()` prompts.
- **Persona**: an English profile derived from US GSS data, never stored in memory. It sits verbatim on the identity line (`_identity()`), beside a stance anchor it has not been reconciled with. `STANCE_BRIDGE_ENABLED = True` instead molds persona and stance into a cached per-stance justification (`_mold_stance()`). Either way the anchoring is **soft**: the persona can pull a draft off-stance, the gate rejects the draft, and the pull is logged as first-attempt fidelity. That tension between social-feedback learning and the persona is a study object.

Persona and memory condition generation only; the Q-update sees only the reward and the stance index.

### The conviction cue

**On in every arm (`CONVICTION_CUE_ENABLED = True`)**; `SIM_CONVICTION_CUE=0` is an ablation that reproduces the uncued prompt byte for byte.

The softmax draw carries the *sign* of Q into the prompt. The cue carries the *magnitude*, so expressed commitment can track learned commitment instead of being a fixed persona trait. `conviction_cue()` returns `max(0, stance × q_gap) / 2`, and `_cue_clause()` bands it into a sub-phrase completing the stance header: *"You currently hold this position with considerable conviction."* A clause rather than a separate sentence, because the generator could drop a separate sentence without breaking anything.

- **The clamp.** `stance` is the pole the prompt anchors to. An off-argmax draw (probability `σ(−β·|Δq|)`) argues the lower-valued pole, so stance-blind `|q_gap|` would claim conviction the agent does not have; the clamp sends that case to band 0. In `reflect()` the anchor is the argmax, so the clamp is inert.
- **Linear, not `tanh`.** Rewards in `[−1, 1]` and the convex TD update bound `q_gap` to `[−2, 2]` exactly, so the scale belongs to the mechanism and needs no refitting per model. `tanh` would be flat in the early rounds, where the regime is decided.
- **The bands** are cut at the midpoints of the classifier's magnitude ladder, so cue and instrument share a scale, but they never reuse `SLIGHT`/`PREDOMINANT`/`CLEAR`: `corr(cue, |graded|)` is the manipulation check and must not score lexical echo. No band names the other pole or invites a comparison. The low bands speak of certainty ("unsure", "lean toward"), which is a risk: a hedging draft scores `BALANCED`/`NEITHER`, abstains, freezes the update and keeps the agent in band 0, where every run starts. **Tier 0's band-0 abstention rate gates this wording** before any run is queued.

| Band | Clause | cue | `stance × q_gap` | nearest graded level |
|---|---|---|---|---|
| 0 | tentatively; you are unsure about it | `[0, 1/6)` | < 0.33 (and every clamped turn) | 0 |
| 1 | with some conviction; you lean toward it | `[1/6, 1/2)` | 0.33–1.00 | 1/3 |
| 2 | with considerable conviction; you are fairly certain of it | `[1/2, 5/6)` | 1.00–1.67 | 2/3 |
| 3 | with full conviction; you no longer question it | `[5/6, 1]` | 1.67–2.00 | 1 |

**Expected effect: a damper.** LLM agents write near `CLEAR` by default, and every band but the top one points below that, so a cue that is read at all pulls `|graded|` down. That shrinks `|r|`, hence `Q* = E[r]`, hence `|q_gap|`: negative feedback on the cue's own scale, which lengthens settling (the reason `A-00` sets the horizon) and moves every graded null. The hedger ceiling may put band 3 out of reach; it stays as a marker on the mechanism's scale. A cue that moves `|graded|` not at all means the mechanism is inert, not null.

**Scope.** Both gate stages of `respond()` and both steps of `reflect()` carry the cue; the raw value is logged per turn (`gate.cue`). Cued reflections are stored as memories, so a tentatively worded insight can resurface later; that path pushes the same way as the main effect. The classifier-audit corpus is drafted across all four bands, because an uncued corpus is no longer the population the classifier meets in a run.

**Cost.** The cue hardens the anchor at high bands, so the flip rate becomes conviction-dependent. It is also the one channel `analysis/benchmark/benchmark_sft.py` cannot import: it lives in the model's response to a sentence. Of the benchmark's reward variants, `graded-agent` holds per-agent magnitude constant and is the right comparison; the three graded nulls fitted on Bracket A inherit the cue's *level* through the marginal reward but draw it iid, so they cannot produce the low-Q trap (band 0 → hedged text → small `|r|` → band 0). A `graded-conviction` variant keyed on `(dyad cell × cue band)` would. **It is planned, not implemented** (`final_schedule.md` § Bracket B).

### Interaction loop (one asymmetric interaction)

```
1. Draw expresser     uniform over agents with ≥ 1 neighbour
2. Draw responder     select_responder(): weight exp(β_sel · mean reward the expresser got
                      from that neighbour); β_sel = 0 is uniform
3. Softmax stance     p(+1) = σ(β · q_gap); β = inf is the deterministic argmax
                      (a tie stays a fair coin); plus the conviction cue
4. Exchange           expresser speaks, responder reacts, both through the gate.
                      First meeting: the expresser answers the moderator's opener,
                      drawn per exchange between two wordings; repeat meeting: the
                      partner's last message
5. Reward             r = graded(expression) · graded(reaction) ∈ [−1, 1], read off
                      the gate records
6. Q-update           Q(final_expressed) ← (1−α)·Q + α·r   [expresser only]
7. Reward record      record_reward(): r into the edge's rolling window (REWARD_WINDOW_M)
8. Edge update        update_edge()  [GRAPH_DYNAMIC]
9. Rewire             rewire_after_drop()  [GRAPH_DYNAMIC, REWIRE_RULE != "off"]
   Steps 6-9 run only on a measured-reward round: a non-zero expressed stance
   and a non-None reward. Otherwise all of them freeze.
```

### Two reward signals

| Signal | Source | Used for |
|---|---|---|
| `reward_a` | graded(a's first turn) · graded(b's last turn) | Q-update, reward record, and (dynamic graph) edge valuation, all on measured-reward rounds only |
| `reward_b` | the mirror | Unused; `None` at one turn each, since a's turn precedes b's |

Each factor is an independent, context-free grading of a single turn, so expression and reward stay causally separate, and both come from one absolute classifier, so the reward is **symmetric by construction**: same side reinforces, opposite sides penalise, whichever side. (An earlier relational reward judge read mutual rejection as disagreement and tilted consensus toward one pole.) A `None` reaction (UNRELATED or unparseable) freezes the round exactly as an abstention does.

### How a reward moves the Q-values

`Q ← (1−α)·Q + α·r` is an exponential moving average, so **`Q(stance)` converges to the mean reward that stance earned**; a sustained `r` has fixed point `Q = r`.

- The sign of `r` sets the direction.
- **`r = 0` is not a no-op.** An ambivalent reaction applies `Q ← (1−α)·Q`, about a 10 % shrink per hit at `α = 0.1`, which narrows `|q_gap|` and randomises the next draw: mixed feedback erodes conviction. The binary model cannot represent this, and Sarközi et al. (2022) found ambivalent feedback the most influential.

Hence the split the classifier keeps: `BALANCED`/`NEITHER` → `0.0` (applied) versus `UNRELATED`/parse failure → `None` (skipped). Fabricating a `0.0` where nothing was measured would inject a moderating force; skipping a genuine `0.0` would discard the graded extension.

---

## Agent primitives

### `respond(message, speaker, softmax_stance, cue=None) -> (str, dict)`

Retrieves up to `MAX_MEMORIES_RETRIEVE` memories, biased toward the anchored stance (see [Memory system](#memory-system)), and generates a reply **through the [expression gate](#sft-expression-gate)**. Returns the accepted reply and its gate record, and stores the exchange as a memory tagged with `final_expressed`. With `SFT_GATE_ENABLED = False` it makes one ungated draft in the same return shape.

**Do not soften the position-taking instruction**: the classifier needs a legible stance.

### `classify_expression_graded(text) -> float | None`

The single classifier. A **forced choice between two named answers**: `TOPIC_TEXT` and `TOPIC_TEXT_OPPOSITE`, with **no question shown**, since both are self-contained propositions and a question would name one pole first in every call. Minimal prompt: no persona, memory or transcript. **Absolute, not relational**: it judges one text against the two answers, never one text against another, which is what makes the reward symmetric.

Favor/oppose vocabulary is out of the rubric, so the only presentation asymmetry left is which answer comes first. **That order is drawn per call** (`_SLOT_RNG`) and the score re-anchored to `TOPIC_TEXT`, so the rater's position bias is zero-mean in `Q* = E[r]` instead of loading onto the +1 stance. The drawn slot is logged per attempt (`topic_first`), which makes the bias estimable from a run's own events (`metrics.order_bias`).

The reply is `{reasoning, direction, strength}` under Ollama's JSON-schema `format`:

- `direction`: `A`, `B`, `BALANCED` (merit in both, settles on neither), `NEITHER` (rejects both, or holds a third position), `UNRELATED` (not the speaker answering: off-topic, or about the prompt, persona or model).
- `strength` (A/B only), a **concession-counting rubric**: `CLEAR` grants the other answer nothing, `PREDOMINANT` at least one point, `SLIGHT` supports its answer only conditionally or barely. `reasoning` comes first and lists the concessions, so `strength` cannot be emitted before they are written out, which keeps the judge from saturating at the endpoints. Gate-enforced turns are unqualified statements and legitimately earn ±1; the rubric makes the intermediate levels reachable for turns that do concede.

`stance_score()` maps this to ±1 / ±2⁄3 / ±1⁄3 (evenly spaced, since the ladder is only ordinal), `0.0` for `BALANCED`/`NEITHER`, `None` for `UNRELATED`. `_invoke_graded()` retries a malformed reply once (including A/B without a strength), so in a run `None` means `UNRELATED` or two failed attempts. `BALANCED` and `NEITHER` stay apart only so the [audit](../analysis/classifier/README.md) can tell them apart.

### `classify_expression(text) -> int | None`

The sign of the above; `reflect()` tags insights with it.

### `reward_from_expressions(graded_expr, graded_resp) -> float | None`

`r = graded_expr · graded_resp`, the `o_i · o_j` form with graded factors; `None` when either factor is `None`.

**Why both magnitudes multiply.** The expresser's own commitment scales every reward it receives, so a habitual hedger's Q converges to at most its own α-weighted mean `|graded|` (the **hedger ceiling**; measured in [`../analysis/design_analysis.ipynb`](../analysis/design_analysis.ipynb) § 4). This is kept on purpose, on four grounds: (1) it is the literal continuous generalisation of B&O's `r = o_i · o_j`, where `o_i` is already the agent's own expressed opinion; (2) product updates are the coactivation (Hebbian) form of connectionist attitude models (van Overwalle & Siebler 2005; Monroe & Read 2008); (3) attitudes consolidate with how unambiguously they were expressed (Bem 1972; Kiesler 1971; Downing, Judd & Brauer 1992; Higgins & Rholes 1978); (4) emphatic approval is a larger reinforcer than tepid assent (Verplanck 1955; Hildum & Brown 1956; Insko 1965). The rival reading, `r = sign(own) · graded(reaction)`, is a planned ablation. *References recalled from memory; verify before citing.*

### `reflect(argmax_stance, cue=None)`

Two steps (Park et al. 2023): two questions from the `MAX_MEMORIES_SEED` most recent memories, then one insight per question from stance-biased retrieval. Both steps are anchored to the argmax stance and carry the cue. Insights are stored as high-importance `"reflection"` memories tagged with their classified stance. Runs every `REFLECT_EVERY` rounds when `MEMORY_ENABLED`. **Not gated**: insights never enter an interaction, and gating would suppress the persona-driven consolidation memory exists to exhibit.

### `draft(message, speaker, stance, cue=None) -> str`

Gate stage 1 stopped before classification, with no memory written: the population the classifier meets in a run. Used only by the classifier audit's corpus.

---

## SFT expression gate

`_sft_gate()` in `agents/agent.py` keeps the learning loop canonical while agents keep opinionated personas.

### The intuition

Canonical SFT assumes an agent expresses the stance it drew. An LLM with a persona often writes the opposite when the draw lands on the stance the persona dislikes; the reward would then score an utterance the Q-machinery never selected, and the dispreferred stance would starve for feedback. The gate enforces the draw **output-side**: the prompt stays soft, so the *first* draft still samples the persona's unconstrained pull (the **persona field**), but a draft enters the conversation only if its sign does not contradict the draw. The pull moves from silently corrupting the update to being logged.

**Abstentions pass.** A `BALANCED` or `NEITHER` draft is not a counter-stance and carries no evidence of persona override; rejecting it would spend retries on legibility noise and contaminate the retry count. It enters the conversation as a **strategic abstention**: the speaker puts no side up for evaluation, and the round's Q-update, reward record and edge dynamics all freeze.

### An agent abstaining vs. a model refusing

What must not pass is the **model** stepping out of the discussion: writing about the prompt, the persona or itself instead of taking a turn. Letting that through would put meta-commentary into the transcript and apply `r = 0.0` on whichever pole the model balks at, a stance-dependent decay and exactly the asymmetry positive pole anchoring (`_stance_pole()`) removes.

**The line is drawn in the rubric, not in the gate.** `graded_stance_prompt()` sends an abstention to `NEITHER`/`BALANCED` (`0.0`, passes) and a frame break to `UNRELATED` (`None`, rejected). The gate sees a number, not text, and prose pattern-matching misfires on emphatic in-character writing ("I cannot support a system that leaves my children with radioactive waste" is an on-stance −1 turn). Stage 2 of the cascade drops the persona, which is the right rescue for a persona-conflict refusal. `_gate_verdict()` records `"unrelated"` apart from `"unparseable"`, so the refusal rate is countable per anchored stance (`metrics.load_gate`'s `n_unrelated`, `persona_field`'s `unrelated`). A rate that differs between anchors is the model's reluctance, not persona pressure (`n_flip`).

### The mechanism

Rejection sampling in three cascade stages. A draft **passes** iff its sign equals the draw or is `0`; a flip, an `UNRELATED` draft and an unparseable reply are retried, with `fail_reason` recorded.

1. **main**: up to `SFT_GATE_MAX_ATTEMPTS` drafts from the soft prompt. Retries resample the same prompt, so rejected drafts are an i.i.d. measure of persona pressure.
2. **fallback**: up to `SFT_GATE_MAX_ATTEMPTS` drafts from the same prompt without memories or anything persona-derived (cue kept), which makes this stage identical under both bridge settings.
3. **final**: a fixed template, `My position: <pole>.`, tagged on-stance without classification. Sound because it states one of the classifier's two answers verbatim; re-verify once if `TOPICS` changes.

A stage stops early if a draft repeats verbatim; memories are retrieved once per turn. Since the cascade emits only on-stance or abstaining text, a performed update always indexes the draw under the gate; in the ungated ablation (`SFT_GATE_ENABLED = False`) the same code path indexes the classified sign. `SFT_GATE_MAX_ATTEMPTS` defaults to 5.

**Cost.** Each attempt is one generation plus one classification, so turn cost scales with 1 / P(first draft on-stance). Realised multipliers are readable from `n_attempts`, `fallback_used` and `final_fallback_used` in `events.jsonl`.

### The gate record (persona-field telemetry)

Every turn in `events.jsonl` carries one:

```json
{
  "n_attempts": 4,
  "passed": true,
  "fallback_used": false,
  "final_fallback_used": false,
  "first_attempt_expressed": -1,
  "final_expressed": 0,
  "final_expressed_graded": 0.0,
  "cue": 0.21,
  "attempts": [
    {"stage": "main", "text": "...", "topic_first": true,  "expressed": -1,   "expressed_graded": -1.0, "fail_reason": "flip"},
    {"stage": "main", "text": "...", "topic_first": false, "expressed": null, "expressed_graded": null, "fail_reason": "unparseable"},
    {"stage": "main", "text": "...", "topic_first": true,  "expressed": null, "expressed_graded": null, "fail_reason": "unrelated"},
    {"stage": "main", "text": "...", "topic_first": false, "expressed": 0,    "expressed_graded": 0.0}
  ]
}
```

With anchor `+1`: draft 1 flipped, draft 2 was unparseable, draft 3 was a frame break, and draft 4 abstained and passed. `final_expressed` is never `null`. `topic_first` is the classifier slot drawn (`null` on the template); because it is drawn independently of the anchor, the two are crossed in the log and the rater's position bias is estimable.

First-attempt fidelity (`first_attempt_expressed` against the anchor), flip / ambiguous / unrelated / unparseable rates, retry counts, fallback and template rates, and drift across a turn's drafts are all recoverable post-hoc at no LLM cost: `analysis/metrics.load_gate` and `persona_field()` (see [`../analysis/README.md`](../analysis/README.md) § Persona-field analysis).

---

## Memory system

One isolated [ChromaDB](https://www.trychroma.com) collection per agent. Retrieval scores the whole collection:

```
score = WEIGHT_RECENCY·recency + WEIGHT_IMPORTANCE·importance
      + WEIGHT_RELEVANCE·relevance + WEIGHT_STANCE_CONGRUENCE·congruence
```

Weights `0.25` each (`config.py`). Recency decays exponentially; importance is LLM-rated 1–10; relevance is inverted cosine distance. **Stance congruence**: each memory is tagged with the stance it expressed (`+1`, `−1`, or `0` when ambiguous); a retrieval anchored to a stance (`respond()`: the softmax draw; `reflect()`: the argmax) scores congruent memories 1, opposing 0, neutral 0.5. That implements motivated recall without hard-filtering opposing memories; `WEIGHT_STANCE_CONGRUENCE` sets its strength, and `0` restores stance-agnostic retrieval (rebalance the other weights).

`MEMORY_ENABLED = False` (Stage 4's ablation) returns no memories and skips reflection; writing continues.

---

## Persona pool

`data/gss_personas.json` comes from `data/GSS2024.dta` (2024 GSS Release 3, 3,986 respondents; both git-ignored) via `notebooks/data_processing.ipynb`. The export keeps an **allowlist** of 91 variables (`INCLUDE_VARIABLES`) and drops the rest of the ~1,000-variable release. `agents/personas.py` then has the LLM invent a first name and a 2–3 sentence second-person persona covering life situation and personality, without political opinions.

**The line is attitude vs. fact.** Explicit attitude and opinion items are excluded whatever their topic, so no stance leaks into a persona; demographic and background facts are kept, including topically adjacent ones (citizenship, birthplace, migration history). Two data corrections: `CONINC`/`CONRINC` (continuous, constant-dollar) replace the top-coded `INCOME`/`RINCOME` brackets, and unanswered fields are dropped per record (ballot splitting leaves 17–86 of 91 fields populated, mean 65.3).

### Retained variables (91)

| Category | Variables |
|---|---|
| **Demographics** | Age, sex, sex at birth, race (plain & detailed), Hispanic origin, ethnic ancestry, siblings, marital status & type, widowed/divorced, number & age at first child, region, place size, US citizenship, born in the US, age moved to current residence, migration history, non-English language spoken at home & fluency |
| **Education** | Years of school, highest degree, college major |
| **Employment** | Employment status, ever worked, self-employed/family business, government employment, industry, occupation, employer size (employees & workplace size), hours & weeks worked, full/part-time, occupational prestige score, socioeconomic index score, union membership, job satisfaction, supervises others |
| **Income & class** | Household & personal income (continuous, constant dollars), subjective social class, financial satisfaction, financial situation vs. average family, standard of living vs. parents/children, chance to improve standard of living |
| **Household** | Household size, number of earners, dwelling type, home ownership, generations in household |
| **Family / spouse** | Spouse's education, degree, employment status, occupation, industry, prestige & SEI scores; marital happiness |
| **Parental background** | Father's & mother's education, occupation, prestige & SEI scores; family situation and relative income growing up; parents born in the US; grandparents born abroad; region & place type at 16; residential mobility since 16 |
| **Religiosity** | Religious preference (current & raised in), denomination, service attendance, prayer frequency, self-rated religiosity |
| **Values** | General happiness, belief about getting ahead in life, work centrality, value placed on hard work in raising children |
| **Lifestyle** | TV hours per day |
| **Moral item** | Approval of spanking children |

### Excluded field categories

- **Immigration/asylum attitudes**: `LETIN1A`, `LETIN1`, `IMMCRIME(1)`, `IMMJOBS(1)`, `IMMFATE`, `LETINHSP`, `LETINASN`, `ADOPTUS`, `IMMAMECO`, `IMMASSIM`, `IMMWLFARE`, `TOPPROB2_IMMG`
- **Race/civil-rights attitudes**: `RACOPEN*`, `RACLIVE`, `RACDIF1-5`, `WRKWAYUP`, `WLTHWHTS/BLKS/HSPS`, `WORKWHTS/BLKS/HSPS`, `MARWHT/BLK/ASIAN/HISP`, `LIVEBLKS/WHTS`, `AFFRMACT`, `DISCAFF*`, `HELPBLK`, `DSNDNTPAY`
- **Gender-role/political attitudes**: `FECHLD`, `FEPRESCH`, `FEFAM`, `FEPOL*`, `MEOVRWRK`, `FEOVRWRK`, `FEHIRE`, `FEJOBAFF`
- **Party, vote, political views and national spending priorities**: `PARTYID`, `POLVIEWS`, `VOTE16`, `VOTE20`, `PRES16`, `PRES20`, `IF16WHO`, `IF20WHO`, all `NAT*` items (including energy and environment spending)
- **Institutional confidence/trust**: all `CON*`, `TRCONG`, `TRCOURTS`, `TRPPL`, `TRUST*`, `CANTRUST`, `FAIR*`, `HELPFUL*`
- **Civil-liberties (Stouffer) items**: `SPK*`/`COL*`/`LIB*` for ATH, RAC, COM, MSLM
- **Abortion**: all `AB*`
- **Government intervention/tax**: `TAX`, `EQWLTH`, `HELPPOOR`, `HELPNOT`, `HELPSICK` (`PARSOL`/`KIDSSOL`/`GOODLIFE` are retained: personal mobility, not policy)
- **Guns, crime, punishment**: `OWNGUN`, `ROWNGUN`, `PISTOL`, `RIFLE`, `SHOTGUN`, `HUNT*`, `GUNLAW`, `CAPPUN`, `GRASS`, `COURTS*`, `POL(HITOK|ABUSE|MURDR|ESCAP|ATTAK)`, `FEAR`
- **Sex/marriage politics**: `SEXEDUC`, `TEENSEX`, `PREMARSX`, `PILLOK`, `MARSAME*`, `HOMOSEX`, `XMARSEX`, `DIVLAW*`, `PRAYER*`, `CHNGDOCS`, `ACPTTG`, `RELIGINF`, `SUICIDE*`, `LETDIE1`, `XMOVIE*`, `PORNLAW`
- **Whole modules**: ANES; ISSP Digital Societies & National Identity; Mental Health Stigma vignettes; High Risk Behaviors; GSS Next Follow-on
- **Technical, paradata, roster and interviewer variables**: IDs, weights, `SAMPLE`, `YEAR`, `DATEINTV`, `BALLOT`, `FORM`, `INT*`, `RLOOKS`, `RGROOMED`, `RESPOND`, `CONSENT`, `MODE`, `FEEUSED`, `SPANENG`, `VPSU`, `VSTRAT`, `KISH`, `DEVTYPE`, `WHOELSE*`, `RELATE/GENDER/OLD/MAR/AWAY/WHERE1-14`
- **Verbal-ability test**: `WORDA`–`WORDSUM*`

---

## Network mode

### Round structure

```
For each round:
  INTERACTIONS_PER_ROUND interactions, each as in "Interaction loop" above;
  a rewire happens inside the loop, so later draws see a consistent graph.
  Then: reflection (every REFLECT_EVERY rounds, MEMORY_ENABLED) and a snapshot.
```

### Initial topology, Stochastic Block Model

A **fixed-edge-count two-block model**: `SBM_M_INTRA` of each block's `C(10,2) = 45` pairs (27, density exactly 0.6) and `SBM_M_INTER` of the `10 × 10 = 100` cross pairs, each layer a uniform draw from its own pair set, within-block layer first. Counts rather than probabilities because `NUM_AGENTS` is fixed at 20: under a probability a coupling level is a mixture over topology sizes (at `p = 0.02`, 13.3 % of draws have no cross edge at all). Drawing the within-block layer first gives one seed the same within-block graph at every coupling. `SBM_P_INTRA`/`SBM_P_INTER` survive as derived densities (`m/45`, `m/100`).

`SBM_M_INTER` is the primary experimental variable: low coupling sustains community-aligned polarization, high coupling drives consensus. `m = 0` (two disconnected blocks) is legal; the loop only draws expressers that have a neighbour. Nodes carry a `"community"` attribute, written into every snapshot as a `"communities"` map beside `n_intra_edges`/`n_inter_edges`, the basis of per-community regime classification.

### Graph dynamics

| `GRAPH_DYNAMIC` | Behaviour |
|---|---|
| `False` (default) | Graph fixed; `record_reward()` still runs on measured-reward rounds (it feeds `select_responder`) |
| `True` | `update_edge()` steps valuations; a severed edge is **replaced in the same step** by `rewire_after_drop()`, so `|E|` is conserved. Isolates are not reconnected |

### Output

```
logs/run_<id>/
├-- params.json        resolved parameters + git_commit, model_digest, ollama_version (logger.log_params)
├-- personas.json      agent names and personas
├-- events.jsonl       one record per event
└-- network_rounds/round_NNNN.json   round_0000 is the baseline before round 1
```

| `events.jsonl` `type` | Key fields |
|---|---|
| `discussion` | `round`, `agent_a` (expresser), `agent_b`, `topic_label`, `opener_topic_first`, `turns` (each with its `gate` record), `reward_a`, `reward_b`, `softmax_a/b`, `cue_a/b`, `gate_attempts_a/b`, `gate_fallback_a/b`, `gate_final_expressed_a/b` (Q-update index), `gate_final_graded_a/b`, `argmax_a/b` (pre-update) |
| `edge_maintained` | `round`, `agent_a`, `agent_b` |
| `edge_dropped` | `agent_a` = the **initiator**, `agent_b`, `signal_severed`/`signal_other`, `strength_severed`/`strength_other` |
| `edge_rewired` | `agent_a` (initiator), `agent_b` (new partner), `replaced`, `rule`; counted per round, the **realised** relocation rate |
| `rewire_failed` | `agent_a`, `agent_b`, `rule`: empty pool, the degree drop stands (only under `reward_local`) |
| `reflection` | `round`, `agent` |

A snapshot holds `round`, `elapsed_s`, `nodes`, `edges` (per-agent `strengths`, `rounds_active`), `metrics` (`density`, `n_components`, `avg_degree`, `n_edges`, `n_intra_edges`, `n_inter_edges`, `n_pos`, `n_neg`, `dispersion`, `mean_q_gap`), `communities`, and `opinion_states`: per agent `q_pos`, `q_neg`, `argmax`, `q_gap`, and `softmax` if it expressed that round. `n_pos`/`n_neg` count strict Q-gap signs, so tied agents are in neither.

---

## Pairwise mode

The network mechanism for two agents on one edge, `INTERACTIONS_PER_ROUND = 2` (fixed locally), no communities. `select_responder()`, `record_reward()` and `update_edge()` run unchanged, including the no-signal freeze. A severance under `GRAPH_DYNAMIC` ends the run, since there is nobody to rewire to.

---

## Theoretical alignment

| Banisch & Olbrich (2019) | This simulation |
|---|---|
| Public opinion o_i ∈ {−1, +1} | The softmax draw, enforced at expression by the gate; `argmax_stance` for metrics |
| Reward r = o_i · o_j | `reward_from_expressions()`: the same product over graded stances in [−1, 1] |
| Q(o_i) ← (1−α)·Q(o_i) + α·r | `update_q_value()`, expresser only, indexed on `gate_final_expressed`; skipped on no-signal rounds |
| Asymmetric update | One expresser per interaction (Jacob & Banisch 2023) |
| Partner selection | `select_responder(β_sel)`: β_sel = 0 uniform; > 0 reward-weighted, an extension beyond Jacob & Banisch's conviction-similarity homophily |
| Network co-evolution | `GRAPH_DYNAMIC = True`: experience-based partner choice that never reads Q-values |
| Phase transition via modularity | the `SBM_M_INTER` ladder |
| Structural polarization | `dispersion`, `n_pos`/`n_neg` |

### Where the extension lives when the scalars look canonical

Both LLM interfaces sit close to B&O by construction: the gate makes expressed equal drawn, and gated turns concentrate at ±1. Their *product* does not: a large share of applied rewards lies strictly between the poles, most of all in the early rounds, drifting toward the canon as opinions harden (pilot measurements: [`../analysis/design_analysis.ipynb`](../analysis/design_analysis.ipynb) § 2). The contribution therefore sits in the interfaces around dynamics that are deliberately canonical:

1. **A graded opinion product.** The canon's `r = o_i · o_j` over actually generated turns, recovering the binary product at ±1 and adding conviction and hedging to the feedback channel.
2. **Asymmetries the canon cannot generate.** The model resists one pole more than the other at a rate measurable from the gate records, a drift source from the language substrate itself.
3. **Expression friction as a measured construct.** B&O's expression is frictionless; the gate's rejected drafts measure the gap between holding and voicing a stance, per agent, direction and round.
4. **No-signal states.** Abstentions and unscorable reactions freeze a round; neutral feedback (`0.0`, applied) stays distinct from absent feedback (`None`).
5. **A discourse corpus under the dynamics.** Every scalar traces to text, enabling argument-level analyses (e.g. whether stage-3 template turns cluster in one camp).
6. **Non-stationary expression.** Memory, reflection and the cue make generation state-dependent, so expression friction can drift within a run.

### Design decisions

- **Neutral initialisation.** Q starts at (0, 0) rather than at an LLM-elicited opinion, which would inject model bias (Chuang et al. 2024) at step 0.
- **Inverse temperature β.** `p(+1) = σ(β·Δq)` as in B&O; a tie is a fair coin at every β, so agents split 50/50 at start without jitter.
- **Asymmetric, one-exchange interactions** (Jacob & Banisch 2023): agents learn from the reaction to what *they* said; `last_message_to[(speaker, listener)]` lets repeat meetings continue the dialogue.
- **Reward-based partner choice** (`β_sel`): expressers return to partners who rewarded them, which can build echo chambers through reinforcement without assuming similarity. `select_responder` stays a standalone unit so a multi-platform neighbour set (Jacob & Banisch 2023) slots in without touching the loop.
- **Two-block topology** rather than Watts-Strogatz: it gives a clean coupling knob for the phase transition.
- **Sequential by construction.** Every interaction mutates state the next draw reads, so a run cannot be parallelised internally without becoming a synchronous-update model. Parallelism is across runs: one job per sub-stage, many jobs per node (RUNBOOK § Kurzfassung). Reflection is the only safely parallel phase within a run; it is not parallelised.

---

## Requirements

- Python 3.10+
- Ollama 0.34.0 (pinned for the confirmatory schedule; `qwen3.8` needs ≥ 0.32.12) at `127.0.0.1:11500`, the `OLLAMA_HOST` default; batch jobs use a per-job port
- Models: `qwen3.8-nodraft:27b`, plus `devstral-small-2:24b` and `gemma4-nodraft:26b` for Bracket E. The `-nodraft` tags are the pulled `qwen3.8:27b` / `gemma4:26b` with `PARAMETER draft_num_predict 0`: Ollama 0.34.0's automatic speculative decoding crashes the llama-server on load. Speculative decoding preserves the sampled distribution, so disabling it costs speed only (RUNBOOK.md § Modelle pullen, step 3)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main_network.py   # network simulation
python main_pairwise.py  # two-agent mode
```

For a smoke test set `NUM_AGENTS = 4` and `NETWORK_MAX_ROUNDS = 3` in `config.py`. The simulation needs a GPU Ollama server; on the SCC it runs through `run_stage.sh` ([RUNBOOK.md](RUNBOOK.md)).

## Parameter sweeps (`run_sweep.py`)

```bash
python run_sweep.py --dry-run      # print the grid
python run_sweep.py --workers 3    # up to 3 grid points in parallel
```

`PARAM_GRID` maps `SIM_*` names to value lists; each combination runs `main_network.py` as its own subprocess with a tagged log directory, and `logs/sweeps/<ts>/manifest.json` records the grid. Every grid name is checked against `SUPPORTED_OVERRIDES`, the names `config.py` reads, and an unknown name **aborts**: an ignored override produces a complete, plausible run at the wrong parameters.

---

## Configuration (`config.py`)

Every tunable lives in `config.py`, with no magic numbers elsewhere. The names in `run_sweep.SUPPORTED_OVERRIDES` are read as `SIM_<NAME>`, which is how `run_stage.sh` gives each queued job its own physics. **No override, on purpose:** `NUM_AGENTS` and `NETWORK_MAX_ROUNDS` (fixed for the whole confirmatory schedule, edited before submitting), `STRENGTH_DELTA` (derived) and `INITIAL_EDGE_STRENGTH` (the unit of the valuation axis). What a run actually used is its `params.json`.

### Shared (both modes)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3.8-nodraft:27b` | Agent, generator and grader at once. `SIM_LLM_MODEL` lets Stage 8 queue beside default-model jobs |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | `sentence-transformers` on CPU |
| `reasoning` | `False` | `OllamaLLM` argument in both entry points. Thinking mode would prepend `<think>` blocks that break `_score_importance()`'s float parse (silent fallback 5.0); the classifier is schema-constrained and robust, but thinking still costs time |
| `NETWORK_MAX_ROUNDS` | `100` | The horizon. Set for the confirmatory schedule by screen `A-00` (the pilot ran 25 and 100) |
| `MEMORY_ENABLED` | `True` | `False` = Stage 4's memory ablation (`SIM_MEMORY_ENABLED`) |
| `REFLECT_EVERY` | `5` | Reflection frequency (rounds) |
| `MAX_MEMORIES_SEED` | `15` | Recent memories fed to reflection |
| `MAX_MEMORIES_RETRIEVE` | `5` | Memories retrieved per call |
| `MEMORY_PERSIST` | `False` | Persist ChromaDB to disk |
| `WEIGHT_RECENCY` / `_IMPORTANCE` / `_RELEVANCE` / `_STANCE_CONGRUENCE` | `0.25` each | Retrieval weights ([Memory system](#memory-system)); congruence is the motivated-recall knob |
| `INTERACTIONS_PER_ROUND` | `= NUM_AGENTS` | Network mode only |

### SFT Q-learning

| Setting | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `0.1` | α |
| `OPINION_BETA` | `5.0` | β. Every confirmatory arm except Stage 2 runs `inf` (deterministic argmax) via `run_stage.sh` |
| `RESPONDER_SELECTION_BETA` | `0.0` | β_sel; 0 = uniform |
| `CONVICTION_CUE_ENABLED` | `True` | [The conviction cue](#the-conviction-cue); `SIM_CONVICTION_CUE=0` is the ablation |
| `STANCE_BRIDGE_ENABLED` | `False` | `True` molds persona and stance into cached per-stance justifications (`SIM_STANCE_BRIDGE=1`) |
| `SFT_GATE_ENABLED` | `True` | `False` = one ungated draft, the ablation (no override) |
| `SFT_GATE_MAX_ATTEMPTS` | `5` | Attempt budget per cascade stage |
| `SEED` | OS entropy | Seeds the persona records drawn (not their LLM-expanded texts) and the topology only (`SIM_SEED`; `run_stage.sh` takes a `SEED_BLOCK` instead) |

At finite β, `p(argmax) = σ(β·|Δq|)`: a gap of 0.4 gives 88 % at β = 5 and 60–69 % at β = 1–2.

### Network topology

| Setting | Default | Description |
|---|---|---|
| `NUM_AGENTS` | `20` | |
| `SBM_NUM_COMMUNITIES` | `2` | Required by the count parameterisation |
| `SBM_M_INTRA` | `27` | Within-block edges per block, of 45 |
| `SBM_M_INTER` | `15` | Cross-block edges, of 100: **the phase-transition axis**. Queued jobs take it from `run_stage.sh` |
| `SBM_P_INTRA` / `SBM_P_INTER` | derived | Densities; `SIM_SBM_P_*` raises at import |

### Graph dynamics (`GRAPH_DYNAMIC = True` only)

| Setting | Default | Description |
|---|---|---|
| `GRAPH_DYNAMIC` | `False` | Endogenous rewiring |
| `REWIRE_RULE` | `"reward_local"` | **RQ2 factor**: `"random"` (any non-neighbour, the churn-only null), `"reward_local"` (2-hop only; every new tie closes a triangle), `"off"` (sever without replacement) |
| `REWIRE_BETA` | `5.0` | Partner-choice decisiveness, `exp(REWIRE_BETA · referral estimate)`; watch `std_degree` / `assort_degree` for hub collapse |
| `REWIRE_TIMESCALE_RATIO` | `3.0` | **w**, see below. Every dynamic sub-stage takes it from screen `D-01` (rule R2); `7-03` pins 1.0 |
| `STRENGTH_CAP` | `3.0` | Valuation ceiling (`SIM_STRENGTH_CAP`, screening only) |
| `STRENGTH_FLOOR` | `0.0` | Severance threshold |
| `REWARD_WINDOW_M` | `5` | Rolling reward window per edge; also read by `select_responder` at β_sel > 0 (`SIM_REWARD_WINDOW_M`, screening only) |
| `INITIAL_EDGE_STRENGTH` | `1.0` | Founding-edge valuation and the unit of the axis: the dynamics depend only on `CAP / INITIAL` and `NEW / INITIAL` |
| `NEW_EDGE_STRENGTH` | `0.5` | Valuation of a rewired tie: less robust than an established one, and safe from instant re-cutting because its empty history reads 0.0 (`SIM_NEW_EDGE_STRENGTH`, screening only) |
| `STRENGTH_DELTA` | derived, `0.3` | `w · α · INITIAL_EDGE_STRENGTH`; echoed at startup |

**Timescale separation.** Coevolution outcomes depend on the ratio of opinion to rewiring rates, and with `|E|` conserved the strength parameters only set how often a relocation fires, so they collapse into one dial:

```
τ_opinion = 1 / LEARNING_RATE,  τ_rewire = INITIAL_EDGE_STRENGTH / STRENGTH_DELTA
w = τ_opinion / τ_rewire
```

`w ≪ 1` recovers the static graph continuously, as a limit; `w ≫ 1` lets structure outrun opinion (fragmentation). The pilot's null screens (`0c-04`, `0c-09`, [`../analysis/test_schedule.md`](../analysis/test_schedule.md) § Stage 0c) found the static baseline at `w ≤ 0.5` and the sorting/mixing turnover near `w ≈ 1.2`; the confirmatory value comes from `D-01`. `w` is **nominal**: it assumes `|mean reward| = 1`, ignores the cap, and an edge of a degree-k agent is touched only every ~k expresser turns, so report the realised rate (`edge_rewired` per round) beside it. The LLM's own conformity timescale (~1–3 rounds) cannot enter `w`; locate it from Bracket A before submitting the `w = 1.0` probe `7-03`.

### Topic

| Setting | Default | Description |
|---|---|---|
| `TOPICS` | registry | `nuclear`, `renewables`, `coffee_tea`: each a pairwise comparison with `text` (+1), `opposite` (−1, the comparison reversed) and two opener wordings (rationale: [`topic_candidates.md`](topic_candidates.md)) |
| `TOPIC_KEY` | `"nuclear"` | `SIM_TOPIC`; Stage 5 selects the other two |
| `TOPIC_TEXT` | nuclear power does more good than harm | The +1 pole and the classifier's answer A. `_stance_pole()` maps −1 to `TOPIC_TEXT_OPPOSITE`, so no prompt states a stance as opposition |
| `TOPIC_TEXT_OPPOSITE` | nuclear power does more harm than good | The −1 pole, answer B. **Derived from `TOPIC_TEXT`**, so Stage 3's flip (the opposite as `SIM_TOPIC_TEXT`) swaps both answers |
| `TOPIC_OPENER` / `_SWAPPED` | the entry's wordings | Drawn per exchange (`draw_opener`), logged as `opener_topic_first`. Never shown to the classifier, and read only by the expresser on a first meeting |

One topic per run keeps the ±1 dimension on one opinion object. Selection criteria (crisp binary, low model lean, guardrail-safe, contested, flippable) are argued in `topic_candidates.md`.

---

## Conventions

- **Language**: English prompts and outputs (US context); code, comments and logs in English.
- **Asymmetric update**: only the expresser's Q moves; do not add a symmetric update without revisiting the theoretical alignment.
- **Q-update index**: `update_q_value()` takes the stance explicitly, never the argmax; the main loops pass `gate_final_expressed`. Do not reintroduce a separate expressed-stance channel.
- **Edge severance**: when either valuation reaches `STRENGTH_FLOOR`; `update_edge()` returns the initiator (lower valuation if both cross, ties to the expresser), and only it rewires. A no-signal round cannot sever.
- **Partner choice never reads opinion state**, 1-hop or 2-hop: scoring on opinion similarity would reinstate the homophily SFT derives around.
- **Logging**: pass SFT metrics via `extra_metrics` and Q-states via `opinion_states` to `snapshot_network()`.

### What not to change without good reason

- **The classifier prompt** (`graded_stance_prompt()`): minimal, single-text and absolute. Adding context or making it relational re-couples expression and evaluation. The two-answer form keeps favor/oppose vocabulary out; the audit imports this function rather than copying it.
- **`reasoning=False`** in both `OllamaLLM` constructors.
- **The position-taking instruction** in `respond()`.
- **The gate's pass rule**: abstentions (`BALANCED`/`NEITHER`) pass; a frame break is kept out by the rubric (`UNRELATED`), not by the gate second-guessing a `0`.
- **Positive pole anchoring** (`_stance_pole()`): a "you oppose X" anchor yields drafts scored `NEITHER` on the −1 side only, a stance-dependent no-signal rate.
- **The soft generation prompt**: the gate enforces output-side precisely so the first draft keeps sampling the persona's unconstrained pull.
- **`LEARNING_RATE` and `OPINION_BETA`** together set the convergence rate; change them together and rerun the phase-transition baseline.
- **The plain-Python loop.** The simulation was first built on LangGraph, whose compile-time graph fits neither stochastic expresser/responder draws nor a rewired topology; the explicit loop keeps the sequential-update semantics visible. `langchain-ollama` stays as the LLM client.

---

## Extension: endogenous tie formation

With `GRAPH_DYNAMIC = True`, each measured-reward round steps the expresser's valuation by its rolling mean reward on that edge. When either valuation reaches the floor the edge is severed, and the **initiator** opens a replacement tie in the same step (`rewire_after_drop()`).

**Why cutting became relocating.** Delete-only dynamics thin the graph monotonically, and falling density produces fragmentation on its own, confounding the structural measurement. Relocation fixes `|E|` at no LLM cost and keeps dynamic runs comparable to static ones. Only the initiator keeps its degree, so the degree *distribution* moves while `|E|` and mean degree stay flat, which is a correctness check, never a finding. The one leak is `rewire_failed` (empty 2-hop pool, `reward_local` only): count it per run.

**No reconnection.** An agent left at degree zero sits out until another agent's rewire picks it. Under `"random"` that remains possible; under `"reward_local"` it cannot happen, because an isolate has no neighbours to be reached through. Reconnecting would mask the attrition the extension measures; it shows in `n_components` and `avg_degree`.

### Experience-based, not perception-based partner choice

Attaching to whoever looks similar is the homophily assumption SFT was built to derive around, and would reduce RQ2 to adaptive bounded confidence. Instead a non-neighbour is valued by **reward chained through mutual contacts**:

```
E[r_ij] = mean over m in N(i) ∩ N(j) of  mean_reward(i<-m) * mean_reward(m<-j)
```

Both factors are already recorded for `select_responder`, so this adds no state and no LLM call. Since reward is a product of graded stances, the chain is ≈ `g_i · g_j · g_m²`: the intermediary only shrinks the estimate (an ambivalent referrer transmits nothing) and never flips its sign. Two negative channels multiply to a positive estimate, so structural balance (Heider) emerges from the reward primitive.

### Sorting is not clustering: isolating reach from scoring

Attribute-based attachment **sorts** agents, but dense community structure comes from **closing triangles**; a sorting-only rule yields assortative but tree-like structure and low modularity whatever the population size, the mechanistic reading of the modularity shortfall reported in recent LLM-ABM work. Because the referral estimate needs a mutual contact, closure is forced by the information channel. `REWIRE_RULE` makes that a controlled comparison:

| Arm | Pool | Scoring | Isolates |
|---|---|---|---|
| `"random"` | any non-neighbour | none | churn alone (the null) |
| `"reward_local"`, `REWIRE_BETA = 0` | 2-hop | none | **reach alone** |
| `"reward_local"`, `REWIRE_BETA > 0` | 2-hop | referral | reach plus selection |

The contrast `"random"` vs `"reward_local"` at `REWIRE_BETA = 0` differs only in closure. An empty 2-hop pool returns `None` rather than falling back to a global draw, which would leak exactly the reach the arm withholds.

Primary structural DVs (`analysis/metrics.structure_trace`): `clustering`, `tie_ratio` / `n_inter_edges` and the round it reaches 0, `std_degree` / `assort_degree`. Failure modes to watch: hub collapse at decisive `REWIRE_BETA`; churn before a tie accumulates history; the graph crystallising before Q diverges (the high-`w` end, a result rather than a bug); thin early referral evidence, so early structure should not be over-read.

---

## Future research

**Continuous, gate-replacing conviction.** The cue is the banded, gate-compatible version of a dropped design in which `tanh(γ · max(0, stance × q_gap))` scaled how assertively the drawn stance was framed *instead of* enforcing it, giving a persona-to-SFT transition from one parameter γ. Its signature, first-attempt fidelity rising with conviction, is readable from cued runs now (`load_gate()` carries `cue` beside `first_expressed`).

**Weight-level social-feedback learning.** [`PLAN_weight_level_rl.md`](PLAN_weight_level_rl.md) rejects RLHF from social feedback as the main mechanism (it would dissolve the Q-state the interpretability claim rests on) and sketches periodic per-agent LoRA distillation from the same reward, with Q remaining the sole seat of opinion. Not implemented.

---

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76-103.

Chuang, Y.-S., et al. (2024). Simulating opinion dynamics with networks of LLM-based agents. *Findings of NAACL 2024*.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *Journal of Artificial Societies and Social Simulation, 26*(3), 11.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.

Sarközi, M., Jütersonke, S., Banisch, S., Poppe, S., & Berger, R. (2022). The effects of social feedback on private opinions. Empirical evidence from the laboratory. *PLOS ONE, 17*(10), e0274903.
