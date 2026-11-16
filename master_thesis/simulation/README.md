# Simulation

LLM agents on the canon's two-community network (Banisch & Olbrich 2019), run on local models via [Ollama](https://ollama.com). Each agent has a name and a memory, nothing else; no theory of influence is built in. In an exchange, two neighbours each declare a pole in private, each answer one moderator question unheard by the other, and store both answers as text. They also read back the pole they took and the one they heard; these two reads measure the agreement each experienced, which no prompt shows. In one condition a declaration shows both turns of every stored exchange, in the other the agent's own turn alone. Feedback reaches an agent only through memory, and memory only its declaration: every later call in an exchange is blind to it.

What the runs are read against, and how: [`../analysis/README.md`](../analysis/README.md); what runs when: `../analysis/schedule.md`; how on the SCC: `RUNBOOK.md`. The SFT-learning-agent version (personas, the expression gate, the graded rater, retrieval, the edge layer) is preserved on branch `sft-loop` (§ Removed).

## Terminology

| Term | Definition |
|---|---|
| **Turn** | one speaker's message, one draft, unchecked |
| **Exchange** | agent i and agent j, from their declarations to their stored entries; the two roles differ only in call order |
| **Round** | `EXCHANGES_PER_ROUND` exchanges, then the snapshot |
| **Declared position** y | what an agent's declaration answers, from memory; the canon's opinion |
| **Argued position** s | what its turn argues: the declared one unless explored |
| **Self-read position** ŝ | its own turn's position as its blind self-read reads it; measured, never shown |
| **Perceived position** ô | the partner's position as the agent's probe reads it from the partner's turn; measured, never shown |
| **Condition** | what a declaration shows of each entry: `full` both turns, `own` the agent's own turn alone (§ Memory) |
| **Explored** | a turn whose argued position exploration drew; it is an *explored entry* only where ŝ lies on the pole argued (§ Self-read) |

Schedule terminology (stage / sub-stage / run) is defined in `../CLAUDE.md`.

## Design

### Positions

A position is a pole x ∈ {+1, −1}, the canon's binary opinion (Banisch & Olbrich 2019); there is no strength and no *neutral*.
- **The declared pole is the canon's opinion,** argmax Q. Its expression is the turn, which argues that pole unless exploration flips it; the turn's text is stored, and ŝ measures what it carried (§ The exchange, § Self-read).
- **One schema** for declarations, probes and self-reads: `{position: "A"|"B"}`, a forced choice between the two poles, as in Schubert et al.'s (2024) two-armed tasks. `_SLOT_RNG` draws per call which pole is listed first (`topic_first`).
- **The wording,** in every prompt and list: the pole's sentence as it stands (`agents.prompts.wording`), with no stance verb around it. Each pole is a claim of its own, never stated as the other's negation.
- **Why two:** every self-read and probe lands on a pole, so r̂ (§ The exchange) is the canon's r = ±1 exactly and no entry earns 0. The five-position design (a strength as the canon's conviction, and *neutral*) is on branch `three-models` (§ Removed).

### Initialization

Before round 1 the global stream draws, in this order:
1. **Names:** placed on the nodes from `agents/names.py`, the 2020 Census's ten most common male and ten most common female first names (U.S. Census Bureau 2026): Michael, John, James, David, Robert, William, Daniel, Joseph, Richard, Thomas; Mary, Maria, Jennifer, Elizabeth, Patricia, Linda, Susan, Jessica, Barbara, Karen. At N = 20 every run holds all twenty; only their placement is drawn. A name carries gender and ethnicity cues an LLM may act on, so the audit reports the declared prior by name and every model fits an intercept per name ([`../analysis/README.md`](../analysis/README.md#the-models) § The models).
2. **Starting positions,** by default (`START` = `polarized`) the canon's polarized start (Banisch & Olbrich 2019, § 5.4): community 0 at +1, community 1 at −1, read off the graph after step 3 and so drawing nothing here. Pilot phase 1 sets `START` = `coin`, the canon's start of § 4: each Q drawn uniformly, the opinion argmax Q (uniform on [0, 1] in the text, [−0.5, 0.5] in the code, the same ΔQ), so pole +1 with `P_START_PLUS` = ½ (`schedule.md` § Pilot). Either is logged as round 0, marked `drawn`.
3. **The graph** (§ Topology).

The starting position stands as the agent's declaration in its first exchange, whose turn argues it; from its second exchange on, the agent declares from memory. An explored first exchange argues the opposite pole (§ The exchange).

### Topology

The canon's two communities (Banisch & Olbrich 2019, § 5.4): N = 20 in two blocks of 10, a cross pair linked with probability p_inter = `SBM_P_INTER` (`SIM_SBM_P_INTER`), a within pair with 1 − p_inter. Every pair draws one uniform in a fixed order at every p_inter, so one seed's graphs nest (`schedule.md` § Set-up). The graph is static: there is no edge layer.

A round is N exchanges, then the snapshot. After round H every agent declares once more, logged as round H + 1. N = 20 is set, not derived; H = 100 is fixed by design, for the pilot and the runs alike (`schedule.md` § Runs, Horizon).

### The exchange

1. **Pairing.** Agent i is drawn uniformly among agents with degree > 0, agent j uniformly among i's neighbours (`network.matching.draw_pair`).
2. **Declarations** (`Agent.declare`), one each, partner-blind: both declare before either reads the other, as in the canon. The prompt holds the identity line, the setting (`agents.prompts.SETTING`, § Memory), the memory and "Which position do you take in the next conversation?"; it is the only call that shows the setting or memory. A declaration is never redrawn and never stored.
3. **Exploration** (`network.opinion.explore`): each side independently, with probability ε = 0.1 (`EXPLORATION_RATE`), argues the opposite pole; otherwise it argues its declaration. This is the canon's ε-greedy expression of the disfavoured opinion at the canon's rate (Banisch & Olbrich 2019), on both sides as in Meylahn & Meylahn's (2024) symmetric variant, since both learn. Through explored entries (§ Self-read) it keeps both own poles in a settled agent's memory, the other arm's outcomes in Schubert et al.'s (2024) terms. The declaration stays the observation, and nothing tells the agent it explored. The draws come from the global stream, like the pairing, so a seed fixes them whatever the texts; at ε = 0 none is made.
4. **Turns** (`Agent.speak`), i's then j's, partner-blind: identity line, the prompt message, then the argued position. It is the only call that shows the argued position. Both prompt messages are the moderator's opener, drawn per exchange by `draw_opener()`; neither turn shows the other or names the partner (Why partner-blind turns, below). Piao et al.'s (2026) communication stage generates the message from both agents' opinions and the reasons of a preceding self-expression stage; here the turn shows the speaker's own position alone, and there is no reasons stage (§ Self-read). A turn may restate its position's wording, so a probe can read a label as well as arguments.
5. **Self-reads and probes.** Each speaker reads its own turn blind, ŝ (`Agent.self_read`, § Self-read); then each probes the partner's (`Agent.perceive`), shown the partner's turn: "Which position did <partner> take in this message?". i's answer is ô_j, j's is ô_i.
6. **Rewards,** each side's from its own self-read ŝ and probe ô: the canon's r = o_i · o_j, both opinions as expressed and as the agent reads them. r̂ = ŝ · ô, ±1 (`network.opinion.perceived_reward`), is logged (`r_hat`) for the models; no prompt shows it.
7. **Memory.** i and j each store one entry: both turns (§ Memory).

**What feeds back.** The two stored turns, which only declarations read. No measurement does: ŝ, ô and r̂ reach no prompt, nor do s and declarations.

`network.exchange.run_exchange` runs all of it, for a run and for the exchange audit alike; with both argued positions given (`argued`) it skips steps 2–3, for the audit's assigned strata.

**Why partner-blind turns.** In the canon both opinions are fixed before either agent learns the other's (Banisch & Olbrich 2019, Fig. 1), and the partner acts only through the reward; in classical influence models, too, an agent expresses its held opinion, never one adapted to the partner within the encounter. LLM agents that answer each other do adapt: in dialogue, two agents converge mostly within their first round, the shift proportional to their difference and second in size only to the model's topic bias (Brockers et al. 2026), and agents from opposed sides moderate each other (Sakurai et al. 2025). A turn that answered its partner's would carry that drift into ŝ, and through it into habit and feedback alike, in every condition, and the surrogate would replay it even at β = 0. Where the opinion is read off the reply to a partner (Chuang et al. 2024), this drift is the opinion change; Piao et al. (2026) separate the update from the message, as here. With partner-blind turns the partner reaches an agent only through its stored turn, which only the declaration reads. The same holds for the reward: it comes from each side's reads of two turns written blind, never from a partner's reaction, which would carry the partner's adaptation to the agent's own turn into what the agent learns from. The cost: an exchange is two statements answering one question, not a dialogue, and arguments can travel only through memory ([`../analysis/README.md`](../analysis/README.md#the-study) § The study).

### Self-read

Every turn is one draft, unchecked. After both turns, each speaker reads its own turn blind (`Agent.self_read`): "You said: '<turn>'" and, after a blank line, "Which position did you take in this message?", under the position schema, with no memory and without naming s. Its answer ŝ is logged, and r̂ computed from it.

ŝ and ô are thus readings of what was said, ŝ of the agent's own turn and ô of the partner's, as the canon rewards the opinion expressed: the models read them, and no prompt shows them or their product. s, what the turn was asked to argue, sets the turn and the audit's strata and is logged; no prompt after the turn shows it. Where a text fails to carry s, memory holds what it did carry: an explored turn read back on the declared pole adds no explored entry. The audit measures ŝ against s (expression inconsistency) and ô against the partner's ŝ (misperception); the surrogate replays both, jointly per exchange.

**Why no self-regulation.** Piao et al. (2026, Methods M3) redraft a message until the speaker's yes/no self-check passes it and drop an agent at a retry limit. That check names the answer it asks about, so a lenient or pole-dependent yes keeps drafts a blind reader places elsewhere; and a cap voids exchanges by what the model will argue, which takes feedback away selectively by pole. The calibration at v2-freeze-7 showed both (its logs: git history, f6fdc183), and the cap of 10 was Piao et al.'s, not derived.

### Memory

Every entry stores both turns. `SIM_CONDITION` sets what a declaration shows of each (`memory.store.render`, the one place this table is applied), the same in either role:

| Condition | An entry, i's and j's alike |
|---|---|
| `full` (default) | `You said: '…'`, `<partner> said: '…'` |
| `own` | `You said: '…'` |

- **The setting, in both conditions:** the declaration opens with "You talk about <issue> with other people, one conversation at a time. In each conversation you state your position and your partner states theirs." (`agents.prompts.SETTING`). It states no rule, goal or reward, so any pull agreement has on a declaration is the agent's own, as the canon's agents value agreement without being told to. It sits in the declaration alone; the turns' system prompt does not hold it.
- **`own` is the negative control:** its prompt holds no partner's turn or name, only the setting's mention that there is one, so a partner term that still gains there is not the partner's ([`../analysis/README.md`](../analysis/README.md#hypotheses-and-tests) § Hypotheses and tests, Validity check).
- **Entries hold text only.** No condition states a position or a reward, marks an explored turn, or stores a declaration; ŝ, ô and r̂ are logged, for the models.
- **Fixed order:** `full` puts the own turn first in either role. Neither turn answers the other, so the order carries no turn-taking; it is not drawn, which would take a fifth generator and change no term the models read ([`../analysis/README.md`](../analysis/README.md#the-models) § The models).
- **Only the declaration reads memory.** Learning then sits in the one call the models fit, as in Schubert et al.'s (2024) in-context learning paradigm: one decision per trial, the history in its context, an environment outside the decision. This is the minimal generative-agent form (Ghaffarzadegan et al. 2024), memory as the decision's context rather than Park et al.'s (2023) memory stream. Turns, probes and self-reads depend only on the argued positions, the names and the opener's drawn wording, a fixed process the audit measures, so the window cannot shape what an agent says or hears. The cost: stored arguments reach an agent's declaration, never its turns, so they can act only there, unseparated from positions ([`../analysis/README.md`](../analysis/README.md#the-study) § The study).
- **The window is the whole run:** the declaration shows every entry the agent has stored, in the order stored (every earlier round plus the current round so far), as Schubert et al.'s (2024) prompts hold every trial of a block. Two entries per agent and round on average, about 200 at the final declaration; the busiest agent of the five-position pilot's phase 2 stored 299 (`analysis/logs/pilot/phase2/` on branch `three-models-logs`).
- **Why the whole run:** every outcome the agent has earned stays in view, as the canon's Q keeps every reward it was updated with, discounted but never dropped; the benchmark's canon scan likewise has no window. The cost is context: at the pilot's ~115–150 tokens per two-turn entry, the final declaration reaches ~46k tokens, hence `LLM_NUM_CTX` = 65,536 (§ LLM calls). Small models may weigh a long context unevenly; the runs' declarations log their `shown` ids, so how far back an effect reaches can be read off.
- **No retrieval or reflection** (Park et al. 2023). Scoring entries by recency, importance and relevance would choose what an agent reads, so part of what looks like learning could be selection (e.g. disagreements rated more important, hence shown more often). Reflections would summarize exactly what the terms model, in text no self-read or probe measures. Exchange entries are the only entries, shown unscored.
- With memory off (`SIM_MEMORY_ENABLED=0`), the declaration shows no window and nothing is stored.

### LLM calls

Every call shows the identity line `You are <name>.` and a system prompt, the turn's (`agents.prompts.TURN_SYSTEM`) or the position calls' (`ANSWER_SYSTEM`). Both open with a persona line after Piao et al.'s (2026, Supplementary Methods), `Assume you are someone who participates in a discussion.`, which names no topic, so no system prompt primes the subject. Only the declaration also shows the setting, whose `You talk about <issue>` names the topic's subject, never a pole (`TOPICS[…]["issue"]`, `topic_candidates.md`), and the memory. Every prompt text lives in [`agents/prompts.py`](agents/prompts.py).

| Call | Shows besides |
|---|---|
| Declaration | the setting, the memory and the question |
| Turn | the argued position and the moderator's opener |
| Self-read, after both turns | the speaker's own turn |
| Probe | the partner's turn |

- **The client** is ollama-python's, direct ([`agents/llm.py`](agents/llm.py)), with `think=False`. Position calls (declaration, probe, self-read) decode under the schema at min_p = 0, so no answer is cut, and log the logprobs of the answers among the top 20 tokens and of the drawn one, which only the audit's Reliability reads (`schedule.md` § Exchange audit). Ollama's `format` is enforced as a grammar, and min_p runs before the grammar check, so at the Modelfile's 0.05 a pole below 1/21 ≈ 4.8% would never be drawn; langchain-ollama's `OllamaLLM` silently dropped a per-call `min_p`, which is why the client is ollama-python's. The Modelfile's min_p 0.05 applies to turns.
- **`LLM_NUM_CTX` = 65,536,** for the whole-run window (§ Memory); an unused context costs only memory. Every call logs Ollama's `prompt_eval_count`, and the pilot checks each cell's largest, a declaration's, against it (`schedule.md` § Pilot).
- **Seeds:** every call samples under the next seed of `_SAMPLE_RNG` ([`agents/llm.py`](agents/llm.py)). A reply that fails to parse is asked again under the next seed, up to `PARSE_MAX_RETRIES` times (1 until the audit's rule reads it); if the last retry fails too, the run aborts (`ParseAbort`) and is replaced.
- **Checked per tag on the SCC** (`pull_models.sh check`, `RUNBOOK.md` § Pulling models): the top logprobs at the answer include both letters; a per-call `min_p` is applied; what a prompt longer than `num_ctx` does.

### What a run logs

Under `logs/run_<id>/` ([`network/logger.py`](network/logger.py)):
- **`params.json`:** every resolved parameter of § Parameters, with `git_commit`, `model_digest`, `ollama_version` and `node`, since no run reproduces bitwise.
- **`exchange` events** (`run_exchange`'s record verbatim):
  - every declaration, probe and self-read as x, with its logprobs and `topic_first`;
  - each side's `explored` and argued position s;
  - each turn's text;
  - both sides' r̂ (`r_hat`);
  - the ids of the entries shown at each declaration (`shown`) and of the two stored (`stored`);
  - the opener and its `opener_topic_first`;
  - every call's seed and `prompt_eval_count`.
- **`declaration` events:** the starting positions (round 0, `drawn`) and the final declarations with their shown ids (round H + 1).
- **`parse_retry` and `parse_abort` events.**
- **Snapshots** `network_rounds/round_NNNN.json`: each agent's latest declared position; round 0 adds the graph.

The run computes no metric; [`../analysis/metrics.py`](../analysis/metrics.py) reads the logs.

## Parameters

Every tunable lives in [`config.py`](config.py), whose comments say what each means. The names in `run_sweep.SUPPORTED_OVERRIDES` are read as `SIM_<NAME>`, which is how `run_stage.sh` gives each queued job its own physics; what a run used is its `params.json`.

| Name | Value | Override | Meaning |
|---|---|---|---|
| `NUM_AGENTS` | 20 | none, deliberately | N, two blocks of 10 |
| `NETWORK_MAX_ROUNDS` | 100 | none, deliberately | H, fixed by design for the pilot and the runs |
| `EXCHANGES_PER_ROUND` | N | | exchanges per round |
| `EXPLORATION_RATE` | 0.1 | | ε, per side |
| `START` | `polarized` | `SIM_START` | the start: `polarized` or `coin` (§ Initialization) |
| `P_START_PLUS` | 0.5 | | a coin start's probability of +1 |
| `SBM_P_INTER` | 0.15 | `SIM_SBM_P_INTER` | p_inter; within pairs at 1 − p_inter |
| `TOPIC_KEY` | `nuclear` | `SIM_TOPIC` | one of `TOPICS` (`topic_candidates.md`) |
| `LLM_MODEL` | `qwen3.5-nodraft:4b` | `SIM_LLM_MODEL` | the Ollama tag |
| `LLM_NUM_CTX` | 65536 | `LLM_NUM_CTX` (env) | context per call |
| `PARSE_MAX_RETRIES` | 1 | | re-asks of an unparseable schema reply |
| `MEMORY_ENABLED` | True | `SIM_MEMORY_ENABLED` | store and show entries |
| `CONDITION` | `full` | `SIM_CONDITION` | what a declaration shows of each entry: `full` or `own` (§ Memory); `SIM_FEEDBACK_MEMORY` is retired and aborts |
| `LEARNING_RATE` | 0.05 | | the canon's α; the benchmark's, and a fitted rate's comparison |
| `SEED` | OS entropy | `SIM_SEED` | the global stream and the three per-call streams |

`NUM_AGENTS` and `NETWORK_MAX_ROUNDS` have no override so that a sweep cannot silently run at a size `config.py` never read; `run_stage.sh` asserts both (`../CLAUDE.md`).

## Layout

```
config.py           Run; Initialization; Topology; Topic; Model; Memory;
                    SFT model (benchmark only); Reproducibility
main_network.py     builds a run, loops the rounds, logs
agents/             agent.py (declare, speak, self_read, perceive),
                    prompts.py (prompts, the setting, the position schema, _SLOT_RNG,
                    _OPENER_RNG),
                    llm.py (the direct client, _SAMPLE_RNG), names.py
memory/store.py     each agent's entries (both turns), render
network/            exchange.py (run_exchange), topology.py, state.py, matching.py,
                    opinion.py (explore, perceived_reward, the traces and ΔQ the fit
                    reads; the canon's Q for the benchmark), logger.py
run_stage.sh        SGE batch runner, one sub-stage per job (RUNBOOK.md)
check_substages.sh  resolves every sub-stage under DRY_RUN=1
run_sweep.py        local grid sweep
pull_models.sh      builds and checks the -nodraft tags on the SCC
environment/        where the SCC capture is filed; none yet
```

## Requirements and usage

- Python 3.12 (the SCC runs 3.12.2); `pip install -r requirements.txt` (ollama-python, networkx).
- Ollama 0.40.13 (each run logs its `ollama_version`) at `127.0.0.1:11500`, the `OLLAMA_HOST` default; batch jobs use a per-job port.
- Models: `qwen3.5-nodraft:4b`, `gemma4-nodraft:e4b` and `granite4.2-nodraft:3b`, the pulled `qwen3.5:4b` / `gemma4:e4b` / `granite4.2:3b` rebuilt with `PARAMETER draft_num_predict 0`, one common sampler (`pull_models.sh` `SAMPLER`: temperature 1, min_p 0.05 as the only cutoff, repetition penalties neutral) and no `SYSTEM`. Ollama 0.34.0's automatic speculative decoding crashed the llama-server on load for the retired 27b/26b tags; speculative decoding preserves the sampled distribution, so disabling it costs speed only (`RUNBOOK.md` § Pulling models).

```bash
python main_network.py              # one run; needs a GPU Ollama server
python run_sweep.py --dry-run       # print the grid, exercise the override guard
python run_sweep.py --workers 3     # up to 3 grid points in parallel
```

`run_sweep.py`'s `PARAM_GRID` maps `SIM_*` names to value lists; each combination runs `main_network.py` as its own subprocess with a tagged log directory, and `logs/sweeps/<ts>/manifest.json` records the grid. Every grid name is checked against `SUPPORTED_OVERRIDES`, and an unknown name **aborts**: an ignored override produces a complete, plausible run at the wrong parameters. On the SCC, runs go through `run_stage.sh` (`RUNBOOK.md`). Locally, only the exchange audit's `--stub` runs an exchange (`../CLAUDE.md` § Verifying a change).

## The benchmark must not move

[`benchmark_sft.py`](../analysis/benchmark/benchmark_sft.py) stays binary and imports what it shares with the runs: the graph (`network/topology.py`), the pairing (`network/matching.py`), `explore()` and the canon's Q (`network/opinion.py`) and config's SFT and topology constants. Its switches (the update rule, agent j's learning and exploration, the start and N; [`../analysis/README.md`](../analysis/README.md#the-references) § The references) default to the canon scan's settings (the canon's update, ε = `EXPLORATION_RATE`, agent i alone learning, the polarized start, N = 20), so with `--max-rounds 20000` the canon scan reproduces. Without `--both-learn`, exploration acts on agent i alone, as in the canon. After every change that touches one of these modules, run the self-test and compare a 20-seed slice with the canon scan on their shared columns (`../CLAUDE.md` § Verifying a change).

## Removed

Everything below stays on branch `sft-loop`:
- personas and their pools (`PERSONA_POOL`);
- the conviction cue, stance block, anchoring and bridges (`CONVICTION_CUE_ENABLED`, `STANCE_BRIDGE_ENABLED`);
- the expression gate, the graded rater and `reward_from_expressions` (`SFT_GATE_ENABLED`, `SFT_GATE_MAX_ATTEMPTS`);
- Park et al.'s retrieval: `memory/scoring.py`, importance (`_score_importance`), stance-congruent recall, ChromaDB and the `sentence-transformers` embeddings (`MAX_MEMORIES_RETRIEVE`, `RECENCY_DECAY`, `MEMORY_PERSIST`, `MEMORY_DIR`, the `WEIGHT_*` constants);
- reflection (`reflect()`, `all_recent()`, `REFLECT_EVERY`, `MAX_MEMORIES_SEED`);
- `respond()`, `run_discussion`, `main_pairwise.py`;
- the edge layer (`network/edges.py`, `GRAPH_DYNAMIC`, `RESPONDER_SELECTION_BETA` β_sel, the strength and rewiring constants) and the finite-β softmax (`OPINION_BETA`), with their overrides;
- the old exchange audit (`build_exchanges.py`, `rerate.py`), its logs and the `metrics.py` that read them.

Removed later, on `main` (git history): Piao et al.'s self-regulation, the yes/no self-check, its retries and cap (`SELF_REGULATION_MAX_TRIES`), voided exchanges and the audit's top-up phase (§ Self-read); j's turn answering i's, the pair's thread (`NetworkState.last_exchange`) and the audit's continued meetings (§ The exchange, Why partner-blind turns); the position lines in memory and the `stated` condition (§ Memory).

On branch `binary-reward`, from `three-models`: the five positions (*strongly*, *moderately*, *neutral*; § Positions) and the 15-round window (`MEMORY_WINDOW_ROUNDS`, `network.opinion.window_rounds`, the benchmark's `--window-rounds` and its windowed benchmark), replaced by two poles and the whole-run window (§ Memory). That design's logs (its audit with calibration, pilots, canon and windowed benchmarks, recovery) are committed on branch `three-models-logs`; read them through a `git worktree`, since checking that branch out here would delete the shared `canon_binary_*` on the way back.

`binary-reward` then replaced the `full` / `own` conditions by framed ones: `implicit` / `explicit` (the rule stated; the points in the frame and the entries, `Entry.points`), later `prime` (the rule and the goal stated) / `control` (the setting alone); its tip is git history (b9dea985). This branch keeps its two poles, whole-run window and `control`'s setting, and restores `full` / `own`, now under `SIM_CONDITION`.

## References

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103.

Brockers, V. C., Ehrlich, D. A., & Priesemann, V. (2026). Disentangling interaction and bias effects in opinion dynamics of large language models. *Nature Communications, 17*, 10077.

Cheng, M., Piccardi, T., & Yang, D. (2023). CoMPosT: Characterizing and evaluating caricature in LLM simulations. *EMNLP 2023*.

Chuang, Y.-S., et al. (2024). Simulating opinion dynamics with networks of LLM-based agents. *Findings of NAACL 2024*.

Donkers, T., & Ziegler, J. (2025). Understanding online polarization through human-agent interaction in a synthetic LLM-based social network. arXiv:2506.15866.

Maier, B. F., Aslak, U., Fiaschi, L., Rismal, N., Fletcher, K., Luhmann, C. C., Dow, R., Pappas, K., & Wiecki, T. V. (2025). LLMs reproduce human purchase intent via semantic similarity elicitation of Likert ratings. arXiv:2510.08338.

Meylahn, B. V., & Meylahn, J. M. (2024). How social reinforcement learning can lead to metastable polarisation and the voter model. *PLOS ONE, 19*(12), e0313951.

Ohagi, M. (2024). Polarization of autonomous generative AI agents under echo chambers. *WASSA 2024*.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.

Petrocelli, J. V., Tormala, Z. L., & Rucker, D. D. (2007). Unpacking attitude certainty: Attitude clarity and attitude correctness. *Journal of Personality and Social Psychology, 92*(1), 30–41.

Tormala, Z. L., & Petty, R. E. (2002). What doesn't kill me makes me stronger: The effects of resisting persuasion on attitude certainty. *Journal of Personality and Social Psychology, 83*(6), 1298–1313.

Zhang, J., Liang, X., Deng, A., Bonge, N., Tan, L., Zhang, L., & Zarrett, N. (2025). Leveraging interview-informed LLMs to model survey responses: Comparative insights from AI-generated and human data. arXiv:2505.21997.

Zhou, K., Hwang, J. D., Ren, X., & Sap, M. (2024). Relying on the unreliable: The impact of language models' reluctance to express uncertainty. *ACL 2024*.

The other works cited here and in [`../analysis/README.md`](../analysis/README.md) are in the thesis bibliography.
