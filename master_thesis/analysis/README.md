# analysis

Everything that reads what [`../simulation/`](../simulation/README.md) produces: the exchange audit, the references, the fit, the surrogate, recovery and the tests. This file holds the study's design and its reasons; what runs when, with which counts and files: `schedule.md`. § Hypotheses and tests is preregistered before the main runs and tested with the run as the unit; everything else is descriptive.

| Path | What |
|---|---|
| `schedule.md` | set-up, exchange audit, references, pilot, recovery, runs, sensitivity |
| [`metrics.py`](metrics.py) | every shared label, table and interval: the declarations (`run_declarations`, `audit_declarations`), the outcome D and its alternatives, the signature table, the audit's readers, `order_bias`, `expression_bias` |
| [`fit.py`](fit.py) | M0–M3 by maximum likelihood, the held-out scores, the outcome score (`crps_score`), the tests (`sign_flip`, `validity_test`, `coupling_test`) and H3's `transition` |
| [`surrogate.py`](surrogate.py) | runs with fitted declarations and replayed audit exchanges: recovery, the outcome tests, sensitivity |
| [`recovery.py`](recovery.py), [`recovery.ipynb`](recovery.ipynb) | one synthetic dataset's parameter and model recovery and test decisions, the ranges' profiles, the merge and the readings; the notebook sets the ranges and reads the rows from `logs/recovery/` |
| [`run_recovery.sh`](run_recovery.sh) | the SGE array on the SCC's CPUs, one task per dataset |
| [`robustness.py`](robustness.py) | the specification tables: each test under what its primary specification fixes |
| [`audit/build_audit.py`](audit/build_audit.py), [`audit/run_audit.sh`](audit/run_audit.sh) | one audit sub-phase per cell (`--stub` runs locally); the SCC submitter |
| [`audit/calibration_analysis.ipynb`](audit/calibration_analysis.ipynb) | the calibration: code paths, time per unit and the audit jobs' length, a first look |
| [`audit/audit_analysis.ipynb`](audit/audit_analysis.ipynb) | the audit's report per cell, its decision and its sample sizes |
| [`benchmark/benchmark_sft.py`](benchmark/benchmark_sft.py) | the binary benchmark, SFT or assimilation: `sweep` writes a CSV with its D per round and sidecar, `selftest` checks the code |
| [`benchmark/benchmark_analysis.ipynb`](benchmark/benchmark_analysis.ipynb) | reads the benchmark CSVs under `logs/benchmark/`, each by name |
| [`pilot_analysis.ipynb`](pilot_analysis.ipynb) | the pilot's runs, phase 1 under `logs/pilot/phase1/` and phase 2 under `logs/pilot/phase2/` (`PHASE`): what was copied back, D, the community means and the dispersion of the declared poles over the rounds per model and topic, the poles' shares, and the change of the declaration against the memory with its response curve |
| [`runs_analysis.ipynb`](runs_analysis.ipynb) | the runs' outcomes, signature table, fits and tests, pilot phase 2's pooled with the main runs' at H |
| `logs/benchmark/` | the reference CSVs; the audit writes `logs/audit/` (its calibration `logs/audit/calibration/`), recovery `logs/recovery/`, copied-back runs go to `logs/runs/`, the pilot's to `logs/pilot/phase1/` and `logs/pilot/phase2/`; git ignores `logs/` |
| [`requirements.txt`](requirements.txt) | the local environment |

`.py` modules run and write; notebooks read and interpret. Every module but the audit builder imports no LLM and runs locally, recovery's datasets also on the SCC's CPUs; `fit.py`, `surrogate.py`, `recovery.py`, `robustness.py` and `benchmark_sft.py` each have a `selftest`.

## Terms

Used below before their section; the mechanism's own terms: [`../simulation/README.md`](../simulation/README.md) § Terminology.

- **Declared position y:** what an agent states when asked, the one call that reads memory and the data the models fit. **Argued position s:** what its turn argues. **Self-read ŝ:** the agent's blind read of its own turn. **Probe ô:** its read of the partner's turn. All are poles, ±1; **r̂ = ŝ·ô** is the perceived reward, logged and never shown.
- **Window:** the entries (both turns of an exchange) a declaration sees: every entry the agent has stored, about 2H at the final declaration. **Trace:** a running mean T ← T + a(v − T) over a window's entries at rate a (§ The models, Traces).
- **Run, cell, group:** one execution of the simulation; one topic × LLM; the R runs of one cell and condition. **Fold:** one leave-one-run-out partition, fitted on the other runs and scored on the one left out.
- **p_inter:** the coupling, the probability that a cross-community pair is linked (a within pair: 1 − p_inter). **ε:** exploration. **α:** learning rate. **N:** agents. **H:** the horizon in rounds. **R:** runs per group.
- **Divergence D, D̄(p_inter, t):** a run's outcome D = |m₀ − m₁| / 2, m_c community c's mean pole, in [0, 1]: 1 when the communities hold opposite poles to an agent, 0 when they lean alike. D̄ is its mean over runs at that coupling after t rounds, the estimate of E[D] (§ Hypotheses and tests, Outcome).
- **Canon:** Banisch & Olbrich (2019). **Surrogate:** a run with fitted declarations and replayed audit exchanges (§ Recovery); a cell's **pool** is the audit's exchanges it replays. **Limit:** the surrogate under the canon's argmax rule (β → ∞, no habit) on a cell's pool (`surrogate.LIMIT`). **Ladder:** the D̄ curves from the canon scan to the runs (§ The references).

## The study

**Research question.** Do LLM agents learn from social feedback as Social Feedback Theory (SFT; Banisch & Olbrich 2019) describes, and does this learning explain how their networks end up?

**Target system:** LLM agents. The findings matter wherever LLM agents stand in for people in simulations of opinion dynamics; they do not say that people learn so, or that human polarization arises so.

**Strategy.** Fit four models, bias, habit, social feedback alone and habit with social feedback, to the positions LLM agents declare in a network, as Schubert et al. (2024) fit learning models to an LLM's choices, and compare them on held-out runs. A second condition withholds the partner's turns, so a partner term that still gains there is not the partner's. Then ask whether the fitted rules reproduce how the runs end.
- **Learning sits in one call.** Only the declaration reads memory, as in Schubert et al.'s paradigm, where each choice is one call with the earlier choices and outcomes in its context: the agents learn a declaration policy in context. The rest of an exchange (turns, self-reads, probes) reads no memory, so it is an environment outside the decision, which the audit measures before any run ([`../simulation/README.md`](../simulation/README.md#memory) § Memory).

**Method.** Wilson & Collins (2019) is the primary methodological guideline: the signature table, recovery, fit and validation follow their ten rules for modelling behavioural data. In their terms the runs are the experiment, the LLM agents its subjects and their declarations the data; the simulation is the surrogate, which runs the candidate models (§ Recovery). Fogarty et al.'s (2022) ten rules for principled simulation modelling guide the simulations, the runs, the benchmarks and the surrogate, chiefly on their limits (`schedule.md` § Sensitivity), on equifinality (H2) and on showing runs or replicates whole beside any summary (Rule 10).

**Not modelled, by design:**
- partner influence: assimilation (DeGroot 1974; Friedkin & Johnsen 1990), bounded confidence (Deffuant et al. 2000; Hegselmann & Krause 2002) and negative influence (Macy et al. 2003; Flache & Macy 2011), the mechanisms Banisch & Olbrich (2019, § 2) set SFT against, by Flache et al.'s (2017) classes, are not fitted: the question is whether the agents learn from social feedback, not which opinion-dynamics model describes them best. Assimilation appears only as reference curves (§ The references);
- homophily: the graph is static, there is no edge layer;
- identity and affect: agents carry no group labels;
- memory selection: see [`../simulation/README.md`](../simulation/README.md#memory) § Memory;
- argument persuasion as a separate mechanism (Vinokur & Burnstein 1978; Mäs & Flache 2013): text is all a declaration reads beside the setting, so arguments act wherever the terms do, and no condition or term tells them from positions (§ Hypotheses and tests, Reading the result). Memory reaches only the declaration, so an agent cannot repeat in its turns the arguments it has met, the exchange that drives Mäs & Flache's model;
- the expression channel as a rival: what a turn carries and how it is read reach every model only as its inputs ŝ and ô, which the audit measures and the surrogate replays, so every hypothesis is tested given the channel. A lean of the channel reaches habit's trace, which the partner term must beat in H1 and H2; an outcome the channel shapes alone is credited to no mechanism (§ Hypotheses and tests, Reading the result).

How the design keeps and extends the canon: § Against the canon.

## Conditions

`SIM_CONDITION` sets what a declaration shows of each stored entry, and nothing else: both conditions store both turns and open the declaration with the same setting, which states no rule, goal or reward, and the exchange, which reads no memory, is the same ([`../simulation/README.md`](../simulation/README.md#memory) § Memory).

| Condition | Shows | Can act | Role |
|---|---|---|---|
| `full` | both turns | habit and feedback | carries the hypothesis tests |
| `own` | the own turn | no partner term | negative control: every partner weight should vanish |

The setting mentions a partner in both; only `full` shows what the partner said. ŝ and ô are logged for the models and reach no prompt. Each turn answers the moderator's opener, never its partner's turn ([`../simulation/README.md`](../simulation/README.md#the-exchange) § The exchange, Why partner-blind turns), so ŝ depends on the agent's own argued position, not the partner's, and the partner reaches an agent only through its stored turn, which only the declaration reads, where the models fit feedback. `own` thus holds nothing of the partner. A turn that answered its partner's would carry the partner's pole into ŝ, and through it into K and r̂, in both conditions.

## The references

The **canon scan** is the canon: the binary benchmark with only agent i learning, from the canon's polarized start (§ 5.4), at the canon's ε = 0.1 (`--epsilon 0.1`), so i's argmax flips with probability ε through `explore()` while agent j answers with its argmax (B&O, Fig. 1). The exploring side follows `--both-learn`: i alone without it, both sides with it. It runs to 20,000 rounds on the grid p_inter = 0, 0.01, …, 0.50 with seeds 1–300 (SE ≤ 0.029 per p_inter), read at H and at 20,000. Its rows hold only the last round's state; its `_divergence.csv` holds D̄(p_inter, t), with D's sd, at every round. The SE bound holds for D too: a value in [0, 1] has variance at most ¼.

The **symmetric benchmark** is the canon scan with both sides learning and exploring (`--both-learn`), as both agents of a run's exchange do, Meylahn & Meylahn's (2024) symmetric variant. It differs from the runs only in what the LLM does: the expression channel and the perceived reward.
- **Memory:** the canon's Q, which keeps every reward it was updated with, discounted; a run's window likewise holds the whole run, so neither truncates. `surrogate.LIMIT` reproduces it with the models' ΔQ, a trace from 0 (`recovery.py selftest`).
- **Start:** the runs' and the canon scan's, the canon's polarized start (§ 5.4): community 0 on +1, community 1 on −1, each pole standing until its agent's first entry, as a run's start stands for its first exchange. A tie in ΔQ keeps the current pole.
- **Exploration:** the canon's ε = 0.1, on both sides since both learn, a run's nominal rate. A run's realised rate differs: an explored turn read back on its declared pole counts as none, and the self-read moves other entries off it (`off_pole`, [`runs_analysis.ipynb`](runs_analysis.ipynb) § 6).
- **Everything else is the canon's:** α = 0.05 and ε-greedy expression (§ What departs).
- **Horizon:** 20,000 rounds, the canon's own, as in the canon scan, with seeds 1–300 on the grid, so both share each seed's graph; read at H and at 20,000. Beside one row per run, the sweep averages D (`metrics.divergence`) per p_inter at every round, so it gives D̄(p_inter, t) at every t without logging every run's every round.

The **assimilation benchmarks** (`--rule assimilation`) are the canon scan and the symmetric benchmark with the update swapped: each learner keeps one trace A ← A + α(s_partner − A) of its partners' argued poles and declares sign(A), a tie keeping its pole. This is Martins' (2008) CODA, DeGroot's averaging (Flache et al. 2017, eq. 3) through the runs' binary channel, which carries a pole, never a strength; classical DeGroot would add influence through information no agent gets. Start, α, ε, graph and horizon are the SFT pair's, and one seed draws the same graph, pairings and exploration under either rule, so each pair of curves differs in the update alone. In the one-sided version agent i's exploration reaches nothing, since i learns only from j's pole. Where an assimilation curve meets its SFT pair, D alone cannot tell feedback from influence at that p_inter.

**The ladder.** Per cell, these D̄(p_inter, t) are read at H, the first three also at 20,000 and the fitted rules' also at 2H. Each curve adds a layer to the one before, so the gap between neighbours describes what that layer does to the outcomes. It enters no test.

| Curve | Adds | The gap reads as |
|---|---|---|
| canon scan | – | – |
| symmetric benchmark | both sides learning and exploring | what symmetry does |
| surrogate: Limit, the cell's replayed exchanges | the LLM's expression and perception | what the channel's noise and pole bias do |
| surrogate: fitted M3 | finite β, habit | what the fitted rule does |
| the runs | the LLM | what the fitted rules miss |

The surrogate at fitted M1 stands beside the ladder, not on it: if its curve falls with p_inter too, a fall is not specific to feedback. So do the assimilation benchmarks, each beside its SFT pair: the gap is what learning from agreement does that tracking the partners' poles does not.

A replay curve below the symmetric benchmark's at p_inter = 0 points to pole bias: with independent communities that each end on one pole, + with probability q, E[D] = 2q(1 − q), below ½ wherever q ≠ ½. A gap only at intermediate p_inter, or one growing with t, points to misperception acting like exploration. The surrogate's curves show what the rules do, not what an LLM would. The fitted rules' go on to 2H: past H, the last round a run can check, they predict where D would go, and no test reads them there. Only the Limit's goes on to 20,000.

## The models

**Outcome.** A declaration y_t is a pole, ±1, modelled as a binary logit, P(y_t = +1) = σ(η), so any lean toward a pole is b's and either pole is equally likely at η = 0, as the canon's agents always take one. A starting position is set, not modelled. Exploration acts after a declaration, so it removes none from the fit.

**Inputs.** C_t is the agent's window at declaration t ([`../simulation/README.md`](../simulation/README.md#memory) § Memory), logged by id in the order stored. Entry k shows the two turns' text in `full`, the own turn's in `own`. The models read what the log measures of it: ŝ_k, the self-read of the agent's turn, ô_k, the probe of the partner's, and r̂_k = ŝ_k ô_k; no prompt shows any of them. An explored turn is stored like any other, unmarked, since Schubert et al. find an LLM learns differently from its own choices' outcomes than from observed ones: ŝ_k is that turn as the agent read it, an *explored entry* where it lies on the pole argued (`metrics.explored_entry`).

**Inputs are measurements, a stated limit.** A declaration reads the text; the models read one self-read and one probe of the text, so every term carries measurement error, whose sampling part the audit's Reliability sizes per cell from the reads' logprobs, apart from any real gap between the two reads (`schedule.md` § Exchange audit). Weights attenuate and power falls. A noisy K also under-controls habit: where partners mostly agree, ΔQ can act as a second proxy of the agent's own history, so M3 could beat M1 with no partner at work; `own`, whose prompt holds no partner, guards against it (§ Hypotheses and tests, Validity check). Nothing sizes the three on synthetic data, since recovery holds no measurement error (§ Recovery). None of this is corrected. The option, not taken: since no prompt sees a draw, the fit could read each expected read in place of the drawn one (regression calibration). That removes sampling noise alone, not any gap between a blind read and the declaration's own reading of the text.

**Order within an entry.** `full` shows the own turn first, the partner's after it ([`../simulation/README.md`](../simulation/README.md#memory) § Memory), and no term reads the order. A primacy or recency effect is a constant tilt between the two texts: it cannot make M3 beat M1 where the partner's text has no effect, but it scales β against γ_K, so the weights hold for this format.

**Traces.** Both terms are traces in the canon's update form, T ← T + a(v_k − T), run over C_t in order from 0: the delta rule of reinforcement learning. Few entries leave each near 0, so conviction builds up with experience. Since both share this form, habit and feedback differ in what they read, not in how they aggregate. The audit's serial-position windows check the form.

**The graph is an input, not a parameter.** No weight is per edge, and none per agent beyond its name's intercept: the graph only sets who meets whom, and each term weighs an entry by what was read in it, so the fitted rules are not tied to one graph and the surrogate can run them on any p_inter's. This is Probine et al.'s (2026) adjacency-based form. On one fixed graph, their fully parameterized per-edge influence matrix lowered training error but raised both held-out errors.

**Full model:** η = b + b_n + κ·F + γ_K·K + β·ΔQ
- **b**, the baseline: the LLM's innate bias on its topic (Probine et al. 2026), as the declarations of the reference name, Barbara, the first in alphabetical order, show it.
- **b_n**, the name: one intercept per other name n of the 20, its departure from Barbara's lean, reference-coded in alphabetical order (`fit.NAME_ORDER`); b_n = 0 for Barbara. A name carries gender and ethnicity cues an LLM may act on, and names recur across runs, so a held-out run's agents keep theirs; left in the noise, a name's lean would pull the other weights toward 0, as answer order's would. It anchors no agent to its initial opinion, which Probine et al. find does not improve fit; a specification table drops the name intercepts (`schedule.md` § Sensitivity).
- **F**, answer order: +1 when the declaration's list starts from `TOPIC_TEXT`'s pole (`topic_first`), −1 otherwise, with one weight κ per cell, the LLM's lean toward the first answer. It is of no interest in itself and is modelled as Wilson & Collins (2019, Box 6) model a side bias: left in the noise, it would pull every other weight of a logit toward 0.
- **K**, habit: the trace of ŝ_k at rate α_K, K ← K + α_K(ŝ_k − K), Wilson & Collins's (2019) choice kernel in its two-option difference form: the repetition of recent turns. It stays in every model a test compares: a learning model without perseveration absorbs choice repetition into its learning parameters (Katahira 2018; Sugawara & Katahira 2021), which here would credit it to ΔQ. M2 leaves it out only to show the canon's rule alone, descriptively (§ Hypotheses and tests). One habit rate is a deliberate simplification: an own history slower than α_K, which ΔQ at a small α could stand in for, is not separated from feedback (§ Identification).
- **ΔQ**, social feedback: Q(+1) − Q(−1), where Q(o) is the trace of r̂_k at rate α over the entries whose own pole ŝ_k is o; an unexpressed pole stays at 0. This is the canon's update applied to what the turns are read to say. β ≥ 0: with β < 0, the pole that met disagreement would become more likely, which is the negative-influence pattern rather than SFT, yet the tests would count it as feedback. One rate serves both signs, the canon's form, though Schubert et al. (2024) find LLMs learn more from positive than from negative prediction errors; part of such an asymmetry can be perseveration, which K takes (Katahira 2018).

| Model | η | Theory |
|---|---|---|
| M0 | b + b_n + κF | baseline |
| M1 | M0 + γ_K K | habit |
| M2 | M0 + βΔQ | social feedback alone: the canon's rule |
| M3 | M1 + βΔQ | habit and social feedback |

**Reading the weights and rates.**
- **β against the canon:** the canon's argmax is M2's limit β → ∞ with b, b_n and κ at 0, which `surrogate.LIMIT` reproduces as the symmetric benchmark (`recovery.py selftest`). A finite β is choice noise the canon lacks, whose only noise is ε, here drawn after the declaration: β reads how closely a declaration follows remembered feedback.
- The window holds the whole run, as the canon's Q does, so a fitted α is compared directly with `LEARNING_RATE`, the canon's 0.05.
- At small rates a trace is nearly a sum, so only the products γ_Kα_K and βα may be identified (§ Recovery).
- **β beside K:** where partners mostly agree, ΔQ follows the agent's own history, which K takes; β then rests on the entries where they disagree (Identification).
- A trace's range depends on its rate, so effects are reported as changes in predicted probability.

**Reading across conditions.** The terms are computed from the logged values in both conditions. In `own` the prompt holds no partner, so a partner weight that does not vanish reads something outside the prompt.

### Identification

Written out, K = Σ_k α_K(1 − α_K)^(n_k) · ŝ_k, with n_k all later entries, and ΔQ = Σ_k ŝ_k · α(1 − α)^(m_k) · r̂_k = Σ_k α(1 − α)^(m_k) · ô_k, with m_k the later entries on entry k's own pole. K reads the agent's own pole, ΔQ the partner's, discounted per own pole. Where partners mostly agree, ô_k follows ŝ_k and ΔQ follows the agent's own history too, so β rests on the entries where they disagree. Exploration supplies them: an explored entry, an explored turn read back on the pole it argued (`metrics.explored_entry`), puts into a window an own pole the partner need not share. Were every one read back so, a settled agent's 2H ≈ 200 entries would hold about 20 at ε = 0.1. The effective rate, ε times the share read back so, is reported per run ([`runs_analysis.ipynb`](runs_analysis.ipynb) § 6). Exploration also keeps a settled run producing disagreements, which it would otherwise stop doing. The audit sets the variation instead of waiting for it: the own and the partner's pole vary across its one-entry windows, where K = α_K ŝ and ΔQ = α ô, so habit moves y with ŝ and feedback with ô.

### Two poles

Each entry adds

| Trace | Entry |
|---|---|
| K, habit | ŝ: the own pole |
| ΔQ, feedback | r̂ = ŝô, credited to pole ŝ, so ô: the partner's pole |

M3 against M1 is identified, since the partner's pole is what ΔQ adds. Its form is the canon's on poles: it does not scale with a strength, which two poles lack, so ΔQ is a trace of the partners' poles, kept per own pole. The five-position design, whose graded reads would have let ΔQ scale with own strength, is on branch `three-models`.

## Signatures

One model-independent table per cell and condition, read before any fit, as Wilson & Collins (2019) advise, and entirely descriptive: it gates nothing and enters no test. `metrics.signatures` gives, per (ŝ_k, ô_k) of an agent's latest entry k, the n and the mean next declaration y_t with its interval; the share of y_t on ŝ_k's pole is (1 + ŝ_k · ȳ) / 2. The audit's one-entry declarations, the runs and the outcome tests' replicates read it alike, the runs also on explored entries alone, whose own pole is drawn, not chosen from the agent's history, and on the rest, which the explored ones would otherwise mix into the cells where ô disagrees with ŝ.

What each model would show at one entry, where K = α_K ŝ and ΔQ = α ô:
- **Habit:** y rises with ŝ, flat along ô within a row.
- **Feedback:** y rises with ô, alike in both rows of ŝ.

`own`, whose agents never see a partner, should be flat along ô in both rows.

A rise along ô that differs between the rows, or one only where ô agrees with ŝ, is not feedback's form; no model here has it, and it is reported as a limit of H1's reading.

## Hypotheses and tests

Banisch & Olbrich (2019) show that polarization can arise from feedback alone, without negative influence, bounded confidence, opinion homophily or argument exchange: a claim of sufficiency, not that agents use nothing else. Here *alone* means feedback as the only partner term; prior, answer order and habit stay in every model, and the arguments in the stored turns are not excluded: the tests read the canon's rule, not its route (Reading the result).

**One contrast on two scales.**

| Hypothesis | Contrast | Scale |
|---|---|---|
| **H1:** social feedback explains more than habit and prior | M3 vs M1 | declarations |
| **H2:** social feedback explains more than habit and prior | M3 vs M1 | outcomes |

**One test, across runs,** serves both. Leaving one run out at a time, both models of a contrast are fitted on the other runs and score the held-out run; its gain d_r is the difference in scores. The test is a one-sided sign-flip permutation test of a mean gain above zero (`fit.sign_flip`), its size read on synthetic data only (§ Recovery). **H0:** E[d_r] ≤ 0, E the expectation over a group's runs; the sign-flip's exact null is a d_r symmetric about zero.
M3 adds ΔQ to M1 and nothing else, so a gain says that the partner's measured agreement, in SFT's form, predicts declarations beyond habit. M1's and M2's gains over M0 are reported on the declarations, descriptively, and enter no test: M2's would credit repetition to ΔQ (§ The models, K).

**Declarations (H1).** d_r is the difference in the log-likelihood of the held-out run's declarations (`recovery.declarations`).

**Outcome**, read by H2 and H3 from a run's final declarations (round H + 1) as a state, never as converged: in Meylahn & Meylahn's (2024, § 2.3) symmetric variant, whose learning and exploring on both sides the runs follow, every state recurs at the runs' ε > 0. The state at H is thus a draw from a recurrent process, and how far apart the communities stand in it is graded, not an event that happens or not. The outcome is the **divergence D = |m₀ − m₁| / 2**, m_c the mean pole of community c's members: Bramson et al.'s (2016) group divergence on the poles, in [0, 1]. D = 1 when the communities hold opposite poles to an agent, the canon's opposed groups (§ 5.3); D = 0 whenever they lean alike, consensus included. It reads poles, the canon's opinion. It is symmetric in the poles, so an LLM favouring one tilts consensus direction, not D. The canon counts consensus instead (§ Regime definitions). D is `metrics.divergence`; both benchmarks are read the same way, from `mag_c0` and `mag_c1`, not from `classify()`. `schedule.md` § Sensitivity reads its alternatives.

**Outcomes (H2).** For each run, the surrogate takes the declaration test's fits of M1 and M3 on the other runs, so no run's outcome is scored by a fit to its own declarations, and runs n_rep replicates of each (`schedule.md` § Runs) on the run's graph from its starting positions. Each run is scored by the continuous ranked probability score of its observed D under each model's replicates X, CRPS = E|X − D| − ½ E|X − X′| (Gneiting & Raftery 2007), by Ferro's (2014) fair estimator, unbiased at any n_rep (`fit.crps_score`); d_r is the difference in −CRPS (`recovery.outcomes`). The score is proper, so no model gains by replicates other than its own; it credits replicates near the observed D, not only those that hit it, and an outcome none reaches still scores finitely. The replicates start from the run's own positions, so habit alone, carrying the starting division forward, may already score well, which H2 has to beat. The replicates replay the cell's audit exchanges, which serve both conditions, since no turn reads memory.

**H3 (coupling).** The expected divergence E[D] decreases as the coupling p_inter rises: the canon's direction, not its transition. In the canon, polarization between two communities loses stability as their coupling grows, with the transition at p* = ½ in the large-system limit (Banisch & Olbrich 2019, § 5.4), from the polarized start the runs share. The canon's topology fixes both ends of the p_inter grid. At p_inter = 0 no cross pair is linked, so the two communities evolve independently, each from its own pole, and D falls below 1 only as far as either leaves its pole by H. At p_inter = ½ every pair is linked with probability ½ whatever its communities, so after the start the dynamics cannot tell the communities apart, and D reads only what is left of the starting division: a floor owed nothing to the topology. In between, E[D] decreases with p_inter.
- **Test** (`fit.coupling_test`): a one-sided permutation test of D's least-squares slope on p_inter, its null distribution from permuting D across runs, within topics when pooled, p_inter centred within each. **H0:** D is independent of p_inter, within topics when pooled. Permutation keeps it exact for D's bounded, skewed and tied values, which a normal slope test would not. Since neither end of E[D] is 0 or 1, the transition is the p_inter at which E[D] falls halfway from its value at p_inter = 0 to its value at ½, read off the non-increasing isotonic fit of D on p_inter (`fit.transition`), which assumes no form beyond H3's direction; it is undefined where the fit does not fall. It is not p*, and it is descriptive, reported beside the symmetric benchmark's (matching N, graph, H and both sides learning) and the canon scan's at H (matching N, graph and H): a rejection says E[D] decreases, not where, nor how closely it tracks the benchmarks, nor by what mechanism. E[D] mixes how often and how far the communities part, so a fall does not say whether opposition grows rarer or shallower; every run's D at its p_inter is shown beside it ([`runs_analysis.ipynb`](runs_analysis.ipynb) § 7).

**Validity check** (`fit.validity_test`). In `own`, M3's mean gain over M1 stays below half its value in `full` (one-sided 0.025, bootstrap over runs): a gain that mostly survives without any partner text is not the partner's but ΔQ standing in for the agent's own history, which one habit rate does not take (§ The models, K). **H0:** E[d_r in `own`] ≥ ½ E[d_r in `full`]. It sits outside the error control (§ Scope); if it fails, the declaration tests are uninterpretable in that cell, and the outcome tests with them, since they run the same fits.

**Reading the result.** Per scale:

| H1, H2 | Reading |
|---|---|
| not rejected | the partner is not shown to matter beyond habit |
| rejected | **the agents' declarations follow their partners' measured agreement, in SFT's form, beyond habit** |

Recovery's power on the cell's pool says whether a *no* means no gain or too little to tell. Report `own` beside it: where ΔQ only proxies habit, M3 beats M1 there too (§ The models, Inputs).

*Feedback* in every hypothesis means the canon's rule as a form: declarations follow ΔQ, the trace of the partner's measured agreement. It does not mean the canon's route, the experience of agreement without argument processing (Banisch & Olbrich 2019, §§ 1–2). Like-minded partners' arguments also strengthen conviction (Mäs & Flache 2013; Flache et al. 2017, § 2.67), and the canon calls its rule an abstraction of such models (§ 6). Text is all a declaration reads beside the setting, so a rejection fits either route; the design does not separate them, and cannot show that an LLM experiences agreement at all.

Feedback explains how the networks end up only if H1 and H2 both reject. Where H2 does not, the fitted partner term generates the outcomes no better than habit. Either habit and the replayed exchanges alone carry them (M1 scores high), or neither model generates them (both score low), so something outside the modelled terms does: in the declaration, the one call that reads memory, or in what the surrogate's draws pool over: the call order, the names and the opener's wording. Noisy reads can also weaken the fitted weights, so that the fitted rule follows its partners less closely than the agents do (§ The models, Inputs).

Where H2 rejects and H1 does not, the outcome test makes no claim about the agents: the rule generates the outcomes but the agents are not shown to follow it (equifinality; Fogarty et al. 2022, Rule 9).

H3 says separately whether E[D] decreases with coupling.

**The signature table against tests.** It is read beside each reading, not into it: where it contradicts a test, the test's reading stands and the contradiction is reported as its limit (Wilson & Collins 2019).

### Scope

Tests run in `full`, per cell, at one-sided levels that sum to 0.05:
- **H1 and H2,** 0.0125 each; neither gates the other.
- **H3,** 0.025, alone: a decrease of E[D] with coupling presumes no feedback in the agents.

**A cell's chance of any false rejection** is thus at most 5%, by Bonferroni, under any dependence among the tests. The price, against 0.025 per test, is about 21% more runs for 80% power (normal approximation). A conclusion on both scales needs both to hold, an intersection-union, so it adds no correction. The outcome tests read the declaration tests' fold fits, so the two are one line of evidence at two scales, not two confirmations: where ΔQ only proxies habit, β > 0 carries into the surrogate, whose agents then respond to ô itself, and only `own` checks that (§ The models, Inputs).

H3 is also reported pooled per LLM, permuting within topic, outside the error control: a cell with a low D̄(0, H) may not reject on its own, and that is reported as its limit. Across cells, the number of cells in which each test rejects is reported, uncorrected: cells replicate the question, not divide one claim. The hypotheses and levels are preregistered before the main runs.

## Recovery

One surrogate simulator ([`surrogate.py`](surrogate.py)) serves recovery, the outcome tests and the sensitivity analysis: its agents follow M3 at given parameters, M0 to M2 being M3 with weights at 0, on the canon's graph, the 20 names placed per seed, each agent's window built as in a run, exploring at the runs' ε. Each exchange replays one of the audit's, drawn uniformly from the meetings at its two argued positions. The two roles differ only in call order, so each meeting counts in both orientations, filed under both orderings of its pair; a perception distribution then rests on the 2N probes of the audit's Perception (`schedule.md` § Exchange audit). The draw fixes each side's self-read ŝ and the ô it perceives; s is the argued position, as in a run, and `explored` adds nothing, since no prompt shows it. A draw keeps both sides' reads together, so nothing about the exchange is pooled but the call order, the names and the opener's wording, the last two drawn in the audit as in a run. No call after the declaration reads memory, so the audit's exchanges describe the exchange, not the history, and serve both conditions. A window follows exactly from the stored history, so nothing about memory is emulated; a synthetic agent responds to ŝ and ô themselves, where a run's reads the turns, so recovery holds no measurement error (§ The models, Inputs).

**Recovery** is Wilson & Collins's (2019) two checks on synthetic groups of the planned design, before the main runs: whether the models' parameters come back and whether the models can be told apart; beside them, the tests' power and false rejections on the same groups. None has a pass mark, since the paper sets none: simulated and recovered values should correlate tightly without bias, and the confusion matrix should lie near the identity. [`recovery.py`](recovery.py) computes one dataset per SCC array task; [`recovery.ipynb`](recovery.ipynb) sets the ranges and reads the merged rows (`schedule.md` § Recovery):
1. **Range:** the models are new and no fit may set it, so the parameters are drawn as widely as behaviour responds to them, read off simulation, and no pilot value enters: b, the name intercepts and κ at M0's fit to the audit's empty-memory declarations, each weight, alone, from where it starts to move behaviour to where doubling it no longer does, drawn on the log scale. A parameter may fail in part of that range, as long as it comes back where the real fits fall.
2. **Parameter recovery:** each model fitted to its own datasets. Simulated against recovered is plotted per parameter, with r and bias, rates and weights on the log scale they are drawn on, where a few large values would otherwise set both; a fit at or below 0, off that scale, is counted; correlations among the recovered parameters that the draws lack show what the fit induces.
3. **Model recovery:** data from each of M0–M3, all four fitted and the one with the best summed held-out score chosen, the score the tests use. The confusion matrix P(chosen | true) and its inversion P(true | chosen) are reported (Box 5), and BIC's beside them, since a matrix far from the identity may reflect the criterion.
4. **Power and false rejection:** every test run on every dataset as on the runs, at its level, at R = 33 and its nested 17 and 9, the outcome tests at n_rep 40, 20, 10 and 5. A rejection where the hypothesis holds under the generating model is power, elsewhere a false rejection (`recovery.truth`). R comes from the run budget against each cell's power at each R; past 33, the normal approximation extrapolates it from the per-run gains.

A failure is first looked at as a bug, then as a design that gives too little data, which moves R or the models. After the real fits, § The fit, Range may repeat it.

**What it speaks to:**

| Hypothesis | Holds under | What it shows |
|---|---|---|
| H1, H2 | M2, M3 | whether feedback can be told from habit; confusion among M1–M3 |
| H3 | M2, M3 | whether a feedback rule's D falls with p_inter detectably, and whether habit alone makes it seem to (a rejection under M1) |

**Per cell.** Recovery runs on each cell's pool, so its synthetic data inherit that cell's reads and the windows its exploration yields, and every check is read per cell, never pooled across cells, whose ranges, intercepts and κ differ. A cell whose lean leaves a partner weight nothing to act on, so that its saturation aborts, is named in `recovery.SKIPPED` and its tests are reported without power (`schedule.md` § Recovery, step 1). What recovery cannot show: agents that respond to what the turns carry beyond their reads, since a synthetic agent responds to ŝ and ô themselves (Not checked, below).

**Not checked, stated limits of the plan:** the sign-flip test's level under correlated folds beyond the generating models, so a test's 0.0125 holds only as far as recovery's false rejections show; the replicate score's point estimates: every replicate runs its fold's fit, so the score holds no uncertainty about the parameters, and folds that share most of their runs pass an error in the estimate to every run's score alike; what measurement error in ŝ and ô does to the fit and tests; the surrogate's draws against a run's exchanges, beyond the shared code. Recovery tests no hypothesis.

**Recovery sets** how the learning rates are read, per cell: estimated; where they do not come back, only the products γ_Kα_K and βα are reported. R and n_rep it informs, not sets (`schedule.md` § Runs).

## The fit

- **Maximum likelihood** for M0–M3, per condition and cell, in [`fit.py`](fit.py), fitted to the declared poles and read in a notebook. The binary logit and its gradient are written out by hand on numpy and minimized with `scipy`. Each fit runs from random starts until the best log-likelihood stops improving (Wilson & Collins 2019).
- **Bounds:** the fits that end at β = 0 are reported per group.
- **Range:** the fitted parameters are checked against the range recovery sampled; outside it, recovery is repeated over the range of the fits and reported beside the tests. Likewise a cell whose lean on its runs, which have memory, departs from the audit's, which has none (`schedule.md` § Recovery, step 7).
- **Held-out scores** leave one run out at a time and feed § Hypotheses and tests. They need no parameter count, which BIC gets wrong here: a rate whose weight is 0 has no effect, at small rates only the products are identified, and β reaches its bound, yet BIC charges each as a free parameter.
- **BIC** of each pooled fit is reported per cell and condition, descriptively, for comparison with work that uses it (Schubert et al. 2024); it enters no test.

## Against the canon

The canon is Banisch & Olbrich (2019, "B&O"), cited by section and equation of the journal version. Meylahn & Meylahn (2024) prove B&O's bi-polarization metastable and give the symmetric variant. The mechanism: [`../simulation/README.md`](../simulation/README.md).

### What is canonical

| Element | Canon | Here |
|---|---|---|
| Opinion | ±1 | ±1, a forced choice between the two poles, each its own sentence |
| Start | Q uniform, opinion argmax Q: [0, 1] in the § 4.1 text, [−0.5, 0.5] in the Fig. 1 code, the same ΔQ | § 5.4's polarized start: community 0 on +1, community 1 on −1; drawn poles (`P_START_PLUS`) a sensitivity setting |
| Staging | both agents' opinions fixed before either reads the other (Fig. 1) | both declare partner-blind before either turn, and each turn answers the moderator's opener, never the partner's |
| Exploration | ε-greedy, ε = 0.1 (eq. 3) | ε = 0.1 through `explore()`, which flips the argued pole |
| Pairing | i uniform, j a uniform neighbour of i (§ 3.1) | the same, i drawn among agents with a neighbour |
| Visibility | only expressed opinions (§ 5.1) | only turns, stored as text; the models read the partner's position as its probe reads it |
| α | 0.05 (§ 3.1); 0.01 in § 5.4 | 0.05 in the benchmarks; fitted in the runs (§ The models) |
| Coupling | two communities, cross pairs linked with p, within pairs with 1 − p (§ 5.4) | `SBM_P_INTER` = p_inter, within pairs at 1 − p_inter |
| Run length | 20,000·N steps (§ 4.3, 5.4) | the canon scan and the symmetric benchmark |

### What departs

| Element | Canon | Here | What it moves |
|---|---|---|---|
| Opinion and expression | one act: argmax Q, ε-greedy, credited as executed | two acts: the declaration is the opinion, the turn its expression, which argues s, is stored as text and is credited in the models as the agent reads it back, ŝ | exploration flips the argued pole, never the declaration; the act is credited as its author reads it, a misreading B&O's ε may stand for ("misunderstanding or misinterpretation of an articulation", § 3.1) |
| Who learns | agent i only (eq. 1, Fig. 1) | both sides learn and explore, as in Meylahn & Meylahn's (2024) symmetric variant | the canon scan keeps the canon's, the symmetric benchmark the runs' |
| Route | the experience of agreement, no argument processing (§§ 1–2); an abstraction of persuasion models (§ 6) | the rule's form, fitted to declarations that read text, where agreement and arguments arrive together | a rejection fits either route (§ Hypotheses and tests, Reading the result) |
| Reward | r = o_i o_j, ±1 (eq. 2) | perceived: r̂ = ŝ·ô, ŝ the agent's own turn and ô the partner's, each as the agent reads it; no prompt states the rule or shows r̂ | equals r wherever both reads land on the poles argued; a misread flips its sign. B&O leave "more complex and possibly heterogeneous rewards" to further exploration |
| Expression channel | frictionless | a turn argues the position, unchecked, and its text is what memory holds; the agent's blind self-read of it and its probe of the partner's are measurements | expression inconsistency and misperception, measured by the audit and replayed by the surrogate |
| State | Q carried forward | the whole run's entries as text, shown only at the declaration | SFT's update is fitted to it; the benchmarks carry the canon's Q (§ The references) |
| Rivals | argued against (§ 2) | not fitted; habit is the null | the question is learning from reward, not which model describes the agents best (§ The study) |
| Size | p* = ½ as the communities grow (§ 5.4) | N = 20, communities of 10; 10 and 40 as sensitivity settings | p_inter = 0.5 is the analogue of p* = ½; communities of 10 are far from the limit |
| Horizon | 20,000·N steps | H = 100 rounds of N exchanges, fixed by design (`schedule.md` § Runs, Horizon) | the outcome is the state at H |
| Pole symmetry | by construction | rests on the LLM; every position call's answer order is drawn, and `metrics.order_bias` reads the bias back; the turns are what memory holds, so the expression channel's lean (`metrics.expression_bias`) enters memory too | an LLM favouring one pole, in its answers or in its turns, tilts consensus direction, not D; the null of direction is not 0.5 |

### Regime definitions

| | B&O (2019) | Here |
|---|---|---|
| Opinion | argmax Q (eq. 3) | the declared pole |
| Unit | every agent | each community's mean pole |
| Consensus | every agent prefers one opinion (§ 4.3; § 5.4), treated as absorbing | — |
| Polarization | anything else, operationally; opposed opinions on ½-cohesive groups, theoretically (§ 5.3, eqs. 7–8) | graded: D, the communities' mean poles apart, 1 at opposite poles |
| DV | consensus probability (Figs. 5–6); stable polarization as its complement (Fig. 11) | mean divergence D̄(p_inter); the benchmarks also report `classify()` |

`benchmark_sft.classify()` is the benchmark's per-round regime label, kept because `settle_round` needs a label every round; its rule and `CONVICTION` are in the code, read in [`benchmark_analysis.ipynb`](benchmark/benchmark_analysis.ipynb) § 3; § 4 reads settling off D̄ instead. The runs' outcome is D, a state at H rather than an equilibrium (§ Stability).

The canon's count suits its polarized start, where every non-consensus run is opposition that held. Read at H, a run need not have reached consensus or kept full opposition, so the outcome tests and H3 read D, which grades how much of the starting opposition is left; it is not the canon's count. From the shared polarized start, D̄(p_inter) and Fig. 11 both measure persistence. The two part most where runs end between consensus and full opposition, which the canon's count reads as polarization throughout. From drawn poles (`schedule.md` § Sensitivity), D̄ measures emergence instead.

### Stability: absorbing or metastable

- **B&O**: consensus is absorbing (§ 5.4), and opposed opinions on ½-cohesive groups are an equilibrium of the opinion game (§ 5.3).
- **Meylahn & Meylahn (2024, Theorem 1)**: for α ∈ (0, 1), ε ∈ (0, 1) and a connected graph, consensus is reached with probability one; B&O's bi-polarization is metastable. In the symmetric variant (§ 2.3) every state recurs.
- **Here** every benchmark explores at the canon's ε = 0.1, so none is outside the theorem any more, and the runs explore on both sides. No run settles: the outcome is the state at H, and what H misses is read on both benchmarks' D̄(p_inter, t) and the rounds after which D̄ stays within sampling error of its 20,000-round value (`benchmark_analysis.ipynb` § 2, § 4).

**References.** Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76–103 (arXiv:1704.02890). Meylahn, B. V., & Meylahn, J. M. (2024). How social reinforcement learning can lead to metastable polarisation and the voter model. *PLOS ONE, 19*(12), e0313951. The other works cited are in the thesis bibliography.
