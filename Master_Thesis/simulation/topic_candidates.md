# Candidate discussion topics

Why `config.TOPICS` holds the three dichotomies it does, and what the alternatives were. The default is **`nuclear`**.

The earlier default, restrictive immigration, carried a directional prior in the *generation* channel: drafts anchored against the model's lean held that stance markedly less often. (That harvest is not in the committed logs, so no figures are given.) The failure mode is documented: LLM debate agents drift toward the model's own stance (Taubenfeld et al. 2024), and networked LLM agents converge toward the model's prior (Chuang et al. 2024). Whatever the topic, the prior is **measured, not assumed** (§ Validation).

## Selection criteria

All five must hold; they are ordered by how expensive a violation is to repair.

1. **Crisp binary.** One opinion object on the ±1 dimension, stable across rounds. "How much" or "under which conditions" answers blur the Q-value semantics.
2. **Low latent valence.** A left-libertarian lean is documented across model families and instruments (Santurkar et al. 2023; Hartmann et al. 2023; Feng et al. 2023; Motoki et al. 2024; Rozado 2024), strongest on identity topics and weakest where **both stances have socially desirable framings**. Survey-style measurements overstate its stability (Röttger et al. 2024a), which is another reason to measure it in the simulation's own setting. Argument quality must be comparable on both sides, or the asymmetry re-enters as a polarization confound.
3. **Guardrail-safe.** Safety hedging on benign-looking prompts is systematic (Röttger et al. 2024b) and attaches to one stance, breaking symmetry at generation. Rule of thumb: questions about *systems* (energy, work, transport) are safe; questions about *groups of people* are not.
4. **Sufficient charge.** Agents need contested substance to build conviction. A bland topic buys symmetry at the cost of nothing to polarize over, and agreeable filler (sycophancy; Sharma et al. 2024) masquerades as convergence.
5. **Flippable as a pairwise comparison.** The −1 pole reverses the +1 pole's comparison, so both are positive statements of equal breadth, no status quo sits between them, and the stance-flip control (Stage 3) introduces no negation.

## The default: nuclear power, good against harm

- **+1:** *"Nuclear power does more good than harm."* **−1:** *"Nuclear power does more harm than good."* **Opener:** *"Does nuclear power do more good than harm, or more harm than good?"* (and the swapped order).
- **Binary:** one object, compared with itself rather than with another energy source; the only position on neither pole, a tie, carries no lean.
- **Valence:** both poles have progressive *and* conservative framings (decarbonisation, energy independence, baseload vs. waste, accidents, cost overruns, proliferation), with real-world anchors on both sides (France vs. Germany's *Atomausstieg*). A mild pro-nuclear lean is possible, since US support for expansion rose from 43 % (2020) to ~59 % (2025) across both parties (Pew Research Center 2025), while Hartmann et al.'s pro-environmental finding cuts the other way. The audit decides.
- **Guardrails:** none; energy policy, not people. **Charge:** high, with decades of live conflict.
- **Flip:** three earlier wordings failed. *Essential* / *too dangerous* never contradicted each other, so the swapped frame asked the rater for an inference. *A larger* / *a smaller role* were contraries: defending the current share sat on neither pole and was the easiest anchored argument. A plain negation closes that gap but puts a negation into every Stage 3 classifier call and stance sentence. Good against harm avoids all three.

## Registered dichotomies (`config.TOPICS`)

All three share the pairwise form and an opener naming both objects, drawn per exchange in either order. They span one axis, how polarized the topic is outside the simulation; the two alternates are Stage 5's arms ([`../analysis/final_schedule.md`](../analysis/final_schedule.md) § Bracket C).

| Key | +1 pole | −1 pole | Why |
|---|---|---|---|
| `nuclear` | nuclear power does more good than harm | more harm than good | the default: contested but cross-cutting |
| `renewables` | renewables should be prioritized over fossil fuels | fossil fuels over renewables | Stage 5, `5-01`…`5-03`: the **polarized** energy topic, mapping onto the partisan divide and where the model's own lean is expected to be strongest (criterion 2; cf. Hartmann et al. 2023) |
| `coffee_tea` | coffee is better than tea | tea is better than coffee | Stage 5, `5-04`…`5-06`: the **barely polarized** control, failing criterion 4 on purpose, to separate what the feedback mechanism produces from what a topic brings |

The same comparison in both directions keeps the audit's frame test clean: a text preferring either object supports exactly one pole, so a rater reading the text must mirror its label when the order is swapped. No proposition names a country, since personas are US records and ground their own arguments; `renewables` therefore reads as a global claim, which may strengthen the model's lean, and the audit measures that.

## Runner-ups (not registered)

Each would need rephrasing as a pairwise comparison (criterion 5) before use.

| Topic | Strength | Weakness |
|---|---|---|
| Universal basic income | balanced, cross-cutting argument space | "introduce" invites hedged intermediate positions; possible left lean |
| Mandated four-day week | "by law" forces the divisive question; relatable to any persona | mild labour lean |
| Compulsory voting | probably the lowest prior: participation vs. liberty, both desirable | charge medium at best |
| Urban congestion pricing | clean trade-off structure, live (NYC 2024-25) | technocratic; low charge unless personas are urban |

## Rejected

| Topic | Failing criterion |
|---|---|
| Illegal immigration (former default) | (2)+(3): measured generation-channel prior; harsher wordings trip guardrails |
| Abortion, gun control, death penalty | (2)+(3): strongest priors and highest hedging risk |
| GM food, vaccine mandates | (2): strong pro-science prior |
| Zoo bans, animal testing | (2): animal-welfare prior |
| Daylight saving time, homework abolition | (4): symmetric but bland |
| Congressional term limits | (2)/(4): ~80 % agreement, no social base for the contra side |
| AI regulation | (3): self-referential for an LLM |

## Validation

A wording's prior is measured, not argued:

1. **Grader side.** The classifier audit ([`../analysis/classifier/`](../analysis/classifier/README.md)) rates one corpus per topic in both answer orders, across all three models; its asymmetry tables and implied dyad cells are the reward-channel test.
2. **Generator side.** Per-stance first-attempt fidelity from the gate records (`analysis/metrics.persona_field`), and the refusal rate per anchored stance (`unrelated`).

The wording is frozen for the confirmatory schedule ([`../analysis/pre_schedule_report.md`](../analysis/pre_schedule_report.md), Decision 1): an edit invalidates every corpus and label file for that topic.

## References

- Chuang, Y.-S., Goyal, A., Harlalka, N., Suresh, S., Hawkins, R., Yang, S., Shah, D., Hu, J., & Rogers, T. T. (2024). Simulating Opinion Dynamics with Networks of LLM-based Agents. *Findings of NAACL 2024*, 3326-3346. <https://aclanthology.org/2024.findings-naacl.211/>
- Feng, S., Park, C. Y., Liu, Y., & Tsvetkov, Y. (2023). From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models. *ACL 2023*. (arXiv:[2305.08283](https://arxiv.org/abs/2305.08283))
- Hartmann, J., Schwenzow, J., & Witte, M. (2023). The political ideology of conversational AI: Converging evidence on ChatGPT's pro-environmental, left-libertarian orientation. (arXiv:[2301.01768](https://arxiv.org/abs/2301.01768))
- Motoki, F., Pinho Neto, V., & Rodrigues, V. (2024). More human than human: measuring ChatGPT political bias. *Public Choice*, 198(1-2), 3-23. <https://doi.org/10.1007/s11127-023-01097-2>
- Pew Research Center (2025). Support for expanding nuclear power is up in both parties since 2020. <https://www.pewresearch.org/short-reads/2025/10/16/support-for-expanding-nuclear-power-is-up-in-both-parties-since-2020/>
- Röttger, P., Hofmann, V., Pyatkin, V., Hinck, M., Kirk, H., Schütze, H., & Hovy, D. (2024a). Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models. *ACL 2024*. <https://aclanthology.org/2024.acl-long.816/>
- Röttger, P., Kirk, H., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024b). XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. *NAACL 2024*. (arXiv:[2308.01263](https://arxiv.org/abs/2308.01263))
- Rozado, D. (2024). The political preferences of LLMs. *PLOS ONE*, 19(7), e0306621. <https://doi.org/10.1371/journal.pone.0306621>
- Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). Whose Opinions Do Language Models Reflect? *ICML 2023*. (arXiv:[2303.17548](https://arxiv.org/abs/2303.17548))
- Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., et al. (2024). Towards Understanding Sycophancy in Language Models. *ICLR 2024*. (arXiv:[2310.13548](https://arxiv.org/abs/2310.13548))
- Taubenfeld, A., Dover, Y., Reichart, R., & Goldstein, A. (2024). Systematic Biases in LLM Simulations of Debates. *EMNLP 2024*. (arXiv:[2402.04049](https://arxiv.org/abs/2402.04049))
