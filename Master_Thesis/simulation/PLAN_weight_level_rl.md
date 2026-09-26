# Weight-Level Social-Feedback Learning: A Middle Ground

Could the social-feedback signal learn *inside the LLM's parameters* instead of in the external Q-layer? This note rejects the pure version as the main mechanism and specifies a middle ground that keeps the interpretability claim. **Nothing here is implemented**; it is an extension-chapter design note.

**Terminology.** "SFT" here always means Social Feedback Theory. The training method is written out as *supervised fine-tuning*; the obvious weight-level variant would otherwise be "SFT on SFT rewards".

---

## 1. The current architecture

- The **Q-layer is a two-armed bandit per agent**: actions {express +1, express −1}, policy `softmax(β · Q)`, learning `Q ← (1−α)·Q + α·r`. This *is* Banisch & Olbrich's process.
- The **LLM is a fixed conditional generator**, steered by the bandit through the prompt anchor; the expression gate forces it to emit the selected action.

"Learning in the model" means merging the two: the generation distribution becomes the policy, and social reward updates its weights.

## 2. The pure version and why it is not the main mechanism

Replace tabular Q with a per-agent LoRA policy `π_θ` over a frozen base (one shared model cannot hold twenty diverging opinions), keep `reward_from_expressions()` as the reward, and optimise `E[r]` by policy gradient (PPO/GRPO).

| # | Drawback | Why it disqualifies |
|---|---|---|
| D1 | Destroys interpretability | Opinion moves from a scalar Q into parameters; the Q-trajectory analysis has nothing left to regress |
| D2 | Severs the Banisch correspondence | The phase transition and the α/β mapping are exact only because the learning rule is theirs |
| D3 | Sample inefficiency | Tabular Q converges in a handful of interactions; policy gradient needs 10³–10⁶ samples, each a generation plus a classification |
| D4 | Sycophancy collapse | Rewarding agreement drives homogenised text and destroys the discourse corpus |
| D5 | Re-confounds model bias | The neutral Q initialisation keeps pretraining bias out of step 0; gradient steps re-entangle it |
| D6 | Unstable, unsweepable co-training | Every agent's reward is non-stationary, and a sweep becomes days of training |

Generative-agent work (Park et al. 2023) keeps weights frozen for these reasons.

## 3. The middle ground: periodic offline adapter distillation

> **The Q-trajectory stays the only opinion state and the only learning rule. A per-agent adapter only renders the Q-selected stance more fluently; it never carries the opinion.**

D1, D2 and the reward semantics are therefore untouched: `update_q_value`, `draw_softmax_stance`, `reward_from_expressions()` and the sweeps are unchanged.

**Mechanism: offline, periodic, per-agent, reward-filtered.** Every `ADAPTER_DISTILL_EVERY` rounds, per agent:

1. **Rejection-sampling fine-tuning (ReST, the default):** fine-tune the agent's LoRA on its accepted, measured-reward turns with `reward_a ≥ ADAPTER_REWARD_FLOOR` and stance = current argmax.
2. **DPO** on reward-ranked pairs from matched contexts: more stable than PPO, costlier to assemble.

Online PPO/GRPO is out of scope (D3, D6).

**Configuration (proposed, default-off like `GRAPH_DYNAMIC`):**

| Setting | Default | Description |
|---|---|---|
| `ADAPTER_DISTILL_ENABLED` | `False` | `False` is the current frozen-generator architecture exactly |
| `ADAPTER_DISTILL_EVERY` | `10` | Rounds between passes; ≫ `REFLECT_EVERY` |
| `ADAPTER_METHOD` | `"rest"` | `"rest"` or `"dpo"` |
| `ADAPTER_REWARD_FLOOR` | `0.3` | Minimum reward for a training turn |
| `ADAPTER_MAX_PAIRS` | `64` | Training examples per agent per pass |

**The experiment it enables**, parallel to the gated/ungated contrast: *does internalising social experience into expression change the macro-dynamics, with opinion learning held fixed?* Predictions:

1. First-attempt fidelity rises across distillation periods (the gate fires fewer retries), the signature that internalisation happens at all.
2. Lexical divergence between the communities grows at fixed Q-gaps: polarization the scalar layer cannot represent.
3. Possibly a shifted consensus threshold on the `SBM_M_INTER` ladder, if entrenched expression feeds back into reactions.

**Residual risks.** The adapter could drift into carrying opinion: the reward floor and the argmax filter admit only Q-consistent turns, and the gate records flag first drafts that start contradicting the argmax. Sycophancy: offline filtering on accepted turns, capped and KL-anchored, cannot chase a live gradient. Pretraining bias: the base stays frozen and shared, so the neutral Q start holds. Reproducibility: every training set is reconstructable from `events.jsonl`.

## 4. Infrastructure

Ollama does no backprop: distillation needs `transformers` + `peft` + `trl` and the GPU memory to fine-tune adapters over the base model. Twenty LoRA adapters over one frozen base is tractable because training is periodic and offline. vLLM serves per-request LoRA directly; Ollama's adapter support would need validating.

## 5. Recommendation

Do not make weight-level RL the main mechanism: it trades away the auditability the thesis exists to show (D1–D2) and is unstable on top (D3–D6). The middle ground is a candidate extension arm behind `ADAPTER_DISTILL_ENABLED = False`. It pays only the part of the auditability price, expression rather than opinion, that the interpretability claim can afford.

---

## References

*Recalled from memory; verify year, venue and title before citing.*

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv:2212.08073*.

Banisch, S., & Olbrich, E. (2019). Opinion polarization by learning from social feedback. *The Journal of Mathematical Sociology, 43*(2), 76-103.

Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS 2017*.

Dong, H., et al. (2023). RAFT: Reward rAnked FineTuning for generative foundation model alignment. *TMLR*.

Gulcehre, C., et al. (2023). Reinforced self-training (ReST) for language modeling. *arXiv:2308.08998*.

Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*.

Jacob, D., & Banisch, S. (2023). Polarization in social media: A virtual worlds-based approach. *JASSS, 26*(3), 11.

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*.
