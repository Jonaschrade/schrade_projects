
\section{Motivation}
The general scope of the thesis revolves around polarization as a phenomenon of collective behavior among Large Language Model (LLM) agents in a social network graph. This topic carries both internal and external relevance. Internally, as pre-print platforms see a recent surge of studies investigating the dynamics and outcomes of collectively interacting AI agents, there is a particular need for contributions grounded in classical agent-based modeling and network theory that extend, rather than replace, established approaches. Externally, LLM-supplemented network simulation bears relevance beyond the academic sphere, since structural polarization remains a phenomenon of considerable societal concern whose underlying mechanisms warrant continued investigation using state-of-the-art methods.

Accordingly, this thesis pursues two central goals. First, it offers an innovative, dynamic extension of the classical agent-based model of Social Feedback Theory (SFT; \cite{Banisch2019}) by introducing LLM agents and network rewiring, thereby advancing understanding of the mechanisms of social-feedback-driven polarization and the societal implications that follow. Second, it contributes to the growing body of LLM behavioral research by shedding light on the black-box nature of these models and its implications for their collective behavior.

\section{Background}
The term Agent-Based Modeling (ABM) denotes a process-oriented, bottom-up simulation method in which populations of autonomous agents follow simple, locally specified decision rules, and from whose interactions higher-level patterns emerge that no single agent intends or perceives. In doing so, the approach preserves individual variation, spatial or network structure, and path dependence. It has thus become central to social-behavioral research, since ABM studies show how, and under what structural conditions, collective patterns arise from individual behavior rather than merely describing an aggregate end-state \parencite{Flache2017,Goldstone2005}.

As such, SFT reflects a conceptual branch of ABM that was first proposed by \textcite{Banisch2019}, with further enrichments by \textcite{Banisch2022}, \textcite{Jacob2023}, and \textcite{Banisch2026}. The framework models opinion expression as a reinforcement learning (RL) problem in which agents hold internal Q-values for two opposing opinions, $o \in \{-1, +1\}$. Agents express the higher-valued stance to their neighbors, receiving agreement or disagreement as a reward signal depending on each neighbor's own preferred stance. Q-values are updated iteratively across expression rounds until the system converges to an equilibrium. Although deliberately parsimonious in its assumptions, this mechanism generates stable bi-polarization, spiral-of-silence, and echo-chamber dynamics without invoking the assumptions of classical polarization models, such as negative influence, bounded confidence, or opinion homophily.

Furthermore, SFT provides an explicit, defensible micro-macro link grounded in social neuroscience at the micro level and mean-field and equilibrium analysis at the macro level, which makes it possible to isolate the structural effects of the network: \textcite{Banisch2019} show that connectivity drives consensus, whereas polarization requires sufficient community structure. Subsequent work demonstrates how the same reward-learning core, combined with different structural overlays, reproduces a range of empirically observed collective phenomena, such as the spiral of silence \parencite{Banisch2022} and platform segregation \parencite{Banisch2026}.

While these results strengthen the model's internal validity, the parsimony of its underlying assumptions remains a core limitation of the Banisch tradition. Binary opinion stances, homogeneous rewards, and single scalar feedback serve simplistic modeling purposes well but arguably fail to capture the diversity and gradation of human cognition and communication, thereby limiting the external validity of these results when transferred to real-world applications.

This argument is supported by \textcite{Sarkozy2022}, who provide experimental evidence for SFT's claims, demonstrating that social feedback can influence not only public expression but even privately held opinions. Notably, whereas classical SFT treats feedback as a single summand within the reward function, they find that ambivalent social signals are the most influential drivers of opinion change, prompting respondents to shift toward stronger disagreement. This directly contradicts the SFT prediction that mixed feedback drives agents toward moderation by equalizing the Q-values of competing stances.

The limitations of SFT reflect a broader issue: the generalizability of ABMs is constrained by design. Impoverished agent cognition, weak empirical calibration, and limited cross-model comparability have prompted calls across the field for richer agent architectures and tighter coupling between simulation and experiment \parencite{Flache2017,Goldstone2005} -- calls that recent developments in the LLM research community appear to heed.

Since the generative-agents architecture of \textcite{Park2023} first demonstrated that LLMs equipped with a memory–reflection–planning cognition could sustain coherent multi-agent social dynamics, the literature integrating LLMs into ABMs has expanded rapidly. The body of work that followed has shown that LLM-based agents can spontaneously reproduce fundamental social phenomena without these being explicitly programmed. Across this literature, such phenomena include coordination and (mis-)alignment \parencite{Bellina2026,DeMarzo2025}, endogenous network formation \parencite{DeMarzo2023,Schneider2025}, (mis-)information and belief propagation \parencite{Chuang2024,Liu2024}, the emergence of shared norms and conventions \parencite{Ashery2025,Cordova2024}, and homophily, polarization, and echo chambers \parencite{Donkers2025,Ferraro2024,Ohagi2024,Piao2025,Sakurai2025,Wang2025}.

As LLM agents replace hand-crafted update rules with natural-language reasoning, persona-conditioned behavior, and richly contextualized interaction, the LLM-ABM paradigm relaxes several of the most restrictive assumptions on which opinion-dynamics models and classical ABMs have long depended. Yet this same flexibility produces a methodological tension: the mechanism driving any given opinion update resides within the LLM forward pass and is therefore largely opaque. Emergent macro-patterns can be observed but not readily attributed to identifiable micro-processes, leaving the field with rich phenomenology but limited mechanistic explanation.

This is precisely where a methodological synthesis offers a productive complement. By providing an explicit, parsimonious, and analytically tractable micro-mechanism, namely reward-driven reinforcement learning over expressed opinion stances, together with an established link from individual learning to collective regimes, SFT supplies exactly the mechanistic traceability that LLM-ABMs currently lack. LLM-based agents, in turn, contribute the natural-language expression, graded and ambivalent feedback, and heterogeneous personas absent from SFT in its canonical form. Integrating the two addresses both sides' core limitations, grounding the LLM-ABM paradigm in falsifiable opinion-dynamics theory while extending SFT beyond the binary, homogeneous-reward setting of its original formulation and assessment.

\section{Proposed Methods}
\subsection{Gap + Research Questions} 
The methodological gap can thus be characterized as follows. LLM-ABM reproduces social phenomena but lacks a falsifiable micro-mechanism, whereas SFT provides one but cannot accommodate natural-language expression, ambivalent feedback, or heterogeneous agent personas. From this tension, the \textbf{overall research question} is formulated:

\begin{quote}
Can Social Feedback Theory serve as the underlying reinforcement-learning mechanism for LLM agents embedded in a structured social network such that the resulting hybrid model preserves SFT's analytic interpretability while extending the empirical scope?
\end{quote}

By mirroring the SFT of the Banisch canon as closely as possible, a tangible replication sub-question is derived.
\begin{quote}
    \textbf{RQ1:} On a static community-structured network, does SFT-governed LLM-ABM reproduce the theory's macro-dynamics, specifically the modularity-driven phase transition between consensus and stable bi-polarization, and do LLM-generated feedback signals yield the same Q-value learning trajectories as the canonical binary reward rule?
\end{quote}
It is important to note that the research body defining SFT treats the agentic network as exogenous and static, itself unaffected by the feedback mechanism and the resulting polarization dynamics. For additional empirical novelty, this work departs from that assumption. Following recent findings in the LLM-ABM literature on endogenous network formation \parencite{DeMarzo2023,Schneider2025} and spontaneous homophily and hub emergence \parencite{Piao2025}, the network is instead treated as a learned outcome. Agents learn not only what to express but also whom to interact with, with both processes governed by the same SFT reward signal. This empirical augmentation manifests as second sub-question:

\begin{quote}
\textbf{RQ2:} When the interaction graph itself is endogenous, i.e., tie formation and survival are governed by the same social-feedback reward signal that drives opinion expression, does the coupled (network $\times$ opinion) mechanism still produce SFT's regimes? Under what conditions does it generate structural patterns (homophily, hub emergence, community modularity) reported in recent LLM-ABM literature? 
\end{quote}


\subsection{Simulation}

The ABM simulation\footnote{Source code: \url{https://github.com/Jonaschrade/thesis/tree/main/simulation}} is initialized with a stochastic block model of $n = 20$ LLM agents in two blocks of ten. Instead of edge probabilities, the model fixes edge counts: 27 of the 45 possible within-block ties per block (density 0.6) and a varied number of the 100 possible cross-block ties, from zero to 60 (where both densities are equal), which forms the phase-transition axis. Each run discusses one topic, framed as two opposing propositions (primarily ``nuclear power does more good than harm'' and ``nuclear power does more harm than good''), and all agents run on one locally hosted LLM (Qwen3.8, 27B), with two further models as a sensitivity check. Each agent holds an internal state comprising
\begin{enumerate}[label=(\arabic*)]
    \item a persona prompt generated from a randomly drawn respondent record of the 2024 US General Social Survey, restricted to 91 demographic, socioeconomic, and background variables; attitude and opinion items are excluded so that no stance leaks into the persona,
    \item an opinion state that implements Banisch's dual Q-value mechanism reflecting the expected reward of expressing either of the two propositions,
    \item and a personal Chroma vector database of past interactions and reflections that is updated and queried in alignment with the memory stream for generative agents as proposed by \textcite{Park2023}, with retrieval additionally favoring memories congruent with the agent's current stance.
\end{enumerate}

In each interaction, an expressing agent is drawn uniformly from the set of agents with at least one connecting edge. From its connected neighbors, a responder is also drawn uniformly. Using a softmax function with inverse temperature $\beta$

\begin{equation}
    P(o^{+}) = \frac{\exp\bigl(\beta\, Q(o^{+})\bigr)}{\exp\bigl(\beta\, Q(o^{+})\bigr)+\exp\bigl(\beta\, Q(o^{-})\bigr)}
\end{equation}

the stances on the topic (the proposition $o^{+}$ or its opposite $o^{-}$) are stochastically drawn for both discussing agents. Before the first interaction, the Q-values of all agents are zero and thus, expressed stances manifest completely at random. After that, deviations of the expressed from the preferred stance are regulated by the inverse temperature $\beta$. In the confirmatory runs, $\beta \to \infty$: agents express their higher-valued stance deterministically, ties are broken by a fair coin, and finite $\beta$ is examined in a separate sensitivity arm.

The expresser is prompted to respond to an external moderator statement\footnote{After the first exchange, the expresser responds to the neighbor's last message of their previous interaction.} with its name, persona, relevant memories, and current stance on the topic injected. The prompt also states the agent's conviction, i.e., the magnitude of its Q-gap, in four verbal bands (from ``tentatively'' to ``with full conviction''), so that expressed commitment can track learned commitment. The responder reacts to the expresser's utterance in the same way.

Because a persona-conditioned LLM may argue against the stance it was assigned, every draft passes an expression gate before entering the conversation. A draft whose classified stance contradicts the drawn one is rejected and resampled: up to five times with the full prompt, up to five times without persona and memories, and finally replaced by a fixed position statement. Drafts that take neither side pass as strategic abstentions, which freeze the Q-update and edge dynamics of that interaction. The rejected drafts are logged and quantify the persona's pull against the learned stance.
Each turn is then graded by a separate, minimal classifier call, detached from persona, memory, and transcript, which judges the text against the two propositions and returns a graded stance $g \in \{0, \pm\tfrac{1}{3}, \pm\tfrac{2}{3}, \pm 1\}$. The reward is the product of both grades,
\begin{equation}
    r = g_{\text{expr}} \cdot g_{\text{resp}} \in [-1, 1]\text{,}
\end{equation}
the graded generalization of the canonical binary product of expressed opinions, $r = o_u\, o_v$ with $o_u, o_v \in \{-1, +1\}$ \parencite{Banisch2019}, which it recovers at $\pm 1$. Neutral feedback ($r = 0$) is applied, whereas an off-topic or unscorable reaction skips the update, so neutral and absent feedback stay distinct. The Q-value of the expressed stance $o$ is accordingly updated only for the expresser, following the Q-learning update, which for this single-state problem reduces to an exponential moving average of received rewards:

\begin{equation}
    Q_{t+1}(o) \leftarrow (1-\alpha)\, Q_t(o) + \alpha\, r_t\text{,}
\end{equation}
following \textcite{Banisch2019}, with learning rate $\alpha = 0.1$.

During each discussion round, a number of agents equal to the total network size is selected as expressers, yielding one expected interaction loop per agent per round. While the classical Banisch approach models this loop as a single expression followed by a single response, extending it to a longer, self-contained exchange, in which only the responder's final message is propagated to the reward evaluation, would be straightforward. Every $k = 5$ discussion rounds, agents are prompted to reflect, implementing a memory retrieval and synthesis mechanism in the sense of \textcite{Park2023} and promising richer, more coherent, and more realistic agent behavior. The simulation terminates after $m$ discussion rounds, a horizon fixed by a dedicated screen. Personas, discussions, reflections, and per-round snapshots of the network and agent states are logged throughout.

\subsection{Analysis}
To assess the simulation outcome and answer above research questions by testing derived hypotheses, polarization metrics and opinion diffusion parameters on three levels are proposed as replication and extension of the Banisch canon.
\subsubsection{Micro / Individual}
\begin{itemize}
    \item \textbf{Individual Q-value trajectories}. Evolution and final state of the Q-value pair quantifies an agent's learned valuations of each opinion stance. Combined with the preferred/expressed opinion and the agent's conviction, as operationalized by the gap between the two Q-values, the metrics provide insight on whether agents are radicalizing, oscillating, or settling.
    \item Extensions: \begin{itemize}
        \item \textbf{Expression as high-dimensional object}. Posts can be embedded and tracked over time, which allows for measurement of semantic drift throughout the simulation.
        \item \textbf{Received reward}. LLM-generated feedback is graded and can be ambivalent; variance and skewness of received rewards enable insights on the mixed-feedback anomaly as suggested by \textcite{Sarkozy2022}.
        \item \textbf{Persona field}. The share of first drafts rejected by the expression gate, per agent, stance, and round, measures the persona's resistance to the learned stance (expression friction).
    \end{itemize}
\end{itemize}
\subsubsection{Meso / Local}
\begin{itemize}
    \item \textbf{Local opinion climate}. Fraction of in-group vs. out-group ties per agent as well as the local agreement rate, i.e. the proportion of an agent's interactions that yield positive feedback, quantify how an agent perceives its immediate neighborhood. Furthermore, \textcite{Jacob2023} turn to the incongruent-links percentage, the share of ties connecting agents with opposing expressed opinions, as operational distinction between structural and unstructural polarization.
    \item Extensions: \begin{itemize}
        \item \textbf{Local semantic coherence}. Within-neighborhood similarity of expressed opinions (via embedding distance) captures whether agents in the same community converge linguistically as well as positionally. 
        \item \textbf{Tie-level reward statistics}. Mean and variance of reward flows across edges provide descriptive metrics on local social feedback dynamics.
    \end{itemize}
\end{itemize}
\subsubsection{Macro / Global}
\begin{itemize}
    \item \textbf{Phase transition between consensus and stable bi-polarization}. Global metrics such as overall opinion dispersion and mean conviction distinguish network polarization from consensus, and neutral consensus from extremized consensus, respectively. Regimes are classified per replicate as consensus on $o^{+}$, consensus on $o^{-}$, or bi-polarization; the main dependent variable is the regime fraction over replicates along the coupling axis, summarized by a bootstrapped transition point. Every LLM result is read against matched null models, i.e., the canonical SFT mechanism simulated without the LLM, fed with binary rewards or with graded rewards drawn from the runs' own events.
    \item Extensions: See Section~\ref{sec:dynamic_graph}
\end{itemize}

\subsection{Extensions}
\subsubsection{Dynamic Graph}\label{sec:dynamic_graph}

In order to address RQ2, endogeneity needs to be introduced to the network structure as a function of the same SFT reward mechanism that also governs agent stances. This dynamic is implemented by adding two additional steps to the interaction loop:
\begin{enumerate}
    \item The \textbf{Edge Update} rule controls edge survival with a unilateral assessment of the communication channel, quantified by each agent's perceived strength of the connection to its neighbor. At the beginning of the simulation, each edge is initialized with a strength of 1.0 for every agent. After the expresser received the social reward of the current interaction and updated its Q-values, the reward is mean-aggregated in a sliding window with the previous $j-1$ rewards received via the assessed edge ($j = 5$). Agent $u$'s strength $s_{uv}$ of edge $(u,v)$ is then updated using
    \begin{equation}
        s_{uv,t+1} \leftarrow s_{uv,t} + \delta\, \bar{r}_{uv,[t-j+1,\,t]}\text{,}
    \end{equation}
    where $\bar{r}_{uv,[t-j+1,\,t]}$ represents the average of the current and last $j-1$ rewards received via the edge and $\delta$ the valuation rate. Valuations are clamped to $[0, 3]$, and $\delta$ is not set freely but derived from the ratio $w$ of the opinion to the rewiring timescale, $\delta = w\,\alpha\,s_0$ with initial valuation $s_0 = 1$, so that a single dial governs how fast structure moves relative to opinions. As soon as the strength reaches zero for either of the connected agents, the communication channel is interrupted and the edge removed.

    \item \textbf{Rewiring} replaces a severed edge in the same step: the agent whose valuation reached zero opens a new tie (initial valuation 0.5), so the number of edges is conserved and falling density cannot fragment the network on its own. Two rules are compared. \emph{Random} rewiring connects to any non-neighbor. \emph{Reward-local} rewiring restricts candidates to second-degree neighbors, values each by the reward chained through mutual contacts,
    \begin{equation}
        \hat{r}_{uv} = \frac{1}{\lvert N(u) \cap N(v) \rvert} \sum_{c \in N(u) \cap N(v)} \bar{r}_{u \leftarrow c}\, \bar{r}_{c \leftarrow v}\text{,}
    \end{equation}
    where $\bar{r}_{u \leftarrow c}$ is the mean reward $u$ received from $c$, and selects one with probability proportional to $\exp(\beta_{\text{rw}}\, \hat{r}_{uv})$. Since every reward-local tie closes a triangle, comparing both rules at $\beta_{\text{rw}} = 0$, where neither scores partners, isolates the effect of triadic closure. Isolated agents are not reconnected, so attrition remains observable.
\end{enumerate}
Once tie formation and dissolution are endogenous, modularity and community assortativity become the primary structural outcomes for investigating the micro-macro relationship between social feedback and structural polarization, complemented by clustering, the share of cross-block ties, and degree dispersion and assortativity as indicators of hub formation.
\subsubsection{Partner Choice}
\textcite{Jacob2023} introduce the concept of probabilistic homophily for selecting the responder from an expresser's set of neighbors. Empirically, the mechanism is rooted in humans' preference for like-minded interaction. The authors operationalize this phenomenon as a weighted probability of interaction between expresser $u$ and neighbor $v$, $w_{uv} = \exp\bigl(-h\, \lvert \Delta Q_u - \Delta Q_v \rvert\bigr)$, where $h$ represents the homophily control parameter and $\Delta Q = Q(o^{+}) - Q(o^{-})$ the gap in Q-values of both stances.

Instead of operationalizing homophily as a weight function of the inherently private gap in Q-values, or implementing a matching weight based on `perceived homophily', this thesis weights the interaction by experience,
\begin{equation}
    w_{uv} = \exp\bigl(\beta_{\text{sel}}\, \bar{r}_{u \leftarrow v}\bigr)\text{,}
\end{equation}
where $\bar{r}_{u \leftarrow v}$ is the mean reward the expresser previously received from neighbor $v$. Since partner choice thus never reads opinion states, echo chambers can emerge through reinforcement alone, without reintroducing the homophily assumption that SFT dispenses with. The confirmatory runs keep $\beta_{\text{sel}} = 0$, i.e., uniform responder choice. As a robustness check, $\beta_{\text{sel}}$ is varied from 0 to 10 at every coupling level in the null model, which runs the identical partner-choice mechanism on graded rewards drawn from the LLM runs, so that its effect on within- versus cross-community reward can be assessed without additional LLM runs.

\printbibliography[]

\end{document}
