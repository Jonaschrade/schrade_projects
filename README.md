# schrade_projects

Data science projects and coursework from my Master's program. The larger, self-contained projects are listed first; the remaining folders collect smaller course assignments.

## Main projects

### [master_thesis](master_thesis/)
Master's thesis (in progress): *Does Social Feedback Cause Structural Polarization Among LLM Agents?* Tests whether LLM agents learn from social feedback as Social Feedback Theory (SFT; Banisch & Olbrich 2019) describes, and whether that learning explains how their networks end up.
- **[simulation/](master_thesis/simulation/):** LLM agents with only a name and a memory exchange messages on SFT's two-community network, running on local models via Ollama. They declare one of two positions from memory, argue it with ε-greedy exploration, and read their own and their partner's messages blind, which yields the perceived agreement without ever showing it to them. Memory shows either both turns of each exchange (`full`) or the agent's own turn only (`own`). Includes a sweep runner and batch scripts for the university's compute cluster.
- **[analysis/](master_thesis/analysis/):** fits four models (prior, habit, SFT's rule, habit plus SFT) to the declared positions by maximum likelihood, with a preregistered test of whether social feedback beats habit on declarations and on network outcomes. Also includes an exchange audit, SFT and assimilation benchmarks, a surrogate simulator for parameter and model recovery, and robustness tables.

This is a snapshot of the thesis repository's working branch.

### [deep_learning](deep_learning/)
Assignments and final project for a deep learning course.
- **Final project:** predicting income inequality (GINI coefficients) for countries with missing data using country similarity networks and graph neural networks, with Optuna hyperparameter tuning (see `final_project/DLSS_Project_Report.pdf`).
- **Assignments 1–4:** tabular learning on job vacancy data, satellite image classification (EuroSAT), graph learning on the Twitch gamers network, and further tasks — each with notebook and PDF write-up.

### [statistical_learning](statistical_learning/)
Final project for the Statistical Learning course (Winter 25/26): predicting FIFA player potential from age, physical, and cognitive attributes. Compares a neural network against lasso regression with nested cross-validation, evaluated via MSE/MAE (Quarto report in `report.pdf`).

### [social_media_data_project](social_media_data_project/)
End-to-end social media data analysis project: data collection, LLM-assisted labeling, and statistical analysis (bootstrap comparisons, engagement distributions) of YouTube video and user activity, with a full report (`SMDA_project.pdf`) and visualizations.

## Course assignments

- **[general_data_assignments](general_data_assignments/)** — four assignments from "Introduction to Computation for the Social Sciences": election prediction data, time series, tweet text processing, and SQL/database queries.
- **[sentiment_analysis](sentiment_analysis/)** — sentiment and emotion classification on Twitter/YouTube comments and the ISEAR dataset, including manual and model-based labeling.
- **[social_impact_analysis_twitter_youtube](social_impact_analysis_twitter_youtube/)** — analysis of US Congress tweets and YouTube channel data.
- **[trends_data_analysis](trends_data_analysis/)** — comparing Google Trends search interest with CDC ILINet influenza surveillance data.
