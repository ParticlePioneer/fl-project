# Project Decisions Log

Every crucial decision made on this project, with the reasoning behind it, logged
in chronological order. Purpose: so neither of us has to re-derive "why did we do
it this way" three weeks from now, and so the README/report writeup has a ready
source to draw from.

---

## 1. Project selection: Federated Learning over Quantum-Inspired Optimization
**Decision:** Pursue "Federated Learning for Privacy-Preserving Prediction" instead
of "Quantum-Inspired Optimization for Recommendations."
**Reasoning:** FL has mature, well-documented libraries (Flower, Opacus), making an
8-week timeline low-risk with a high probability of solid results. The quantum
project has more novelty upside but higher risk of an inconclusive or negative
result given QAOA's sensitivity to tuning on simulators.

## 2. Domain: Finance (credit risk) over Healthcare
**Decision:** Frame the project around credit risk / loan approval prediction
across simulated banking clients, not disease prediction across simulated
hospitals.
**Reasoning:** Finance is a more differentiated portfolio story (most student FL
projects default to healthcare), and more locally resonant given Pakistan's
fintech data-localization pressure. Healthcare was the "safer" choice for
citation density, but finance was chosen deliberately for differentiation.

## 3. Positioning: Learning exercise, not a novelty/research claim
**Decision:** The project's goal is hands-on experience in a relatively
underexplored area of the field -- not to challenge, outperform, or position
against existing published research (e.g. Naresh & Ayyappa 2026).
**Reasoning:** Explicitly stated by the user. This changes framing throughout:
related work is cited as validation that the setup is realistic and worth doing,
not as a competitor to be "beaten." Statistical rigor and strict epsilon values
are pursued because doing the experiment properly is itself the skill being
built, not because the project needs to outperform prior work.
**Impact:** All "how we differ from Naresh & Ayyappa" framing in earlier planning
was softened to "informed by and extending" language for the README/report.

## 4. Adaptive/personalized noise: stretch goal, not core pillar
**Decision:** Do not build a custom adaptive/personalized DP noise mechanism as
part of the core 8-week scope. Keep the core scope to: strict epsilon-sweep,
statistical rigor (seeds + confidence intervals), fairness analysis, and
production-shaped engineering (Flower, Docker, MLflow).
**Reasoning:** Adaptive/personalized DP noise is an actively published,
crowded research subfield (APDP-FL, FedADDP, DP-pFedDSU, and others,
2024-2026) -- designing a genuinely new mechanism is out of scope for 2 months
on top of everything else already planned. Implementing one existing published
adaptive method as a bounded comparison arm is feasible as a Week 6-7 stretch
goal, but is not a headline pillar.
**Impact:** Architecture is designed so this can be added later without a
redesign (see Decision 5).

## 5. Privacy layer architecture: `NoiseStrategy` abstract interface
**Decision:** Build the privacy/noise layer behind an abstract `NoiseStrategy`
interface with a `FixedNoiseStrategy` implementation for the 8-week core
scope, rather than hardcoding a flat epsilon value directly into the training
loop.
**Reasoning:** This is the single architectural decision that determines
whether adding adaptive/personalized noise later (Decision 4) is a drop-in
swap or a full rewrite. `client.py` calls `strategy.get_epsilon(metadata)`
instead of reading a flat config value -- so an `AdaptiveNoiseStrategy` can be
added later by writing one new subclass, with zero changes to client/server
code.
**Consequence:** `configs/base.yaml`'s `privacy` section is structured with a
`strategy` selector (currently only `fixed` is implemented) rather than a flat
`epsilon` field. Per-client metadata (dataset size, round number, client id) is
plumbed through to the privacy layer from day one via `ClientMetadata`, even
though `FixedNoiseStrategy` ignores most of it -- retrofitting this plumbing
later would be the expensive part to add after the fact.

## 6. FL framework: Flower (`flwr`), not a hand-rolled FedAvg loop
**Decision:** Use Flower for federation orchestration instead of writing a
custom `for client in clients: train()` simulation loop.
**Reasoning:** Flower separates ML code from the communication layer via
gRPC, so the same client/server code used in local simulation can later point
at real physical machines without rewriting training logic. This was a
requirement once the project was reframed as a portfolio piece meant to be
extensible, not a disposable class exercise.

## 7. Config-driven design, nothing hardcoded
**Decision:** Client count, number of rounds, local epochs, epsilon values,
dataset path, model type, and target/partition/fairness column names all live
in `configs/base.yaml`, not in code.
**Reasoning:** Supports both scalability (Decision 6) and the statistical
rigor requirement -- running the epsilon-sweep across multiple values and
seeds needs to be a config change, not a code change, to be practical within
the timeline.

## 8. Non-IID partition axis: US state (`addr_state`)
**Decision:** Simulate 5 "banks" by partitioning the dataset by
`addr_state`, rather than an artificial random skew.
**Reasoning:** Real regional lending data has genuine, defensible statistical
skew (some states have higher default rates, some have far higher loan
volume) -- this is a realistic non-IID scenario rather than a synthetic one,
which is more defensible in the methodology writeup.

## 9. Fairness axis: income bracket (`annual_inc_bracket`), not demographic parity
**Decision:** Use income bracket (and secondarily loan purpose) as the
fairness subgroup axis, rather than gender/demographic parity as in
Naresh & Ayyappa 2026.
**Reasoning:** The chosen dataset (LendingClub) does not include demographic
fields by design (avoids fair-lending data collection issues), so gender-based
fairness analysis isn't directly replicable on this data. Income-bracket
fairness is arguably a more directly actionable question anyway: does privacy
noise disproportionately hurt prediction accuracy for lower-income applicants,
who can least afford a bad credit decision.

## 10. Dataset: LendingClub subset over Kaggle Loan Prediction or German Credit
**Decision:** Use a LendingClub subset (~20-50k rows, 2015-2018 originations)
as the primary dataset.
**Reasoning:** German Credit (UCI, ~1,000 rows) is heavily overused in ML
coursework to the point of being a portfolio red flag. Kaggle Loan Prediction
(~600 rows) is too small to survive a 5-10 client non-IID split without some
clients having under 100 samples, which would undermine the statistical rigor
that's a core project pillar. LendingClub offers real scale and a genuine
`addr_state` field for the non-IID partition axis (Decision 8).

## 11. Model choice: logistic regression first, MLP as secondary comparison
**Decision:** Build the centralized baseline and all FL/DP experiments with
logistic regression first; only add an MLP as a secondary comparison if time
allows.
**Reasoning:** Matches the literature (Naresh & Ayyappa and others also use
logistic regression), is interpretable, fast to iterate, and avoids
introducing an extra variable (model capacity) that would make the
epsilon-sweep results harder to attribute cleanly to privacy noise alone.

## 12. Client count: 5 clients to start
**Decision:** Simulate 5 banking clients as the primary configuration, with
10 clients as a possible robustness variant in Week 6 if time allows.
**Reasoning:** Matches Naresh & Ayyappa's setup, making any comparison to
that work more direct if referenced in the literature review, without
committing to it as a competitive benchmark (see Decision 3).

## 13. Epsilon sweep range: includes strict privacy (ε ≤ 4)
**Decision:** The epsilon sweep for the core experiment includes strict
values (e.g. 0.5, 1, 2, 4) in addition to looser values, rather than stopping
at the weaker range (ε ≥ 5) used in most related work found (EPFL 2024 report
used ε up to 100; Naresh & Ayyappa 2026 used ε ≥ 5.74).
**Reasoning:** Nobody in the reviewed related work tests what happens to
fairness and accuracy at genuinely strict privacy budgets on this kind of
data -- this is the most defensible, interesting open question to explore,
consistent with Decision 3 (learning-oriented, not competitive) since it's
about doing the experiment thoroughly rather than "winning."

## 14. Dev environment: local scripts, not Jupyter/Colab, for core FL code
**Decision:** Core FL code (client/server/model/privacy modules) is written
as plain Python in `src/`, run via terminal -- not built or run inside Jupyter
notebooks or Google Colab.
**Reasoning:** Flower's simulation mode runs multiple concurrent client/server
processes, which fights against a notebook's single-kernel linear execution
model. Colab additionally can't run a local Docker daemon, which is required
for the Week 5 containerization step. Notebooks are still used, but only for
EDA and one-off exploratory plotting (see repo `notebooks/` folder).

## 15. Class imbalance handling: `class_weight="balanced"` + PR-AUC as primary metric
**Decision:** Use `class_weight="balanced"` in the logistic regression
classifier (and require the same treatment in every FL client's local
training later), and track PR-AUC and F1 as primary metrics rather than
accuracy or ROC-AUC alone.
**Reasoning:** First baseline run on synthetic data (89.28% accuracy, 0.93%
recall) showed the model was just predicting the majority class -- a classic
symptom of unhandled class imbalance, which is expected in credit default
data (defaults are always the minority class). Accuracy and even ROC-AUC can
look deceptively good in this situation; PR-AUC is the more honest metric
for imbalanced binary classification and will be used as primary going
forward, including in the Week 5 epsilon-sweep and Week 3 fairness analysis.
**Consequence:** Every FL client's local training loop (Week 2) and the
DP-SGD integration (Week 4) must carry the same imbalance handling
consistently -- otherwise fairness comparisons across income-bracket
subgroups (Decision 9) risk being confounded by inconsistent imbalance
handling rather than reflecting genuine privacy-driven disparity.

## 16. Synthetic data stub built to unblock development
**Decision:** Built `src/data/synthetic.py`, a synthetic dataset generator
matching the real dataset's expected schema (`annual_inc`, `loan_amnt`,
`int_rate`, `dti`, `emp_length`, `addr_state`, `purpose`, `loan_status`,
`annual_inc_bracket`), including a deliberately skewed state distribution.
**Reasoning:** Allows the baseline model pipeline, and later the partitioning
and FL pipeline, to be built and tested before the real cleaned LendingClub
CSV is delivered. Since column names match exactly, switching from synthetic
to real data is a one-line config change (`processed_path` in
`configs/base.yaml`), not a code change.

## 17. Decision threshold left at default 0.5, not tuned
**Decision:** Keep the classification decision threshold at the default 0.5
for the centralized baseline and all downstream FL/DP experiments, rather
than tuning it for asymmetric misclassification costs (false positive =
rejecting a good borrower vs. false negative = approving a defaulter).
**Reasoning:** After adding `class_weight="balanced"` (Decision 15), the
first real baseline run on synthetic data showed accuracy drop from 89.3%
to 66.2% while recall rose from 0.93% to 67.4% -- the expected and correct
trade-off, confirmed by ROC-AUC staying unchanged (0.7226 in both runs,
since `class_weight` shifts the decision threshold's effective operating
point rather than the model's underlying ranking ability). Threshold tuning
for asymmetric costs is a legitimate real-world consideration for credit
risk, but introducing it now would add another variable that could confound
comparisons across the epsilon-sweep (Decision 13) and fairness analysis
(Decision 9) -- every FL/DP run needs to be compared on a consistent basis.
**Consequence:** Documented explicitly as a stated limitation/future-work
item in the final report, not left as an unstated gap. Threshold stays fixed
at 0.5 across every experiment for the full 8-week core scope.