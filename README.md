# Theory-as-Channel

Treat ensemble/student predictions as a **communication channel**: use error correction and channel capacity (from ML-Toolbox).

## What's standard vs what's distinctive

**Not different from the rest of the world**

- Train/val/test, weights from validation only, report on test – **standard ML practice**. Same idea as any serious toolkit or course.
- Majority vote, weighted voting by accuracy, soft voting (average `predict_proba`) – **sklearn and others already have this** (e.g. `VotingClassifier` with `weights=`, `voting='soft'`).
- Caveats about test-set reuse and selection bias – **well known** in the community; we’re just making them explicit here.
- “Improve with less” / avoid method proliferation – **good practice**, but not a unique feature.

So if you only do the “recommended” pipeline with those three combiners, you’re doing **rigorous, standard ML**. Nothing in that pipeline is novel.

**What is different**

- **Framing**: Treating the ensemble explicitly as a **communication channel** (Shannon, capacity \(C = B \log_2(1 + S/N)\), error correction as decoding). Most ensembles don’t use that language or tie weights to capacity.
- **Capacity-weighted voting**: Weights proportional to \(\log_2(1 + p/(1-p))\) from accuracy – a **theory-driven** option (channel capacity as a function of “success rate”). Unusual in standard libraries.
- **One small bundle**: Communication-theory + information-theory (entropy, mutual information, KL) + channel view + a few optional combiners in one place, with the **“theory as channel”** narrative. The building blocks exist elsewhere; this project packages them around that story.
- **Docs that stress caveats and “improve with less”** – the *tone* and emphasis are part of the design, so the repo doesn’t oversell the standard parts.

**Bottom line:** The recommended pipeline is **not** different from best practice elsewhere. What’s different is the **channel-theory story**, the **capacity-based weighting** option, and the **explicit packaging** of theory + caveats + minimal default path. Use the repo for that framing and those options; don’t expect the strict train/val/test setup itself to be novel.

**Is that enough to set this app apart?** It depends on the bar. For a learning project or a clear narrative ("ensembles as channels"), it can be. For a widely adopted library or a research-grade contribution, usually not by itself. To stand out more you'd want at least one of: **evidence** (e.g. when capacity weighting reliably beats accuracy weighting, with proper validation and write-up); a **concrete use case** (e.g. theory-as-channel for compression or robust ML under noise, with worked examples); or **adoption** (e.g. used or cited by ML-Toolbox or others). Right now the repo is "theory-as-channel framing + one theory-driven combiner + rigor and caveats" – enough to be coherent and different in story, not enough to claim a unique technical edge without that extra step.

## Using the distinctive parts elsewhere

Yes. The same ideas can be reused in other settings:

| What's different here | Use elsewhere for |
|------------------------|--------------------|
| **Channel framing** (noisy channels → decode) | **Sensor fusion**, **multi-annotator labels**, **crowdsourcing**, **multi-expert panels**, **federated or aggregated predictions**, **multi-view learning**. Any setting where several noisy sources produce a "message" you want to recover. |
| **Capacity-weighted combination** \(\log_2(1+p/(1-p))\) | **Combining forecasters** (weight by past accuracy), **combining retrievers or rankers**, **combining APIs or models from different domains**. Whenever you have a per-source "success rate" and want a theory-driven weight. |
| **Error correction / majority decoding** | **Redundant sensors**, **multiple models or APIs**, **multi-annotator consensus**, **robustness to one bad channel**. Same decoding idea; the "channel" can be humans, sensors, or models. |
| **Information-theory bundle** (entropy, MI, KL) | **Feature selection**, **representation learning**, **fairness/independence**, **interpretability**, **clustering**. Standard tools; this repo just bundles them with the channel story. |
| **Caveats + "improve with less"** | **Any ML or data project**. The mindset (validate properly, don't leak test, prefer fewer well-validated methods) applies everywhere. |

So the distinctive parts are **building blocks**: use the channel view and capacity weighting in other pipelines (e.g. for compression, robust ML, or multi-source fusion) without limiting the repo to "just ensemble voting."

## Best practice: what to use, what to avoid, improve with less

**Incorporate**

- **One validated pipeline** – Use `run_recommended.py`: train base models on a train set, compute weights from a **validation set only**, report on a **test set** once. That gives unbiased estimates and a single “recommended” path.
- **Few combiners by default** – In the recommended path, use only: unweighted majority, weighted majority (weights = val accuracies), and optionally soft weighted. Add more methods only when you have a clear need and proper validation.
- **Core API** – Keep `ErrorCorrectingPredictions`, `channel_capacity`, and the idea of ensemble-as-channel. They are the thesis of the app.
- **Caveats** – Keep the caveats section and the reminder that illustrative numbers can be biased if weights or method choice use the test set.

**Take out or de-emphasize**

- **Weights from test** – Do not compute per-model accuracies or any weights from the test set. Use a validation set (or out-of-fold estimates) so the app can “continue to improve” without misleading numbers.
- **Many methods in the default path** – Treat entropy-weighted, capacity-weighted, diversity-weighted, median/trimmed, stacking as **optional extras** (e.g. in `run_real_world_tests.py` for exploration). The recommended run should not depend on trying all of them and picking the best.
- **Unnecessary complexity** – If a method rarely beats the simple baseline in your setting, don’t add it to the default pipeline. Keep the library surface small for the main use case.

**Improve with less**

- The app improves when you **improve data, base models, or validation**, not when you add more ensemble tricks. Use:
  - **Better base models** or more training data.
  - **Stricter validation**: train/val/test, or CV with a single method chosen on val.
  - **Multiple seeds** (e.g. `python run_recommended.py --multi`) to see variance.
- Keep the “recommended” surface small: one script, three combiners, weights from val, report on test. New ideas (e.g. new theory or methods) can live in `ensemble_extras` and in exploratory scripts without cluttering the default path.

## Idea

- **Ensemble as channel:** Send the same "message" (true label) through several noisy channels (models). Majority vote decodes more reliably than any single channel.
- **Compression as channel:** Teacher→student knowledge transfer is limited by channel capacity \( C = B \log_2(1 + S/N) \).

## Would this be used to train LLMs?

**No.** This repo is not for training LLMs. It has no transformer code, no LLM training loop, and no support for the kind of data or compute used to train or fine-tune large language models. It is for **combining predictions** from already-trained models (e.g. small classifiers) using voting and information/communication theory.

Where the ideas might still touch LLM work:

- **Ensembling LLM outputs:** If you have several LLMs (or several samples from one), you could use the same combiners (majority vote, capacity-weighted, etc.) to merge their outputs at **inference** time. That’s using the repo to combine answers, not to train the LLMs.
- **Distillation as a concept:** The “compression as channel” story (teacher→student, capacity limits) is a useful **way of thinking** about distilling a big model into a smaller one. This repo doesn’t implement LLM distillation; it only offers the channel-capacity lens and small combiners.

So: **not for training LLMs.** Optionally useful for combining LLM outputs or for the theoretical framing around distillation.

## Code source

Code is taken from **ML-Toolbox**:

- `ml_toolbox.textbook_concepts.communication_theory` → `theory_as_channel.communication_theory`
- `ml_toolbox.textbook_concepts.information_theory` → `theory_as_channel.information_theory`

## Setup

Create and use a virtual environment (recommended):

```bash
# Create venv
python -m venv venv

# Activate: Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate: Windows (cmd) or macOS/Linux
# venv\Scripts\activate.bat   (Windows cmd)
# source venv/bin/activate     (macOS/Linux)

# Install dependencies
pip install -r requirements.txt
```

## Run

**Recommended (train/val/test, weights from val, report on test):**

```bash
python run_recommended.py
python run_recommended.py --multi   # 3 seeds, mean +/- std
```

**Concept example (ensemble + channel capacity + teacher–student):**

```bash
python example_theory_as_channel.py
```

**Viability test:**

```bash
python run.py
```

**Exploratory (many methods; weights from test – illustrative only):**

```bash
python run_real_world_tests.py
```

## If you're not familiar with pipelines – one simple guide

You don't have to understand everything to make good choices. Here's the minimum:

- **Pipeline** here just means: train some models, combine their predictions (e.g. by voting), then report an accuracy number. The only "decision" is how we combine and how we compute that number so it's honest.
- **Train / val / test** means we split the data into three parts. We train on the first part, we use the second part only to choose weights (no peeking at the third). We report the final number only on the third part. That way the reported number isn't inflated by using the same data for choices and for evaluation.
- **What to run:**  
  - For **numbers you can trust** (no test leakage): run **`python run_recommended.py`**. That script does the train/val/test split and uses only val for weights. You don't have to pick among methods – it uses a small, fixed set (majority, weighted majority, soft).
  - For **seeing the theory-as-channel idea** with minimal setup: run **`python example_theory_as_channel.py`** or **`python run.py`**.
  - **`run_real_world_tests.py`** is for **exploration** (lots of methods, comparison tables). Its numbers are illustrative because weights are computed from the same set we evaluate on; use it to play with ideas, not to claim "method X is better."
- **You don't need to choose** among all the methods in the repo. The recommended script already makes a good default choice. When you're more comfortable, you can try other combiners or add your own data; until then, sticking to `run_recommended.py` is enough.

## Unusual / underused methods (for toolbox or elsewhere)

In `theory_as_channel.ensemble_extras` (and wired into `run_real_world_tests.py`):

| Method | Idea | Use case |
|--------|------|----------|
| **Entropy-weighted soft** | Weight each model by 1/(1+mean entropy); confident (low-entropy) models count more | When some models are overconfident wrong |
| **Capacity-weighted** | Weight by Shannon-style C = log2(1 + p/(1-p)) from accuracy | Theory-as-channel: better channels get more weight |
| **Median-of-probas** | Median probability per class across models, then argmax | Robust to one badly calibrated or adversarial channel |
| **Trimmed-mean soft** | Drop min/max proba per class across models, average the rest | Robust to a few extreme channels |
| **Diversity-weighted majority** | Weight = accuracy × (1 − agreement with others); accurate disagreeing models upweighted | When “diverse” correct voters break ties |
| **Simple stacking** | Meta-learner (e.g. LogisticRegression) on base model probas | Learn how to combine channels; ML-Toolbox has full CV stacking |

## Caveats: can more theory/methods give biased or misleading results?

Yes. Adding many methods or theory-driven tweaks can make results **look good but be wrong** in several ways:

1. **Selection bias** – We try many ensemble variants (majority, weighted, soft, entropy-weighted, capacity-weighted, stacking, etc.). Some will beat the best single model on this run by chance. Reporting only the best or highlighting “wins” overstates benefit. **Mitigation:** Report all methods in the summary table; use a **held-out test set** (or nested CV) and **choose the ensemble strategy once** on a dev set, then evaluate once on test.

2. **Test-set reuse** – In `run_real_world_tests.py`, weights (e.g. per-model accuracy, capacity weights) are computed from the **same test set** we evaluate on. That leaks test information into the “method” and can inflate scores. **Mitigation:** Compute weights from a **validation set** (or out-of-fold estimates); evaluate only on a test set that was never used for any choice or weight.

3. **Overfitting the “theory”** – Fitting a story (e.g. “capacity weighting helps”) after seeing the numbers can be post-hoc. The theory might be right in general but the specific gain on one dataset can be noise. **Mitigation:** Pre-register which methods you compare; replicate on multiple datasets and report variance; prefer simple baselines (e.g. unweighted majority) unless gains are stable.

4. **Goodhart effects** – Optimizing a proxy (e.g. accuracy on one benchmark) can improve that number while hurting real-world performance (distribution shift, different loss, fairness). **Mitigation:** Validate on multiple datasets and, when possible, on downstream or out-of-distribution data.

**Bottom line:** The numbers in `run_real_world_tests.py` are **illustrative** on a single split. For any claim that “method X is better,” use proper validation (e.g. cross-validation, separate val/test, multiple seeds) and report uncertainty.

## API (from `theory_as_channel`)

- **Communication:** `ErrorCorrectingPredictions`, `channel_capacity`, `signal_to_noise_ratio`, `NoiseRobustModel`, `RobustMLProtocol`
- **Information theory:** `entropy`, `mutual_information`, `kl_divergence`, `information_gain`, and class wrappers `Entropy`, `MutualInformation`, `KLDivergence`, `InformationGain`
- **Unusual ensembles:** `entropy_weighted_soft_weights`, `capacity_weighted_weights`, `soft_vote_median`, `soft_vote_trimmed_mean`, `diversity_weighted_weights`, `SimpleStackingEnsemble`, `build_meta_features_from_probas`, `build_meta_features_from_predictions`
