# Theory-as-Channel

Treat ensemble/student predictions as a **communication channel**: use error correction and channel capacity (from ML-Toolbox).

## Idea

- **Ensemble as channel:** Send the same "message" (true label) through several noisy channels (models). Majority vote decodes more reliably than any single channel.
- **Compression as channel:** Teacher→student knowledge transfer is limited by channel capacity \( C = B \log_2(1 + S/N) \).

## Code source

Code is taken from **ML-Toolbox**:

- `ml_toolbox.textbook_concepts.communication_theory` → `theory_as_channel.communication_theory`
- `ml_toolbox.textbook_concepts.information_theory` → `theory_as_channel.information_theory`

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Full example (ensemble + channel capacity + teacher–student):**

```bash
python example_theory_as_channel.py
```

**Viability test:**

```bash
python run.py
```

## API (from `theory_as_channel`)

- **Communication:** `ErrorCorrectingPredictions`, `channel_capacity`, `signal_to_noise_ratio`, `NoiseRobustModel`, `RobustMLProtocol`
- **Information theory:** `entropy`, `mutual_information`, `kl_divergence`, `information_gain`, and class wrappers `Entropy`, `MutualInformation`, `KLDivergence`, `InformationGain`
